import logging
import time
from typing import List, Literal, TypedDict, Optional, Dict, Any
from dataclasses import dataclass

from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, validator
from prompt import PARSER_PROMPT, REFINE_PROMPT, SUMMARIZE_PROMPT, SummarizeResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SummaryConfig:
    """Configuration class for summary generation."""
    model_name: str = "mistral-small3.2:24b"
    temperature: float = 0.1
    max_retries: int = 3
    retry_delay: float = 1.0
    max_tokens: Optional[int] = None
    timeout: float = 120.0
    
    def __post_init__(self):
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        if self.max_retries < 1:
            raise ValueError("Max retries must be at least 1")


# Global configuration
config = SummaryConfig()

# Initialize LLM with configuration
def create_llm(model_config: SummaryConfig) -> ChatOllama:
    """Create and configure the ChatOllama instance."""
    try:
        llm = ChatOllama(
            model=model_config.model_name,
            temperature=model_config.temperature,
            timeout=model_config.timeout
        )
        # Test the connection
        test_response = llm.invoke("Hello")
        logger.info(f"LLM initialized successfully with model: {model_config.model_name}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

llm = create_llm(config)
parser = PydanticOutputParser(pydantic_object=SummarizeResponse)
initial_summary_chain = SUMMARIZE_PROMPT | llm | StrOutputParser()
refine_summary_chain = REFINE_PROMPT | llm | StrOutputParser()
chain_parse = PARSER_PROMPT | llm | parser

class State(TypedDict):
    """Enhanced state with error handling and metadata."""
    contents: List[str]
    index: int
    summary: str
    language: str
    output: Optional[Any]  # Changed from SummarizeResponse to Any for compatibility
    metadata: Optional[Dict[str, Any]]
    errors: Optional[List[str]]
    processing_time: Optional[float]
    retry_count: Optional[int]

def validate_state(state: State) -> bool:
    """Validate state integrity."""
    if not state.get("contents"):
        return False
    if state.get("index", 0) < 0:
        return False
    if not state.get("language"):
        return False
    return True

def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
        raise last_exception
    return wrapper


def generate_initial_summary(state: State, config: RunnableConfig = None) -> Dict[str, Any]:
    """Generate initial summary with error handling and validation."""
    start_time = time.time()
    
    try:
        # Validate input state
        if not validate_state(state):
            raise ValueError("Invalid state provided to generate_initial_summary")
        
        language = state["language"]
        contents = state["contents"]
        
        if not contents or len(contents) == 0:
            raise ValueError("No content provided for summarization")
        
        logger.info(f"Generating initial summary for content of length: {len(contents[0])}")
        
        # Generate summary with optional config
        if config:
            summary = initial_summary_chain.invoke(
                {"context": contents[0], "language": language},
                config,
            )
        else:
            summary = initial_summary_chain.invoke(
                {"context": contents[0], "language": language}
            )
        
        # Validate summary output
        if not summary or not summary.strip():
            raise ValueError("Empty summary generated")
        
        processing_time = time.time() - start_time
        logger.info(f"Initial summary generated successfully in {processing_time:.2f}s")
        
        return {
            "summary": summary.strip(),
            "index": 1,
            "processing_time": processing_time,
            "metadata": {
                **state.get("metadata", {}),
                "initial_content_length": len(contents[0]),
                "summary_length": len(summary.strip()),
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in generate_initial_summary: {e}")
        return {
            "errors": state.get("errors", []) + [f"Initial summary error: {str(e)}"],
            "retry_count": state.get("retry_count", 0) + 1
        }


def refine_summary(state: State, config: RunnableConfig = None) -> Dict[str, Any]:
    """Refine summary with enhanced error handling and progress tracking."""
    start_time = time.time()
    
    try:
        # Validate state
        if not validate_state(state):
            raise ValueError("Invalid state provided to refine_summary")
            
        current_index = state["index"]
        contents = state["contents"]
        
        if current_index >= len(contents):
            raise ValueError(f"Index {current_index} out of range for contents length {len(contents)}")
        
        content = contents[current_index]
        language = state["language"]
        existing_summary = state.get("summary", "")
        
        if not existing_summary:
            raise ValueError("No existing summary to refine")
        
        logger.info(f"Refining summary with chunk {current_index + 1}/{len(contents)}")
        
        # Refine summary with optional config
        if config:
            refined_summary = refine_summary_chain.invoke(
                {
                    "existing_answer": existing_summary,
                    "context": content,
                    "language": language
                },
                config,
            )
        else:
            refined_summary = refine_summary_chain.invoke(
                {
                    "existing_answer": existing_summary,
                    "context": content,
                    "language": language
                }
            )
        
        # Validate refined summary
        if not refined_summary or not refined_summary.strip():
            logger.warning("Empty refined summary, keeping existing summary")
            refined_summary = existing_summary
        
        processing_time = time.time() - start_time
        progress = (current_index + 1) / len(contents) * 100
        
        logger.info(f"Summary refined for chunk {current_index + 1}/{len(contents)} "
                   f"({progress:.1f}%) in {processing_time:.2f}s")
        
        return {
            "summary": refined_summary.strip(),
            "index": current_index + 1,
            "processing_time": state.get("processing_time", 0) + processing_time,
            "metadata": {
                **state.get("metadata", {}),
                f"chunk_{current_index}_processed": True,
                f"chunk_{current_index}_length": len(content),
                "progress_percent": progress,
                "last_update": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in refine_summary: {e}")
        return {
            "errors": state.get("errors", []) + [f"Refine summary error: {str(e)}"],
            "retry_count": state.get("retry_count", 0) + 1
        }

def should_refine(state: State) -> Literal["refine_summary", "parse_json", "handle_error"]:
    """Enhanced routing logic with error handling."""
    
    # Check for errors first
    if state.get("errors") and state.get("retry_count", 0) >= config.max_retries:
        logger.error("Max retries exceeded, routing to error handling")
        return "handle_error"
    
    # Check if we need to retry due to errors
    if state.get("errors") and state.get("retry_count", 0) < config.max_retries:
        logger.info(f"Retrying due to errors (attempt {state.get('retry_count', 0) + 1})")
        return "refine_summary"
    
    # Normal flow
    current_index = state.get("index", 0)
    total_contents = len(state.get("contents", []))
    
    if current_index >= total_contents:
        logger.info("All chunks processed, moving to JSON parsing")
        return "parse_json"
    else:
        logger.info(f"Processing chunk {current_index + 1}/{total_contents}")
        return "refine_summary"

def handle_error(state: State) -> Dict[str, Any]:
    """Handle errors and provide fallback summary."""
    logger.error("Handling errors in summary generation")
    
    errors = state.get("errors", [])
    summary = state.get("summary", "")
    
    # If we have at least a partial summary, try to salvage it
    if summary and summary.strip():
        logger.info("Using partial summary as fallback")
        fallback_output = SummarizeResponse(
            summary="Summary generation encountered errors but partial content was processed: " + summary[:500],
            agendas=[],
            action_items=[]
        )
    else:
        logger.warning("No usable summary available, creating error response")
        fallback_output = SummarizeResponse(
            summary="Failed to generate summary due to processing errors.",
            agendas=[],
            action_items=[]
        )
    
    return {
        "output": fallback_output,
        "metadata": {
            **state.get("metadata", {}),
            "has_errors": True,
            "error_count": len(errors),
            "errors": errors
        }
    }
    
def parse_json_output(state: State, config: RunnableConfig = None) -> Dict[str, Any]:
    """Parse final summary to JSON with enhanced validation."""
    start_time = time.time()
    
    try:
        summary = state.get("summary", "")
        language = state.get("language", "English")
        
        if not summary or not summary.strip():
            raise ValueError("No summary available for JSON parsing")
        
        logger.info("Parsing summary to structured JSON format")
        
        # Parse to structured format with optional config
        if config:
            parsed_output = chain_parse.invoke(
                {
                    "summary": summary,
                    "language": language
                },
                config
            )
        else:
            parsed_output = chain_parse.invoke(
                {
                    "summary": summary,
                    "language": language
                }
            )
        
        # Validate parsed output
        if not parsed_output:
            raise ValueError("No output from JSON parsing")
        
        # Ensure we have a valid response object
        if not hasattr(parsed_output, 'summary') and not isinstance(parsed_output, dict):
            raise ValueError("Failed to parse output to proper format")
        
        # Quality checks
        summary_text = getattr(parsed_output, 'summary', '') if hasattr(parsed_output, 'summary') else parsed_output.get('summary', '')
        agendas = getattr(parsed_output, 'agendas', []) if hasattr(parsed_output, 'agendas') else parsed_output.get('agendas', [])
        action_items = getattr(parsed_output, 'action_items', []) if hasattr(parsed_output, 'action_items') else parsed_output.get('action_items', [])
        
        if not summary_text or len(str(summary_text).strip()) < 10:
            logger.warning("Generated summary is very short")
        
        if not agendas:
            logger.warning("No agenda items found in summary")
        
        if not action_items:
            logger.warning("No action items found in summary")
        
        processing_time = time.time() - start_time
        total_time = state.get("processing_time", 0) + processing_time
        
        logger.info(f"Summary parsed successfully in {processing_time:.2f}s "
                   f"(total processing time: {total_time:.2f}s)")
        
        return {
            "output": parsed_output,
            "processing_time": total_time,
            "metadata": {
                **state.get("metadata", {}),
                "parsing_time": processing_time,
                "total_processing_time": total_time,
                "agenda_count": len(agendas),
                "action_item_count": len(action_items),
                "final_summary_length": len(str(summary_text)),
                "completed_at": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in parse_json_output: {e}")
        return {
            "errors": state.get("errors", []) + [f"JSON parsing error: {str(e)}"],
            "retry_count": state.get("retry_count", 0) + 1
        }

# Create enhanced graph with error handling
def create_summary_graph() -> StateGraph:
    """Create and configure the summary processing graph."""
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("generate_initial_summary", generate_initial_summary)
    graph.add_node("refine_summary", refine_summary)
    graph.add_node("parse_json", parse_json_output)
    graph.add_node("handle_error", handle_error)
    
    # Add edges
    graph.add_edge(START, "generate_initial_summary")
    graph.add_conditional_edges("generate_initial_summary", should_refine)
    graph.add_conditional_edges("refine_summary", should_refine)
    graph.add_edge("parse_json", END)
    graph.add_edge("handle_error", END)
    
    return graph

# Initialize the graph
graph = create_summary_graph()
summary_meeting_agent = graph.compile(debug=False)  # Set to True for debugging

def create_initial_state(contents: List[str], language: str = "English") -> State:
    """Create initial state with proper structure and validation."""
    if not contents:
        raise ValueError("Contents cannot be empty")
    
    if not language:
        raise ValueError("Language must be specified")
    
    return State(
        contents=contents,
        index=0,
        summary="",
        language=language,
        output=None,
        metadata={
            "created_at": time.time(),
            "total_chunks": len(contents),
            "total_content_length": sum(len(c) for c in contents)
        },
        errors=[],
        processing_time=0.0,
        retry_count=0
    )

# Utility functions for external use
def summarize_meeting(contents: List[str], language: str = "English", 
                     custom_config: Optional[SummaryConfig] = None) -> Dict[str, Any]:
    """
    High-level function to summarize meeting contents.
    
    Args:
        contents: List of text chunks to summarize
        language: Output language (English, Chinese, etc.)
        custom_config: Optional custom configuration
        
    Returns:
        Dictionary containing the summary result and metadata
    """
    if custom_config:
        global config
        config = custom_config
        # Reinitialize LLM with new config
        global llm, initial_summary_chain, refine_summary_chain, chain_parse
        llm = create_llm(config)
        initial_summary_chain = SUMMARIZE_PROMPT | llm | StrOutputParser()
        refine_summary_chain = REFINE_PROMPT | llm | StrOutputParser()
        chain_parse = PARSER_PROMPT | llm | parser
    
    # Create initial state
    initial_state = create_initial_state(contents, language)
    
    # Process with agent
    result = summary_meeting_agent.invoke(initial_state)
    
    return result

# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name: str) -> Optional[float]:
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            metric: {
                "count": len(values),
                "average": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0
            }
            for metric, values in self.metrics.items()
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()
