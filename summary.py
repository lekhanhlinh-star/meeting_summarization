import logging
import time
from typing import List, Literal, TypedDict, Optional, Dict, Any
from dataclasses import dataclass

from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from openai import base_url
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
    base_url: str = "http://localhost:11434"
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
            timeout=model_config.timeout,
            base_url= model_config.base_url,
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
    output: Optional[SummarizeResponse]
    metadata: Dict[str, Any]
    errors: List[str]
    processing_time: float
    retry_count: int

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


@retry_on_failure
def generate_initial_summary(state: State, config: RunnableConfig) -> Dict[str, Any]:
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
        
        # Generate summary
        summary = initial_summary_chain.invoke(
            {"context": contents[0], "language": language},
            config,
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


def refine_summary(state: State, config: RunnableConfig):
    content = state["contents"][state["index"]]
    language = state["language"]

    summary = refine_summary_chain.invoke(
        {"existing_answer": state["summary"], "context": content,"language": language},
        config,
    )

    return {"summary": summary, "index": state["index"] + 1}

def should_refine(state: State) -> Literal["refine_summary","parse_json"]:
    if state["index"] >= len(state["contents"]):
        return "parse_json"
    else:
        return "refine_summary"
    
def parse_json_output(state: State):
    summary = state["summary"]
    summary = chain_parse.invoke(
        {"summary":summary,
         "language": state["language"]
         },
        
    )
    return {"output":summary}

graph = StateGraph(State)
graph.add_node("generate_initial_summary", generate_initial_summary)
graph.add_node("refine_summary", refine_summary)
graph.add_node("parse_json", parse_json_output)

graph.add_edge(START, "generate_initial_summary")
graph.add_conditional_edges("generate_initial_summary", should_refine)
graph.add_conditional_edges("refine_summary", should_refine)
graph.add_edge("refine_summary", "parse_json")  # Final step
graph.set_finish_point("parse_json")
summary_meeting_agent = graph.compile(debug=True)
