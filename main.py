import os
import json
import logging
from typing import Optional
from summary import summary_meeting_agent, create_initial_state, SummaryConfig, performance_monitor
from utils import create_documents_from_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_transcript(file_path: str) -> str:
    """Read transcript content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return ""
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def save_summary_to_file(summary_result, output_path: str):
    """Save the summary result to a JSON file."""
    try:
        # Convert Pydantic model to dict if needed
        if hasattr(summary_result, 'dict'):
            summary_dict = summary_result.dict()
        else:
            summary_dict = summary_result
            
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(summary_dict, file, ensure_ascii=False, indent=2)
        print(f"Summary saved to {output_path}")
    except Exception as e:
        print(f"Error saving summary to {output_path}: {e}")

def process_transcript(transcript_path: str, language: str = "English", 
                      output_dir: str = "output", 
                      config: Optional[SummaryConfig] = None) -> Optional[dict]:
    """Process a single transcript file and generate summary with enhanced error handling."""
    
    # Read transcript content
    logger.info(f"Reading transcript from: {transcript_path}")
    transcript_content = read_transcript(transcript_path)
    
    if not transcript_content.strip():
        logger.error("Empty or invalid transcript content.")
        return None
    
    try:
        # Create documents from text using the utility function
        logger.info("Creating document chunks...")
        documents = create_documents_from_text(transcript_content)
        
        if not documents:
            logger.error("No documents created from transcript content.")
            return None
        
        # Extract content from documents
        contents = [doc.page_content for doc in documents]
        logger.info(f"Created {len(contents)} document chunks")
        
        # Create initial state using the new function
        logger.info("Creating initial state...")
        initial_state = create_initial_state(contents, language)
        
        # Process with the summary agent
        logger.info("Processing with summary agent...")
        result = summary_meeting_agent.invoke(initial_state)
        
        # Check for errors in result
        if result.get("errors"):
            logger.warning(f"Processing completed with errors: {result['errors']}")
        
        # Extract the final output
        if result and "output" in result and result["output"]:
            summary_output = result["output"]
            logger.info("Summary generation completed successfully!")
            
            # Log performance metrics
            if "metadata" in result:
                metadata = result["metadata"]
                if "total_processing_time" in metadata:
                    performance_monitor.record_metric("processing_time", metadata["total_processing_time"])
                if "agenda_count" in metadata:
                    performance_monitor.record_metric("agenda_count", metadata["agenda_count"])
                if "action_item_count" in metadata:
                    performance_monitor.record_metric("action_item_count", metadata["action_item_count"])
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            base_filename = os.path.splitext(os.path.basename(transcript_path))[0]
            output_filename = f"{base_filename}_summary.json"
            output_path = os.path.join(output_dir, output_filename)
            
            # Prepare enhanced output with metadata
            enhanced_output = {
                "summary": summary_output.dict() if hasattr(summary_output, 'dict') else summary_output,
                "metadata": result.get("metadata", {}),
                "processing_info": {
                    "source_file": transcript_path,
                    "language": language,
                    "chunk_count": len(contents),
                    "has_errors": bool(result.get("errors")),
                    "errors": result.get("errors", [])
                }
            }
            
            # Save summary to file
            save_summary_to_file(enhanced_output, output_path)
            
            return summary_output
        else:
            logger.error("No output generated from summary agent.")
            return None
            
    except Exception as e:
        logger.error(f"Error during summary generation: {e}")
        return None

def process_all_transcripts(transcript_dir: str = "transcript", language: str = "English", 
                           output_dir: str = "output", config: Optional[SummaryConfig] = None):
    """Process all transcript files in the specified directory with enhanced monitoring."""
    
    if not os.path.exists(transcript_dir):
        logger.error(f"Transcript directory {transcript_dir} not found.")
        return
    
    # Get all txt files in the transcript directory
    transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith('.txt')]
    
    if not transcript_files:
        logger.warning(f"No transcript files found in {transcript_dir}")
        return
    
    logger.info(f"Found {len(transcript_files)} transcript files to process.")
    
    successful_summaries = 0
    failed_summaries = 0
    
    for transcript_file in sorted(transcript_files):
        transcript_path = os.path.join(transcript_dir, transcript_file)
        logger.info(f"{'='*50}")
        logger.info(f"Processing: {transcript_file}")
        logger.info(f"{'='*50}")
        
        try:
            result = process_transcript(transcript_path, language, output_dir, config)
            
            if result:
                successful_summaries += 1
                logger.info(f"✓ Successfully processed {transcript_file}")
            else:
                failed_summaries += 1
                logger.error(f"✗ Failed to process {transcript_file}")
        except Exception as e:
            failed_summaries += 1
            logger.error(f"✗ Exception processing {transcript_file}: {e}")
    
    # Print performance summary
    logger.info(f"{'='*50}")
    logger.info("Processing completed!")
    logger.info(f"Successfully processed: {successful_summaries}/{len(transcript_files)} files")
    logger.info(f"Failed: {failed_summaries}/{len(transcript_files)} files")
    logger.info(f"Output directory: {output_dir}")
    
    # Print performance metrics
    perf_summary = performance_monitor.get_summary()
    if perf_summary:
        logger.info("Performance Metrics:")
        for metric, stats in perf_summary.items():
            logger.info(f"  {metric}: avg={stats['average']:.2f}s, "
                       f"min={stats['min']:.2f}s, max={stats['max']:.2f}s")
    
    logger.info(f"{'='*50}")

def main():
    """Main function to run the meeting summarization system with enhanced configuration."""
    
    logger.info("Meeting Summarization System")
    logger.info("="*40)
    
    # Configuration settings
    TRANSCRIPT_DIR = "transcript"
    OUTPUT_DIR = "output"
    LANGUAGE = "Chinese"  # Change to "English" if needed
    
    # Create custom configuration for better performance
    custom_config = SummaryConfig(
        model_name="mistral-small3.2:24b",
        temperature=0.1,  # Lower temperature for more consistent results
        max_retries=3,
        retry_delay=1.0,
        timeout=180.0  # 3 minutes timeout
    )
    
    # Check if transcript directory exists
    if not os.path.exists(TRANSCRIPT_DIR):
        logger.error(f"Transcript directory '{TRANSCRIPT_DIR}' not found.")
        logger.info("Please ensure you have a 'transcript' directory with .txt files.")
        return
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {custom_config.model_name}")
    logger.info(f"  Temperature: {custom_config.temperature}")
    logger.info(f"  Language: {LANGUAGE}")
    logger.info(f"  Max Retries: {custom_config.max_retries}")
    logger.info(f"  Timeout: {custom_config.timeout}s")
    
    # Process all transcripts
    try:
        process_all_transcripts(TRANSCRIPT_DIR, LANGUAGE, OUTPUT_DIR, custom_config)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")

if __name__ == "__main__":
    main()
