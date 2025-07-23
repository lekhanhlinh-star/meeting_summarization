# Meeting Summarization Research

An AI-powered meeting summarization system that automatically analyzes and synthesizes meeting content into structured, clear, and actionable insights using advanced language models and prompt engineering.

## 🚀 Key Features

### Structured Summaries
- **General Overview**: Concise description of main topics and key decisions
- **Agenda Items**: Detailed discussion points for each agenda topic
- **Action Items**: Specific actionable tasks agreed upon during the meeting

### Intelligent Processing
- **Automatic Chunking**: Splits long transcripts into manageable segments
- **Iterative Refinement**: Updates summaries with new content progressively
- **Multi-language Support**: Default Chinese output with customizable language settings
- **JSON Export**: Structured data output for easy integration

## 📁 Project Structure

```
Meeting_Summarization_Research/
├── main.py                 # Main application entry point
├── summary.py              # Summary processing logic and LangGraph workflow
├── prompt.py               # Prompt templates and Pydantic models
├── utils.py                # Text processing utility functions
├── config.py               # System configuration (currently unused)
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── transcript/            # Sample transcript files directory
│   ├── transcript_1.txt
│   ├── transcript_2.txt
│   ├── transcript_3.txt
│   ├── transcript_4.txt
│   └── transcript_5.txt
└── output/                # Generated results directory (auto-created)
```

## 🛠️ Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Ollama Server
Ensure Ollama is running with the required model:
```bash
ollama serve
ollama pull mistral-small3.2:24b
```

### 3. Run the Application
```bash
python main.py
```

### 4. Process Custom Transcripts
Place your `.txt` meeting transcript files in the `transcript/` directory and run the application.

## ⚙️ Configuration

### Model Configuration (in summary.py)
```python
class SummaryConfig:
    model_name: str = "mistral-small3.2:24b"
    temperature: float = 0.1
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    max_retries: int = 3
```

### Prompt Customization
Modify templates in `prompt.py` to customize summarization behavior:
- `SUMMARIZE_PROMPT`: Initial summary generation template
- `REFINE_PROMPT`: Summary refinement template
- `PARSER_PROMPT`: Structured data extraction template

### Output Format
Results are saved as JSON with the following structure:
```json
{
  "summary": "General meeting overview...",
  "agendas": [
    {
      "name": "Agenda item name",
      "points": ["Discussion point 1", "Discussion point 2"]
    }
  ],
  "action_items": [
    {
      "task": "Specific action to be taken"
    }
  ]
}
```

## 📋 System Requirements

- **Python**: 3.9+
- **Ollama**: With `mistral-small3.2:24b` model
- **RAM**: Minimum 8GB (16GB recommended for large models)

### Core Dependencies
- `langchain` & `langgraph`: AI workflow framework
- `langchain-ollama`: Ollama integration
- `pydantic`: Data validation and serialization

## 🔧 Troubleshooting

### Ollama Connection Issues
- Verify Ollama server is running: `ollama list`
- Check base_url in config: `http://localhost:11434`
- Ensure model is downloaded: `ollama pull mistral-small3.2:24b`

### Timeout Errors
- Increase `timeout` value in `SummaryConfig`
- Reduce chunk size for large transcripts
- Check system resources and model performance

### Memory Issues
- Decrease `chunk_size` in processing config
- Ensure sufficient RAM for the model
- Consider using a smaller model variant

## 🏗️ Architecture

The system uses a graph-based workflow powered by LangGraph:

1. **Text Chunking**: Splits large transcripts into processable segments
2. **Initial Summarization**: Generates first-pass summary using AI
3. **Iterative Refinement**: Improves summary by processing additional chunks
4. **Structure Extraction**: Parses final summary into structured JSON format

## 📊 Performance Monitoring

The system includes built-in performance monitoring:
- Processing time tracking
- Error handling and retry mechanisms
- Token usage monitoring
- Quality validation checks

## 🎯 Use Cases

- **Corporate Meetings**: Board meetings, team standups, project reviews
- **Academic Conferences**: Research presentations, panel discussions
- **Client Calls**: Sales meetings, consultation sessions
- **Training Sessions**: Workshops, seminars, educational content

## 🔄 Workflow

1. Load transcript from file
2. Split text into manageable chunks
3. Generate initial summary using LLM
4. Refine summary with additional context
5. Extract structured data (agendas, action items)
6. Save results in JSON format



## 📄 License

MIT License - see LICENSE file for details.

