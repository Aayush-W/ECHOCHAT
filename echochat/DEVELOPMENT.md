# Development Setup

Quick setup guide for developers wanting to work on EchoChat.

## Prerequisites

- Python 3.8+ (`python --version`)
- Git (`git --version`)
- Ollama running locally (download from [ollama.ai](https://ollama.ai))

## Quick Setup (5 minutes)

### 1. Clone & Navigate

```bash
git clone https://github.com/yourusername/echochat.git
cd echochat
```

### 2. Virtual Environment

**Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Runtime
pip install -r requirements.txt

# With GPU/training support (optional)
pip install -r requirements.txt -r requirements-train.txt

# Development (testing, linting)
pip install pytest pytest-cov
```

### 4. Prepare Sample Data

```bash
# Copy a sample chat to the expected location
# You'll need a WhatsApp chat export (.txt) at:
# data/uploads/chat.txt
```

### 5. Start Ollama

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull the default model
ollama pull mistral
```

### 6. Run the Pipeline

```bash
python -m backend.main
```

## Development Workflow

### Running Different Interfaces

```bash
# CLI Interface
python -m backend.cli

# API Server (http://localhost:5000)
python -m backend.api

# Full Pipeline (all 5 phases)
python -m backend.main
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest backend/test_main.py -v

# Run with coverage report
pytest --cov=backend --cov-report=html
```

### Code Formatting (Optional but Recommended)

```bash
# Install formatters
pip install black flake8

# Format code
black backend/

# Check style
flake8 backend/ --max-line-length=100
```

## Project Structure Reference

```
backend/
├── main.py                 # Main orchestration (START HERE)
├── config.py               # Configuration constants
├── chat_parser.py          # WhatsApp parsing
├── dataset_builder.py      # Dataset creation
├── personality_analyzer.py # Style analysis
├── memory_store.py         # Semantic memory
├── responder.py            # LLM interface
├── cli.py                  # CLI interface
├── api.py                  # REST API
└── utils.py                # Helper functions
```

## Common Development Tasks

### Add a New Analysis Feature

1. Add method to `PersonalityAnalyzer` class
2. Call it in the `analyze()` method
3. Update `config.py` if needed
4. Test with `python -m backend.main`

### Modify LLM Behavior

Edit `backend/config.py`:
- `LLM_TEMPERATURE` (0=deterministic, 1=creative)
- `LLM_MAX_TOKENS` (response length)
- `LLM_MODEL_NAME` (llama2, neural-chat, orca-mini, etc.)

### Add API Endpoint

1. Open `backend/api.py`
2. Define request/response models (Pydantic)
3. Create route with `@app.get()` or `@app.post()`
4. Test with curl or Postman

### Debug Response Generation

Add verbose mode:
```python
from backend.responder import Responder

responder = Responder(...)
result = responder.generate_response("Your message", verbose=True)
print(result)  # Shows prompt, temperature, tokens used, etc.
```

## Troubleshooting

### Ollama Connection Failed

```bash
# Check if running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

### Memory/Performance Issues

- Reduce `VECTOR_DB_TOP_K` in config (fewer memories retrieved)
- Use smaller embedding model: `all-MiniLM-L6-v2` (current, fast)
- Reduce `LLM_MAX_TOKENS` (shorter responses)

### No Messages Parsed

```bash
# Verify file exists
ls data/uploads/chat.txt

# Check encoding
file data/uploads/chat.txt
```

## Debugging Tips

### Use Python's Interactive Mode

```bash
python
>>> from backend.chat_parser import parse_whatsapp_chat
>>> msgs = parse_whatsapp_chat("data/uploads/chat.txt")
>>> print(len(msgs), "messages parsed")
>>> print(msgs[0])  # View first message structure
```

### Check Module Functionality

```bash
# Test chat parser
python -c "from backend.chat_parser import parse_whatsapp_chat; print(len(parse_whatsapp_chat('data/uploads/chat.txt')))"

# Test embeddings
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print(m.encode('hello').shape)"
```

### Verbose Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run code to see debug output
```

## Setting Up IDE

### VS Code

```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.python"
  }
}
```

### PyCharm

- Create new project in the repo folder
- Set interpreter to `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Mac/Linux)
- Mark `backend/` as Sources Root

## Before Committing

```bash
# Run tests
pytest

# Format code
black backend/

# Check style
flake8 backend/

# Commit
git add .
git commit -m "[TYPE] Your commit message"
```

## Need Help?

- Read [README_CONTRIBUTORS.md](README_CONTRIBUTORS.md) for architecture details
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- Review existing code for patterns
- Ask in GitHub Discussions

---

**Ready to contribute?** Check out [CONTRIBUTING.md](CONTRIBUTING.md) for the full guidelines!
