# EchoChat - Contributor's Guide

> **A local, privacy-first AI system that learns to mimic a person's chat style from WhatsApp conversations**

This guide is designed to help contributors understand, develop, and extend the EchoChat project. Whether you're fixing bugs, adding features, or improving documentation, this document has you covered.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Module Breakdown](#architecture--module-breakdown)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [How It Works](#how-it-works)
6. [Module Reference](#module-reference)
7. [API Documentation](#api-documentation)
8. [Development Workflow](#development-workflow)
9. [Contributing Guidelines](#contributing-guidelines)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

**EchoChat** is a local, privacy-preserving GenAI system that:

- **Analyzes** WhatsApp chat exports to understand communication patterns
- **Builds** training datasets and semantic memory stores
- **Profiles** personality traits (language mixing, emoji usage, humor style)
- **Generates** contextually relevant responses matching the target person's style
- **Runs locally** using Ollama (no cloud APIs, no data leakage)

### Key Features

✅ Local LLM inference via Ollama  
✅ Semantic memory with sentence-transformers  
✅ Personality profiling (Hinglish detection, emoji patterns, tone analysis)  
✅ Support for multilingual content (Marathi, Hindi, English)  
✅ Multiple interfaces (CLI, REST API, Web UI)  
✅ Optional QLoRA fine-tuning for custom models  
✅ Message parsing that handles emojis, system messages, and media placeholders

---

## 🏗️ Architecture & Module Breakdown

```
User Input
    ↓
┌─────────────────────────────────────┐
│  RESPONDER (responder.py)           │
│  - Handles LLM inference            │
│  - Manages personality injection    │
│  - Applies safety filters           │
└─────────────────────────────────────┘
    ↑                                  ↑
    │                                  │
┌───────────────────────┐   ┌──────────────────────┐
│ MEMORY STORE          │   │ PERSONALITY PROFILE  │
│ (memory_store.py)     │   │ (personality_...py)  │
│ - Semantic search     │   │ - Language patterns  │
│ - Embedding storage   │   │ - Emoji analysis     │
│ - Context retrieval   │   │ - Humor detection    │
└───────────────────────┘   └──────────────────────┘
     ↑                            ↑
     │                            │
┌─────────────────────────────────────────┐
│ DATASET BUILDER (dataset_builder.py)    │
│ - Training pair creation                │
│ - Memory data formatting                │
│ - Message filtering                     │
└─────────────────────────────────────────┘
     ↑
     │
┌─────────────────────────────────────────┐
│ CHAT PARSER (chat_parser.py)            │
│ - WhatsApp export parsing               │
│ - Emoji & Unicode handling              │
│ - Multi-line message support            │
│ - System message filtering              │
└─────────────────────────────────────────┘
     ↑
     │
WhatsApp Chat Export (chat.txt)
```

---

## 📁 Project Structure

```
echochat/
├── README.md                    # Original user README
├── README_CONTRIBUTORS.md       # This file
├── requirements.txt             # Runtime dependencies
├── requirements-train.txt       # Optional training dependencies
├── pytest.ini                   # Testing configuration
│
├── backend/                     # Core application logic
│   ├── __init__.py
│   ├── main.py                  # Main orchestration script (5-phase pipeline)
│   ├── cli.py                   # Interactive CLI interface
│   ├── api.py                   # FastAPI REST API server
│   │
│   ├── config.py                # Configuration: paths, LLM settings, hyperparameters
│   ├── chat_parser.py           # WhatsApp chat text parsing
│   ├── dataset_builder.py       # Training & memory dataset creation
│   ├── personality_analyzer.py  # Communication pattern analysis
│   ├── memory_store.py          # Vector-based semantic memory
│   ├── responder.py             # LLM inference & response generation
│   │
│   ├── utils.py                 # Utility functions (text cleaning, emoji extraction)
│   ├── train_qlora.py           # Optional QLoRA fine-tuning script
│   ├── test_main.py             # Unit tests
│   └── __pycache__/             # Python bytecode cache
│
├── data/                        # Data directory (git-ignored for privacy)
│   ├── memory_data.json         # Semantic memory entries
│   ├── personality_profile.json # Analyzed personality traits
│   ├── training_data.jsonl      # Training dataset (JSONL format)
│   ├── models/
│   │   └── echobot-lora/        # Fine-tuned model storage
│   └── uploads/
│       └── chat.txt             # WhatsApp chat export (user-provided)
│
├── ui/                          # Static web UI files
│   ├── index.html               # Web interface
│   ├── app.js                   # Frontend JavaScript
│   └── style.css                # UI styling
│
└── frontend/                    # Optional React frontend (under development)
    ├── public/
    └── src/
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running locally
- Git (for version control)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/echochat.git
cd echochat
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Runtime only
pip install -r requirements.txt

# Or with training support (optional, requires GPU)
pip install -r requirements.txt -r requirements-train.txt
```

### 4. Prepare WhatsApp Chat Export

1. Open WhatsApp → Chat settings → Export chat (without media)
2. Save as `.txt` file
3. Place at: `data/uploads/chat.txt`

### 5. Configure Settings (Optional)

Edit `backend/config.py`:

```python
# Change the person to simulate
ECHO_PERSON = "Your Contact Name"

# Change LLM model (mistral, llama2, neural-chat, orca-mini)
LLM_MODEL_NAME = "mistral"

# Adjust temperature (0-1, higher = more creative)
LLM_TEMPERATURE = 0.7
```

### 6. Start Ollama Server

```bash
ollama serve
# In another terminal:
ollama pull mistral
```

### 7. Run the Pipeline

```bash
python -m backend.main
```

---

## 🔄 How It Works

### Phase 1: Chat Parsing (`chat_parser.py`)

**Purpose:** Extract structured data from WhatsApp export

**What it does:**
- Parses timestamps, sender names, messages
- Handles multi-line messages
- Filters system messages (user added, removed, left, etc.)
- Detects and preserves emojis
- Supports Unicode and code-mixed text (Hinglish, Marathi)

**Input:** `data/uploads/chat.txt`  
**Output:** List of `{timestamp, sender, message, length, has_emoji}`

**Example:**
```
Input: "12/01/2024, 3:45 pm - Lady Parus: Hey! 🎉 How are you?"
Output: {
  "timestamp": "2024-01-12 15:45:00",
  "sender": "Lady Parus",
  "message": "Hey! 🎉 How are you?",
  "length": 24,
  "has_emoji": true
}
```

---

### Phase 2: Dataset Building (`dataset_builder.py`)

**Purpose:** Create training and memory datasets

**Training Data:**
- Message pairs: `(previous_message_from_other) → (response_from_target_person)`
- Format: `{instruction, input, output, metadata}`
- Used for optional LLM fine-tuning

**Memory Data:**
- All messages from target person with metadata
- Includes: timestamp, word count, emoji presence, question status
- Used for semantic retrieval

**Output Files:**
- `data/training_data.jsonl`
- `data/memory_data.json`

---

### Phase 3: Personality Analysis (`personality_analyzer.py`)

**Purpose:** Extract communication patterns and style markers

**Analyzed Features:**

| Feature | Description |
|---------|-------------|
| **Language Mixing** | Hinglish/code-mixing detection, percentage English vs Hindi/Marathi |
| **Emoji Patterns** | Most used emojis, frequency, position (start/middle/end) |
| **Sentence Structure** | Average length, complexity, punctuation habits |
| **Humor Style** | Sarcasm, self-deprecation, meme references |
| **Emotional Patterns** | Response type distribution (positive, negative, neutral) |
| **Topic Preferences** | Common discussion topics |
| **Response Characteristics** | Question rate, capitalization style, use of abbreviations |

**Output:** `data/personality_profile.json`

**Example Profile:**
```json
{
  "total_messages": 1250,
  "language_mixing": {
    "hinglish_percentage": 35.2,
    "code_mixing_detected": true
  },
  "emoji_patterns": {
    "total_emojis": 320,
    "most_used": ["😂", "❤️", "😅", "🎯", "💀"],
    "emoji_frequency": 0.256
  },
  "humor_style": {
    "sarcasm_score": 0.7,
    "self_deprecation_score": 0.4
  },
  ...
}
```

---

### Phase 4: Memory Store Initialization (`memory_store.py`)

**Purpose:** Create semantic memory for context-aware responses

**Features:**
- Loads all memory data from `data/memory_data.json`
- Generates embeddings using sentence-transformers (`all-MiniLM-L6-v2`)
- Supports semantic search (find similar past messages)
- Falls back to recent messages if embeddings unavailable

**How Semantic Search Works:**
1. User message → Generate embedding
2. Compare with all stored message embeddings (cosine similarity)
3. Retrieve top-K most similar messages
4. Inject retrieval results into LLM prompt

---

### Phase 5: LLM Response Generation (`responder.py`)

**Purpose:** Generate context-aware responses via Ollama

**Workflow:**
1. Get user message
2. Retrieve personality profile and stored context
3. Fetch relevant memories via semantic search
4. Construct prompt with:
   - System message (instructions)
   - Personality injection (style guidelines)
   - Retrieved memories (context)
   - User message (input)
5. Call Ollama API with configured temperature/tokens
6. Apply safety filters
7. Return response

**Prompt Structure:**
```
[SYSTEM PROMPT]
"You are a conversational AI based on a real person's communication style."

[PERSONALITY INJECTION]
"Personality Profile:
- Uses Hinglish frequently
- Enjoys humor and sarcasm
- Uses emojis in ~25% of messages"

[MEMORY INJECTION]
"Recent similar conversations:
- User: 'How's work?'
  Person: 'Ata chalyach aahe, too much chaos 😅'"

[USER INPUT]
"User: What's up?"

[GENERATE RESPONSE]
```

---

## 📚 Module Reference

### `config.py`

Central configuration file. Modify here to customize behavior.

**Key Settings:**

```python
# Target person to simulate
ECHO_PERSON = "Lady Parus"

# LLM Settings
LLM_MODEL_NAME = "mistral"          # Model choice
LLM_TEMPERATURE = 0.7               # 0=deterministic, 1=creative
LLM_MAX_TOKENS = 256                # Max response length
LLM_API_ENDPOINT = "http://localhost:11434"  # Ollama endpoint

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 384-dim
VECTOR_DB_TOP_K = 5                 # Retrieved memories per query

# Training Configuration
TRAIN_BATCH_SIZE = 8
TRAIN_EPOCHS = 3
TRAIN_LEARNING_RATE = 5e-4
TRAIN_LORA_R = 16                   # QLoRA rank
```

### `chat_parser.py`

**Main Function:** `parse_whatsapp_chat(file_path: str) -> List[Dict]`

Handles:
- WhatsApp timestamp format (DD/MM/YYYY, HH:MM am/pm)
- Multi-line messages
- Emoji preservation
- Unicode (Hinglish, Marathi, etc.)
- System message filtering

**Usage:**
```python
from backend.chat_parser import parse_whatsapp_chat
messages = parse_whatsapp_chat("data/uploads/chat.txt")
```

### `dataset_builder.py`

**Main Functions:**
- `build_datasets(messages, echo_person)` → (training_data, memory_data)
- `load_training_data(path)` → List[Dict]
- `load_memory_data(path)` → List[Dict]

**Usage:**
```python
from backend.dataset_builder import build_datasets
train_data, mem_data = build_datasets(messages, "Lady Parus")
```

### `personality_analyzer.py`

**Main Class:** `PersonalityAnalyzer`

**Key Methods:**
- `analyze()` → Dict (complete personality profile)
- `save_profile(path)` → Saves JSON
- `get_summary()` → Human-readable summary

**Usage:**
```python
from backend.personality_analyzer import PersonalityAnalyzer
analyzer = PersonalityAnalyzer(messages)
profile = analyzer.analyze()
analyzer.save_profile("data/personality_profile.json")
```

### `memory_store.py`

**Main Class:** `MemoryStore`

**Key Methods:**
- `load_memories()` → Loads data from JSON
- `retrieve(query: str, k: int) -> List[str]` → Semantic search
- `get_stats()` → Memory statistics

**Usage:**
```python
from backend.memory_store import MemoryStore
store = MemoryStore(memory_data_path="data/memory_data.json")
similar = store.retrieve("How's college?", k=3)
```

### `responder.py`

**Main Class:** `Responder`

**Key Methods:**
- `generate_response(message: str, include_memories: bool) -> Dict`
- `_test_connection()` → Checks Ollama connection

**Return Format:**
```python
{
  "response": "Ata college ch bomb ahe 😅",
  "memories_used": ["Similar past message 1", "Similar past message 2"],
  "model": "mistral",
  "success": True
}
```

**Usage:**
```python
from backend.responder import Responder
responder = Responder(personality_profile=profile, memory_store=store)
result = responder.generate_response("Hey! What's up?", True)
print(result["response"])
```

### `utils.py`

Utility functions for text processing:

- `clean_text(text)` → Normalized text
- `extract_emojis(text)` → List of emojis
- `count_hinglish_words(text)` → Hinglish count
- `calculate_message_stats(messages)` → Statistics dictionary

---

## 🌐 API Documentation

### Starting the Server

```bash
python -m backend.api
# Server runs at http://127.0.0.1:5000/
```

### Endpoints

#### `GET /health`

Check server status and model info

**Response:**
```json
{
  "status": "ok",
  "model": "mistral"
}
```

---

#### `POST /chat`

Generate a response

**Request Body:**
```json
{
  "message": "How's the day?",
  "include_memories": true
}
```

**Response:**
```json
{
  "response": "Din kaafi acha gela! 🎯",
  "memories_used": [
    "I had a great meeting earlier",
    "Just finished my project"
  ],
  "model": "mistral",
  "success": true
}
```

**Error Response (if Ollama not running):**
```json
{
  "response": "",
  "memories_used": [],
  "model": "mistral",
  "success": false
}
```

---

#### `GET /` & `GET /ui`

Serve web interface at `ui/index.html`

---

## 💻 Development Workflow

### Running Different Interfaces

**CLI (Interactive Chat)**
```bash
python -m backend.cli

# Commands available:
# /help        - Show commands
# /mem on      - Enable memory
# /mem off     - Disable memory
# /exit, /quit - Exit
```

**API Server**
```bash
python -m backend.api
# Access at http://127.0.0.1:5000/
```

**Full Pipeline**
```bash
python -m backend.main
# Runs all 5 phases: Parse → Build → Analyze → Memory → Test
```

---

### Adding New Features

**Example: Adding a new analysis metric to PersonalityAnalyzer**

1. Add method to `PersonalityAnalyzer` class:
```python
def _analyze_response_time(self) -> Dict:
    """Analyze time gaps between messages."""
    # Implementation
    return {"avg_gap_minutes": 5.2, "max_gap_hours": 12}
```

2. Call it in `analyze()` method:
```python
self.profile['response_timing'] = self._analyze_response_time()
```

3. Update config if needed
4. Test with `python -m backend.main`

---

### Testing

Run tests:
```bash
pytest backend/test_main.py -v
```

Write new tests in `backend/test_main.py`:
```python
def test_parse_chat():
    messages = parse_whatsapp_chat("data/uploads/chat.txt")
    assert len(messages) > 0
    assert all("sender" in msg for msg in messages)
```

---

## 🤝 Contributing Guidelines

### Before You Start

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/your-feature-name`
3. **Keep commits focused:** One feature per commit
4. **Write descriptive commit messages**

### Code Standards

- **Python Version:** 3.8+
- **Style:** PEP 8 with max line length of 100
- **Type Hints:** Use where helpful
- **Documentation:** Docstrings for all functions/classes

### Commit Message Format

```
[TYPE] Short description

Longer explanation if needed.

- List key changes
- One per line
```

**Types:** `[FEATURE]`, `[FIX]`, `[DOCS]`, `[REFACTOR]`, `[TEST]`

### Pull Request Process

1. Update `README.md` if needed
2. Add tests for new functionality
3. Run tests: `pytest`
4. Keep PR focused on one thing
5. Write clear PR description

---

## 🛠️ Troubleshooting

### Ollama Connection Error

```
❌ Cannot connect to Ollama at http://localhost:11434
```

**Solution:**
```bash
# Install Ollama
# Start Ollama server
ollama serve

# In another terminal, pull a model
ollama pull mistral
```

---

### No Messages Parsed

```
ERROR: No messages parsed. Check chat file.
```

**Causes:**
- Wrong file path
- File encoding not UTF-8
- Wrong WhatsApp export format

**Solution:**
```bash
# Verify file exists
ls data/uploads/chat.txt

# Check encoding
file data/uploads/chat.txt

# Try re-exporting from WhatsApp
```

---

### ImportError: sentence_transformers

```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution:**
```bash
pip install -r requirements.txt
```

---

### Empty Embeddings

```
sentence-transformers not installed. Embeddings disabled.
```

**Impact:** System falls back to recent messages instead of semantic search

**Solution:**
```bash
pip install sentence-transformers
```

---

### LLM Response Generation Fails

**Cause:** Ollama not running or wrong model

**Solution:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# If curl fails, start Ollama
ollama serve

# Verify model is pulled
ollama list
```

---

## 🚀 Future Enhancements

Potential areas for contribution:

- **[ ] Support for Telegram exports**
- **[ ] Multi-person simulation (multiple personalities)**
- **[ ] Web UI improvements (React frontend)**
- **[ ] Vector database (Pinecone, Weaviate)**
- **[ ] Fine-tuning dashboard**
- **[ ] Conversation topic clustering**
- **[ ] Sentiment analysis improvements**
- **[ ] Voice response generation (TTS)**
- **[ ] Docker containerization**
- **[ ] GitHub Actions CI/CD**

---

## 📞 Support & Questions

- **Issues:** Use GitHub Issues for bugs and feature requests
- **Discussions:** Use GitHub Discussions for general questions
- **Documentation:** Check the [original README.md](README.md)

---

## 📄 License

[Add your license here]

---

## 👥 Contributors

<!-- Add contributor list here once you have some -->
- Aayush Walsangikar (Original Author)

---

**Happy Contributing! 🎉**

If you have questions or run into issues, don't hesitate to open an issue or start a discussion.
