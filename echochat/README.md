# EchoChat

EchoChat is a local pipeline that simulates a specific person's chat style based on a WhatsApp export.
It builds training data and a semantic memory store, analyzes personality signals, and generates responses via a local Ollama model.

## Quick Start
1. Export a WhatsApp chat to text and place it here:
   echochat/data/uploads/chat.txt
2. Install dependencies:
   pip install -r echochat/requirements.txt
3. Run the pipeline:
   python -m echochat.backend.main

## Local API
Start the API server:
python -m echochat.backend.api

Default endpoints:
- GET /health
- POST /chat with JSON:
  {"message": "hello", "include_memories": true}

## CLI Chat
Start the interactive CLI:
python -m echochat.backend.cli

## Web UI
Start the API server, then open:
http://127.0.0.1:5000/

## What It Does
- Parses WhatsApp chat messages.
- Builds training pairs and memory data.
- Analyzes language, emoji patterns, and topics.
- Loads a sentence-transformer model for semantic recall.
- Uses Ollama to generate a response in the target style.

## Ollama Setup
1. Install Ollama.
2. Start the server:
   ollama serve
3. Pull a model (default is mistral):
   ollama pull mistral

Configure the model and endpoint in echochat/backend/config.py.

## Training (Optional)
If you want to finetune a model with QLoRA:
1. Install training deps:
   pip install -r echochat/requirements-train.txt
2. Run:
   python echochat/backend/train_qlora.py

This requires a GPU and is not needed for basic usage.

## Privacy
The echochat/data/ folder typically contains personal chat content and attachments.
A .gitignore is included to prevent accidental commits of sensitive data.

## Troubleshooting
- If embeddings are unavailable, the system falls back to recent messages.
- If Ollama is not running, response generation will fail.

