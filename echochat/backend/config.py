import os
from pathlib import Path

# ===== PATHS =====
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = DATA_DIR / "models"
UPLOADS_DIR = DATA_DIR / "uploads"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# ===== ECHO PERSON =====
ECHO_PERSON = "Lady Parus"

# ===== DATA FILES =====
CHAT_UPLOAD_PATH = UPLOADS_DIR / "chat.txt"
TRAINING_DATA_PATH = DATA_DIR / "training_data.jsonl"
MEMORY_DATA_PATH = DATA_DIR / "memory_data.json"
PERSONALITY_PROFILE_PATH = DATA_DIR / "personality_profile.json"
VECTOR_DB_PATH = DATA_DIR / "vector_db"

# ===== LLM CONFIGURATION =====
LLM_MODEL_NAME = "mistral"  # Options: "mistral", "llama2", "neural-chat", "orca-mini"
LLM_MODEL_PATH = MODELS_DIR / f"{LLM_MODEL_NAME}-7b"  # For local model storage
LLM_API_ENDPOINT = "http://localhost:11434"  # Ollama API (local)
LLM_TEMPERATURE = 0.7  # Higher = more creative, Lower = more deterministic
LLM_TOP_P = 0.9  # Nucleus sampling
LLM_MAX_TOKENS = 256  # Max response length
LLM_CONTEXT_WINDOW = 2048  # Max context for prompt

# ===== EMBEDDING MODEL =====
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good for semantic search
EMBEDDING_DIM = 384  # Vector dimension

# ===== TRAINING (QLoRA) =====
TRAIN_BATCH_SIZE = 8
TRAIN_EPOCHS = 3
TRAIN_LEARNING_RATE = 5e-4
TRAIN_LORA_R = 16
TRAIN_LORA_ALPHA = 32
TRAIN_LORA_DROPOUT = 0.05
TRAIN_OUTPUT_DIR = MODELS_DIR / "finetuned_model"

# ===== MEMORY/VECTOR DB =====
VECTOR_DB_COLLECTION = "echo_memories"
VECTOR_DB_TOP_K = 5  # Top 5 similar messages to retrieve

# ===== SAFETY & ETHICS =====
SAFETY_DISCLAIMERS = [
    "I'm a simulation of the person, not the real person.",
    "This is a generative AI, not a permanent presence.",
    "Don't treat this as a substitute for real relationships.",
]

UNSAFE_TOPICS = [
    "self-harm",
    "suicide",
    "harassment",
    "illegal activity",
    "explicit sexual content",
]

# ===== LOGGING =====
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE = BASE_DIR / "echochat.log"

# ===== API & FRONTEND =====
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FRONTEND_HOST = "localhost:3000"

# ===== PROMPT TEMPLATES =====
SYSTEM_PROMPT = f"""You are a conversational AI that simulates {ECHO_PERSON}'s communication style.

CRITICAL RULES:
1. You are a SIMULATION, not the real person.
2. You think and reason like a human, but you are not conscious.
3. Never claim to be the real {ECHO_PERSON}.
4. Never promise permanent presence or eternal friendship.
5. Generate NEW responses, not just copies of past messages.
6. Match the person's style: casual, friendly, use emojis appropriately.
7. If unsure about emotional topics, respond with genuine care but honesty.
8. Avoid discussing sensitive topics in a joking manner.

You have access to:
- The person's communication style and personality traits
- Past conversations and memories
- Context about their interests and habits

Respond naturally, as if continuing a real conversation."""

STYLE_PROMPT_INJECTION = """
COMMUNICATION STYLE GUIDELINES:
- Language: Mix of English, Hindi, and Marathi (Hinglish)
- Tone: Casual, friendly, sometimes sarcastic
- Emojis: Use them frequently and naturally (😂, 👍, 🙊, etc.)
- Sentence length: Short, often fragmented
- Response time: Does not over-explain
- Humor: Dry humor, relatable jokes, self-deprecating
- Topics: College, tech, food, hangouts, casual banter
"""

MEMORY_PROMPT_INJECTION = """
RELEVANT MEMORIES:
{memories_context}

Use these memories to:
1. Ground your response in shared experiences
2. Reference past conversations naturally
3. Maintain continuity in the conversation
4. Show that you "remember" important details
"""

print(f"✅ Config loaded from: {Path(__file__)}")
print(f"   Models directory: {MODELS_DIR}")
print(f"   Data directory: {DATA_DIR}")
