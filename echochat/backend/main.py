"""
EchoChat: Main orchestration script.

WORKFLOW:
1. Parse WhatsApp chat
2. Build datasets (training + memory)
3. Analyze personality
4. Initialize memory store
5. Load/test LLM
6. Generate sample responses
"""

from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.chat_parser import parse_whatsapp_chat
from backend.dataset_builder import build_datasets, load_training_data, load_memory_data
from backend.personality_analyzer import PersonalityAnalyzer
from backend.memory_store import MemoryStore
from backend.responder import Responder
from backend.config import (
    CHAT_UPLOAD_PATH,
    TRAINING_DATA_PATH,
    MEMORY_DATA_PATH,
    PERSONALITY_PROFILE_PATH,
    ECHO_PERSON,
)


def main():
    print("\n" + "="*60)
    print("ECHOCHAT - LOCAL GENAI CONVERSATIONAL SYSTEM")
    print("="*60 + "\n")
    
    # ===== PHASE 1: PARSE CHAT =====
    print("[1/5] PARSING WHATSAPP CHAT...")
    print("-" * 60)
    
    chat_messages = parse_whatsapp_chat(str(CHAT_UPLOAD_PATH))
    if not chat_messages:
        print("ERROR: No messages parsed. Check chat file.")
        return
    
    print(f"Parsed {len(chat_messages)} messages")
    
    # Filter by echo person
    echo_messages = [msg for msg in chat_messages if msg['sender'] == ECHO_PERSON]
    print(f"Found {len(echo_messages)} messages from {ECHO_PERSON}")
    
    # ===== PHASE 2: BUILD DATASETS =====
    print("\n[2/5] BUILDING DATASETS...")
    print("-" * 60)
    
    training_data, memory_data = build_datasets(chat_messages, echo_person=ECHO_PERSON)
    print(f"Training: {len(training_data)} pairs")
    print(f"Memory: {len(memory_data)} entries")
    
    # ===== PHASE 3: ANALYZE PERSONALITY =====
    print("\n[3/5] ANALYZING PERSONALITY...")
    print("-" * 60)
    
    analyzer = PersonalityAnalyzer(echo_messages)
    personality_profile = analyzer.analyze()
    analyzer.save_profile(str(PERSONALITY_PROFILE_PATH))
    
    print(analyzer.get_summary())
    
    # ===== PHASE 4: INITIALIZE MEMORY STORE =====
    print("\n[4/5] INITIALIZING MEMORY STORE...")
    print("-" * 60)
    
    memory_store = MemoryStore(memory_data_path=str(MEMORY_DATA_PATH))
    print(memory_store.get_stats())
    
    # ===== PHASE 5: LOAD LLM & TEST =====
    print("\n[5/5] LOADING LLM & TESTING...")
    print("-" * 60)
    
    responder = Responder(
        personality_profile=personality_profile,
        memory_store=memory_store,
    )
    
    print(responder.get_debug_info())
    
    # ===== TEST GENERATION =====
    print("\n" + "="*60)
    print("TESTING RESPONSE GENERATION")
    print("="*60 + "\n")
    
    test_inputs = [
        "Show tr 1:30 ch ahe na?",
        "What's up?",
        "How's college going?",
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        print("-" * 40)
        
        result = responder.generate_response(user_input, verbose=False)
        
        print(f"Response: {result['response']}")
        if result['memories_used']:
            print(f"Memories: {result['memories_used'][:2]}")
        print(f"Success: {result['success']}")
    
    print("\n" + "="*60)
    print("ECHOCHAT INITIALIZATION COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
