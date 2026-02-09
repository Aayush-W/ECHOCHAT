import json
import sys

from .config import MEMORY_DATA_PATH, PERSONALITY_PROFILE_PATH
from .memory_store import MemoryStore
from .responder import Responder


def _load_personality_profile() -> dict:
    if PERSONALITY_PROFILE_PATH.exists():
        try:
            with open(PERSONALITY_PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def run_cli() -> None:
    print("EchoChat CLI")
    print("Type your message and press Enter.")
    print("Commands: /exit, /quit, /mem on, /mem off, /help")

    profile = _load_personality_profile()
    memory_store = MemoryStore(memory_data_path=str(MEMORY_DATA_PATH))
    responder = Responder(personality_profile=profile, memory_store=memory_store)

    include_memories = True

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in {"/exit", "/quit", "exit", "quit"}:
            print("Exiting.")
            return
        if cmd == "/help":
            print("Commands: /exit, /quit, /mem on, /mem off, /help")
            continue
        if cmd == "/mem on":
            include_memories = True
            print("Memory use: ON")
            continue
        if cmd == "/mem off":
            include_memories = False
            print("Memory use: OFF")
            continue

        result = responder.generate_response(
            user_input,
            include_memories=include_memories,
            verbose=False,
        )
        print(f"Echo: {result.get('response', '')}")


if __name__ == "__main__":
    run_cli()
