import json

from backend.chat_parser import parse_whatsapp_chat
from backend import memory_store as memory_store_module


def test_parse_whatsapp_chat_multiline(tmp_path):
    content = (
        "01/01/2024, 1:00 pm - Alice: Hello\n"
        "This is a continuation line\n"
        "01/01/2024, 1:01 pm - Bob: Hi\n"
    )
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text(content, encoding="utf-8")

    messages = parse_whatsapp_chat(str(chat_file))
    assert len(messages) == 2
    assert "continuation line" in messages[0]["message"]


def test_memory_store_fallback_without_embeddings(tmp_path):
    data = [
        {"text": "hello", "timestamp": "2024-01-01T00:00:00", "length": 5, "has_emoji": False},
        {"text": "world", "timestamp": "2024-01-02T00:00:00", "length": 5, "has_emoji": False},
    ]
    memory_file = tmp_path / "memory.json"
    memory_file.write_text(json.dumps(data), encoding="utf-8")

    original = memory_store_module.SentenceTransformer
    memory_store_module.SentenceTransformer = None
    try:
        store = memory_store_module.MemoryStore(memory_data_path=str(memory_file))
        results = store.search("test", top_k=1)
        assert results
        assert results[0]["message"]["text"] == "world"
    finally:
        memory_store_module.SentenceTransformer = original
