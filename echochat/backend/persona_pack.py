import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .text_filter import is_blocked, is_file_related


def _is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F1E6 <= cp <= 0x1F1FF
        or 0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
    )


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_persona_pack(messages: List[Dict]) -> Dict:
    if not messages:
        return {}

    stop_words = {
        "the", "a", "an", "is", "are", "am", "to", "of", "in", "and", "or", "on",
        "at", "de", "ka", "na", "u", "i", "me", "my", "you", "your", "we",
        "us", "he", "she", "it", "they", "them", "this", "that", "hi", "hey",
        "ok", "okay", "yes", "no",
    }
    word_counts = []
    char_counts = []
    emoji_counter = Counter()
    starters = Counter()
    endings = Counter()
    slang_counter = Counter()
    short_replies = []

    clean_messages = []
    for msg in messages:
        text = msg.get("message", "").strip()
        if not text:
            continue
        if is_blocked(text) or is_file_related(text):
            continue
        clean_messages.append(msg)

    if not clean_messages:
        return {}

    for msg in clean_messages:
        text = msg.get("message", "").strip()
        if "http://" in text or "https://" in text:
            continue

        words = _tokenize(text)
        if words:
            word_counts.append(len(words))
        char_counts.append(len(text))

        if len(words) >= 2:
            starter = " ".join(words[:2])
            ending = " ".join(words[-2:])
            if not any(is_file_related(tok) for tok in starter.split()):
                starters[starter] += 1
            if not any(is_file_related(tok) for tok in ending.split()):
                endings[ending] += 1
        elif len(words) == 1:
            if not is_file_related(words[0]):
                starters[words[0]] += 1
                endings[words[0]] += 1

        for word in words:
            if word in stop_words:
                continue
            if is_file_related(word):
                continue
            if word.startswith("http"):
                continue
            if len(word) < 2:
                continue
            slang_counter[word] += 1

        emojis = [ch for ch in text if _is_emoji_char(ch)]
        for emoji in emojis:
            emoji_counter[emoji] += 1

        if len(words) <= 6 and len(short_replies) < 50:
            if not any(is_file_related(word) for word in words) and "http" not in text:
                short_replies.append(text)

    total = len(clean_messages)
    avg_words = sum(word_counts) / max(len(word_counts), 1)
    avg_chars = sum(char_counts) / max(len(char_counts), 1)
    emoji_rate = len([1 for msg in clean_messages if msg.get("has_emoji")]) / max(total, 1)

    return {
        "total_messages": total,
        "avg_words_per_message": avg_words,
        "avg_chars_per_message": avg_chars,
        "emoji_rate": emoji_rate,
        "top_emojis": [e for e, _ in emoji_counter.most_common(8)],
        "common_starters": [s for s, _ in starters.most_common(8)],
        "common_endings": [s for s, _ in endings.most_common(8)],
        "slang_words": [s for s, _ in slang_counter.most_common(20)],
        "short_reply_samples": short_replies[:12],
    }


def save_persona_pack(pack: Dict, path: str | Path) -> None:
    if not pack:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)


def load_persona_pack(path: str | Path) -> Dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}
