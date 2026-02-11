"""
Enhanced Dataset Builder: Builds better training and memory datasets with quality control.

Improvements:
- Filters contaminated training data
- Improved memory data structuring
- Quality metrics tracking
- Better context extraction
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from .data_filter import TrainingDataFilter
from .text_filter import is_blocked, is_file_related


def build_datasets_enhanced(
    messages: List[Dict],
    echo_person: str,
    skip_quality_check: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Build training and memory datasets with quality enhancement.

    Returns:
        (training_data, memory_data, quality_stats)
    """
    # Extract messages from target person
    echo_messages = [msg for msg in messages if msg["sender"] == echo_person]

    if not echo_messages:
        return [], [], {"error": f"No messages from {echo_person}"}

    # Build training pairs (input-output)
    training_pairs = _build_training_pairs(messages, echo_person)

    # Filter contaminated training data
    if not skip_quality_check:
        training_pairs, filtered_pairs, filter_stats = TrainingDataFilter.filter_training_data(
            training_pairs,
            min_quality=0.4,
            max_contamination=0.6,
            verbose=True,
        )
    else:
        filter_stats = {"total_input": len(training_pairs), "valid_output": len(training_pairs)}

    # Build memory data with rich context
    memory_data = _build_memory_data(echo_messages)

    quality_stats = {
        "total_echo_messages": len(echo_messages),
        "training_pairs_generated": filter_stats.get("total_input", 0),
        "training_pairs_valid": filter_stats.get("valid_output", 0),
        "training_pairs_filtered": filter_stats.get("filtered_count", 0),
        "memory_entries": len(memory_data),
        "filter_results": filter_stats,
    }

    return training_pairs, memory_data, quality_stats


def _build_training_pairs(
    messages: List[Dict],
    echo_person: str,
    context_window: int = 2,
) -> List[Dict]:
    """
    Build training pairs from conversation history.

    Context window: how many previous messages to include as context.
    """
    pairs = []

    for i, msg in enumerate(messages):
        if msg["sender"] != echo_person:
            continue

        # Get the input (previous message from other person)
        input_text = None
        for j in range(i - 1, max(i - context_window, -1), -1):
            if messages[j]["sender"] != echo_person:
                input_text = messages[j].get("message", "").strip()
                if input_text:
                    break

        # Get the output (this message)
        output_text = msg.get("message", "").strip()

        if input_text and output_text:
            # Skip if blocked or file-related
            if is_blocked(input_text) or is_blocked(output_text):
                continue
            if is_file_related(input_text) or is_file_related(output_text):
                continue

            # Create training pair
            pair = {
                "instruction": f"Reply as {echo_person} in usual conversational style",
                "input": input_text,
                "output": output_text,
                "metadata": {
                    "input_sender": [m["sender"] for m in messages[max(0, i-context_window):i] if m["sender"] != echo_person][-1:] or ["Unknown"],
                    "timestamp_input": messages[i-1].get("timestamp", "") if i > 0 else "",
                    "timestamp_output": msg.get("timestamp", ""),
                },
            }
            pairs.append(pair)

    return pairs


def _build_memory_data(
    echo_messages: List[Dict],
    min_length: int = 3,
) -> List[Dict]:
    """
    Build memory database from user's messages.

    Includes rich context for semantic search.
    """
    memory_entries = []

    for msg in echo_messages:
        text = msg.get("message", "").strip()

        # Skip very short or blocked messages
        if len(text) < min_length:
            continue
        if is_blocked(text):
            continue

        # Classify characteristics
        is_question = _is_question(text)
        contains_code = _contains_code(text)
        word_count = len(text.split())
        has_emoji = any(ord(c) > 0x1F000 for c in text)

        entry = {
            "text": text,
            "timestamp": msg.get("timestamp", ""),
            "length": len(text),
            "has_emoji": has_emoji,
            "context": {
                "word_count": word_count,
                "contains_code": contains_code,
                "is_question": is_question,
            },
        }

        memory_entries.append(entry)

    return memory_entries


def _is_question(text: str) -> bool:
    """Check if text is a question."""
    text = text.strip()
    if text.endswith("?"):
        return True
    question_words = ["what", "when", "where", "who", "why", "how", "ka", "ki", "kya", "kaisa"]
    if any(text.lower().startswith(w) for w in question_words):
        return True
    return False


def _contains_code(text: str) -> bool:
    """Check if text contains code or technical content."""
    code_patterns = [
        r"```|<code>|def |import |function|print\(",
        r"[{}\[\]()]",  # Multiple brackets
        r"[a-zA-Z_$][a-zA-Z0-9_$]*\(",  # Function calls
    ]
    for pattern in code_patterns:
        if re.search(pattern, text):
            return True
    return False


def validate_and_fix_datasets(
    training_data: List[Dict],
    memory_data: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate and fix any issues in datasets.
    """
    # Remove duplicate training pairs
    seen_pairs = set()
    cleaned_training = []

    for pair in training_data:
        key = (pair.get("input", "").lower(), pair.get("output", "").lower())
        if key not in seen_pairs:
            seen_pairs.add(key)
            cleaned_training.append(pair)

    # Sort memory data by timestamp for consistency
    try:
        sorted_memory = sorted(
            memory_data,
            key=lambda x: datetime.fromisoformat(x.get("timestamp", "")) if x.get("timestamp") else datetime.min
        )
    except:
        sorted_memory = memory_data

    return cleaned_training, sorted_memory


# Keep original functions for backward compatibility
def build_datasets(
    messages: List[Dict],
    echo_person: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Original function for backward compatibility."""
    training_pairs, memory_data, _ = build_datasets_enhanced(messages, echo_person)
    return training_pairs, memory_data


def save_datasets(
    training_data: List[Dict],
    memory_data: List[Dict],
    training_path: str = "data/training_data.jsonl",
    memory_path: str = "data/memory_data.json",
) -> Tuple[bool, bool]:
    """Save datasets to disk."""
    try:
        # Clean data
        training_data, memory_data = validate_and_fix_datasets(training_data, memory_data)

        # Save training data (JSONL format)
        with open(training_path, "w", encoding="utf-8") as f:
            for pair in training_data:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        # Save memory data (JSON format)
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)

        return True, True
    except Exception as e:
        print(f"Error saving datasets: {e}")
        return False, False


def load_training_data(path: str = "data/training_data.jsonl") -> List[Dict]:
    """Load training data from JSONL file."""
    pairs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
    except FileNotFoundError:
        print(f"Training data file not found: {path}")
    except json.JSONDecodeError as e:
        print(f"Error parsing training data: {e}")
    return pairs


def load_memory_data(path: str = "data/memory_data.json") -> List[Dict]:
    """Load memory data from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Memory data file not found: {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing memory data: {e}")
        return []
