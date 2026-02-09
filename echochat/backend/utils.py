import re
from typing import List, Dict
from datetime import datetime


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    return text


def extract_emojis(text: str) -> List[str]:
    """Extract all emojis from text."""
    emoji_regex = r'[\U0001F300-\U0001F9FF]|[^\w\s\-.,!?]'
    return re.findall(emoji_regex, text)


def count_hinglish_words(text: str) -> int:
    """Count Hinglish indicators in text."""
    hinglish_words = {
        'ahe', 'ch', 'ka', 'na', 'ky', 'to', 'te', 'be', 'ata', 'pan',
        'var', 'saang', 'show', 'kya', 'ho', 'jaun', 'gela', 'aaley', 'mazya'
    }
    text_lower = text.lower()
    return sum(1 for word in hinglish_words if f" {word} " in f" {text_lower} ")


def format_timestamp(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%d/%m/%Y, %I:%M %p")


def calculate_message_stats(messages: List[Dict]) -> Dict:
    """Calculate statistics about messages."""
    if not messages:
        return {}
    
    lengths = [msg.get('length', 0) for msg in messages]
    
    return {
        'total_messages': len(messages),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'total_characters': sum(lengths),
        'messages_with_emoji': sum(1 for msg in messages if msg.get('has_emoji', False)),
        'emoji_percentage': (sum(1 for msg in messages if msg.get('has_emoji', False)) / len(messages)) * 100,
    }


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def is_valid_message(message: str) -> bool:
    """Check if message is valid (not empty, not spam)."""
    if not message or len(message.strip()) < 1:
        return False
    if len(message) > 10000:  # Unreasonably long
        return False
    return True