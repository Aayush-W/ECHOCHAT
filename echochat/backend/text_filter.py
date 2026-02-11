import re
from typing import Iterable


SYSTEM_KEYWORDS = {
    "messages you send to this group are now encrypted",
    "security code changed",
    "this message was deleted",
    "message was deleted",
    "you deleted this message",
    "media omitted",
    "changed this group's icon",
    "changed the subject",
}

MEDIA_PATTERNS = [
    r"^<media omitted>$",
    r"^image omitted$",
    r"^video omitted$",
    r"^audio omitted$",
    r"^document omitted$",
    r"^sticker omitted$",
    r"^gif omitted$",
    r"^file attached$",
]

FILE_EXTENSIONS = {
    "jpg", "jpeg", "png", "gif", "webp", "bmp", "heic",
    "mp4", "mov", "avi", "mkv", "mp3", "m4a", "opus", "wav",
    "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "csv",
    "zip", "rar", "7z", "tar", "gz",
    "txt", "md", "json", "xml", "yaml", "yml",
    "py", "js", "ts", "html", "css", "java", "c", "cpp", "h",
    "vcf", "dxf",
}

ATTACHMENT_MARKERS = [
    "(file attached)",
    "<media omitted>",
]

BANNED_OUTPUT_TOKENS = {
    "file attached",
    "media omitted",
    "was deleted",
    "message was deleted",
}

FILE_KEYWORDS = {
    "file", "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "image",
    "photo", "pic", "picture", "video", "audio", "attachment", "attached",
    "link", "url", "drive", "pdfs", "docs",
}

CHAT_INTENT_PHRASES = [
    "how are you",
    "how r u",
    "how're you",
    "what's up",
    "whats up",
    "wassup",
    "kya haal",
    "kya hal",
    "kya scene",
    "kya chal",
    "aur kya",
    "kaisa hai",
    "kaisi hai",
    "kaise ho",
    "kaisa ho",
    "kaise hai",
    "kaisa",
    "kaise",
    "kaisi",
]

CHAT_INTENT_TOKENS = {
    "hi", "hello", "hey", "yo", "sup", "hru", "bro", "bhai", "yaar",
    "kya", "kaisa", "kaisi", "kaise", "haal", "scene", "chal", "kese",
}

INFO_INTENT_TOKENS = {
    "why", "what", "when", "where", "which", "who", "how", "explain",
    "tell", "detail", "details", "meaning", "matlab", "kyun", "kyu",
}


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def is_system_message(text: str) -> bool:
    lowered = _normalize(text)
    if not lowered:
        return True
    for keyword in SYSTEM_KEYWORDS:
        if keyword in lowered:
            return True
    return False


def is_media_placeholder(text: str) -> bool:
    lowered = _normalize(text)
    if not lowered:
        return True
    for pattern in MEDIA_PATTERNS:
        if re.match(pattern, lowered):
            return True
    return False


def _is_file_name(token: str) -> bool:
    token = token.strip().strip('"').strip("'")
    if "." not in token:
        return False
    base, ext = token.rsplit(".", 1)
    if not base:
        return False
    return ext.lower() in FILE_EXTENSIONS


def is_file_name_only(text: str) -> bool:
    lowered = _normalize(text)
    if not lowered:
        return True
    if any(marker in lowered for marker in ATTACHMENT_MARKERS):
        return True
    lines = [line.strip() for line in lowered.splitlines() if line.strip()]
    if not lines:
        return True
    # If every line is a file-like token, treat as file-only
    for line in lines:
        tokens = re.split(r"\s+", line)
        if not any(_is_file_name(token) for token in tokens):
            return False
    return True


def contains_banned_output(text: str) -> bool:
    lowered = _normalize(text)
    if not lowered:
        return False
    if any(token in lowered for token in BANNED_OUTPUT_TOKENS):
        return True
    tokens = re.findall(r"\b\w+\b", lowered)
    if "file" in tokens or "attached" in tokens:
        return True
    return False


def is_file_related(text: str) -> bool:
    lowered = _normalize(text)
    if not lowered:
        return False
    if any(marker in lowered for marker in ATTACHMENT_MARKERS):
        return True
    tokens = re.findall(r"\b[\w\.-]+\b", lowered)
    for token in tokens:
        if _is_file_name(token):
            return True
    word_tokens = re.findall(r"\b\w+\b", lowered)
    return any(token in FILE_KEYWORDS for token in word_tokens)


def is_chat_safe(text: str) -> bool:
    return not is_blocked(text) and not is_file_related(text)


def is_blocked(text: str) -> bool:
    if not text:
        return True
    if is_system_message(text):
        return True
    if is_media_placeholder(text):
        return True
    if is_file_name_only(text):
        return True
    return False


def classify_intent(text: str) -> str:
    lowered = _normalize(text)
    if not lowered:
        return "chat"
    if is_file_related(lowered):
        return "file"

    tokens = re.findall(r"\b\w+\b", lowered)
    if any(phrase in lowered for phrase in CHAT_INTENT_PHRASES):
        if len(tokens) <= 8 or "?" not in lowered:
            return "chat"

    if any(token in CHAT_INTENT_TOKENS for token in tokens) and len(tokens) <= 6:
        return "chat"

    if "?" in lowered:
        return "info"

    if any(token in INFO_INTENT_TOKENS for token in tokens):
        return "info"

    if len(tokens) <= 6:
        return "chat"
    return "info"


def filter_texts(texts: Iterable[str]) -> list[str]:
    return [text for text in texts if not is_blocked(text)]
