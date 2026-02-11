import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F1E6 <= cp <= 0x1F1FF
        or 0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
    )


QUESTIONNAIRE_VERSION = "2026-02-11-q7-v1"

DEFAULT_QUESTIONS: List[Dict] = [
    {
        "id": "greeting",
        "text": "How do they usually greet close friends?",
        "hint": "Example: oye kya scene, yo bro, kya haal",
    },
    {
        "id": "language",
        "text": "Preferred language mix?",
        "options": [
            "Mostly English",
            "Hinglish mix",
            "Mostly Hindi/Marathi",
        ],
    },
    {
        "id": "length",
        "text": "Typical reply length?",
        "options": [
            "Short (1-6 words)",
            "Medium (7-15 words)",
            "Long (16+ words)",
        ],
    },
    {
        "id": "emoji",
        "text": "Emoji usage? (none/rare/occasional/frequent) + favorites",
        "hint": "Example: occasional, uses 😂🔥",
    },
    {
        "id": "slang",
        "text": "Common slang or phrases they use (comma separated)",
        "hint": "Example: bhai, scene kya, chill",
    },
    {
        "id": "humor",
        "text": "Humor vibe?",
        "options": [
            "Playful",
            "Dry",
            "Sarcastic",
            "Serious",
        ],
    },
    {
        "id": "boundaries",
        "text": "Any topics to avoid or style boundaries?",
        "hint": "Example: avoid politics, keep it short",
    },
]


def default_questionnaire() -> Dict:
    return {
        "version": QUESTIONNAIRE_VERSION,
        "questions": DEFAULT_QUESTIONS,
        "answers": {},
        "started_at": None,
        "completed_at": None,
    }


def get_next_question(payload: Dict) -> Optional[Dict]:
    answers = payload.get("answers", {}) if isinstance(payload, dict) else {}
    questions = payload.get("questions", [])
    for question in questions:
        qid = question.get("id")
        if qid and qid not in answers:
            return question
    return None


def record_answer(payload: Dict, answer: str, question_id: Optional[str] = None) -> Tuple[Dict, Optional[Dict], bool]:
    if not isinstance(payload, dict):
        payload = default_questionnaire()
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        answers = {}
    payload["answers"] = answers

    question = None
    questions = payload.get("questions", [])
    if question_id:
        for q in questions:
            if q.get("id") == question_id:
                question = q
                break
    if question is None:
        question = get_next_question(payload)
    if question is None:
        payload["completed_at"] = payload.get("completed_at") or _utc_now()
        return payload, None, True

    qid = question.get("id")
    if qid:
        answers[qid] = answer

    next_q = get_next_question(payload)
    completed = next_q is None
    if completed:
        payload["completed_at"] = payload.get("completed_at") or _utc_now()
    return payload, next_q, completed


def parse_language_preference(answer: str) -> Optional[str]:
    if not answer:
        return None
    text = answer.lower()
    if "hinglish" in text or "mix" in text:
        return "hinglish"
    if "hindi" in text or "marathi" in text:
        return "hindi"
    if "english" in text:
        return "english"
    return None


def parse_length_preference(answer: str) -> Optional[str]:
    if not answer:
        return None
    text = answer.lower()
    if "short" in text or "1-6" in text:
        return "short"
    if "medium" in text or "7-15" in text:
        return "medium"
    if "long" in text or "16" in text or "lengthy" in text:
        return "long"
    return None


def extract_emojis(answer: str) -> List[str]:
    if not answer:
        return []
    emojis = []
    for ch in answer:
        if _is_emoji_char(ch) and ch not in emojis:
            emojis.append(ch)
    return emojis


def extract_slang(answer: str) -> List[str]:
    if not answer:
        return []
    if "," in answer:
        parts = [p.strip() for p in answer.split(",") if p.strip()]
    else:
        parts = re.split(r"\s+", answer.strip())
    slang = []
    for part in parts:
        cleaned = re.sub(r"[^\w'-]+", "", part.lower())
        if not cleaned or len(cleaned) < 2:
            continue
        if cleaned not in slang:
            slang.append(cleaned)
    return slang


def extract_avoid_topics(answer: str) -> List[str]:
    if not answer:
        return []
    parts = re.split(r"[;,/]| and ", answer.lower())
    topics = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned or len(cleaned) < 3:
            continue
        if cleaned not in topics:
            topics.append(cleaned)
    return topics


def apply_questionnaire_overrides(
    questionnaire: Dict,
    persona_pack: Optional[Dict] = None,
    personality_profile: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    persona_pack = persona_pack or {}
    personality_profile = personality_profile or {}
    answers = questionnaire.get("answers", {}) if isinstance(questionnaire, dict) else {}

    greeting = answers.get("greeting") or ""
    language_pref = parse_language_preference(answers.get("language", ""))
    length_pref = parse_length_preference(answers.get("length", ""))
    humor_pref = answers.get("humor") or ""
    emoji_answer = answers.get("emoji") or ""
    slang_answer = answers.get("slang") or ""
    boundaries = answers.get("boundaries") or ""

    overrides = persona_pack.get("questionnaire_overrides", {})

    if greeting:
        persona_pack["preferred_greeting"] = greeting.strip()
        overrides["preferred_greeting"] = greeting.strip()

    if language_pref:
        persona_pack["preferred_language"] = language_pref
        overrides["preferred_language"] = language_pref

    if length_pref:
        persona_pack["preferred_reply_length"] = length_pref
        overrides["preferred_reply_length"] = length_pref

    emojis = extract_emojis(emoji_answer)
    if emojis:
        existing = persona_pack.get("top_emojis", [])
        combined = list(dict.fromkeys(emojis + existing))
        persona_pack["top_emojis"] = combined[:8]
        overrides["top_emojis"] = emojis[:5]
        persona_pack["emoji_rate_hint"] = emoji_answer
    elif emoji_answer:
        persona_pack["emoji_rate_hint"] = emoji_answer

    slang_terms = extract_slang(slang_answer)
    if slang_terms:
        existing = persona_pack.get("slang_words", [])
        combined = list(dict.fromkeys(slang_terms + existing))
        persona_pack["slang_words"] = combined[:24]
        overrides["slang_words"] = slang_terms[:10]

    if humor_pref:
        personality_profile.setdefault("humor_style", {})
        personality_profile["humor_style"]["humor_style"] = humor_pref.lower().strip()
        overrides["humor_style"] = humor_pref

    avoid_topics = extract_avoid_topics(boundaries)
    if avoid_topics:
        persona_pack["avoid_topics"] = avoid_topics
        overrides["avoid_topics"] = avoid_topics

    if boundaries:
        persona_pack["tone_notes"] = boundaries.strip()
        overrides["tone_notes"] = boundaries.strip()

    persona_pack["questionnaire_overrides"] = overrides
    return persona_pack, personality_profile


def load_questionnaire(path: Path) -> Dict:
    if not path.exists():
        return default_questionnaire()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default_questionnaire()


def save_questionnaire(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
