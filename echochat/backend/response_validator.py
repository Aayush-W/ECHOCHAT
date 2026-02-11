"""
Enhanced Response Validator: Validates and improves response quality.

Features:
- LLM detection with better accuracy
- Response coherence checking
- Personality consistency validation
- Appropriate length enforcement
- Language consistency checks
"""

import re
import logging
from typing import Dict, List, Tuple
from collections import Counter

logger = logging.getLogger("echochat.response_validator")


class ResponseValidator:
    """Validates and measures response quality."""

    # LLM patterns with weights
    LLM_PATTERNS = {
        # Formal greetings
        r"\b(hello there|good morning|good afternoon|good evening|hope you're doing well)\b": 2.5,
        r"\b(i appreciate|i hope|thank you for|please let me know)\b": 2.0,
        # AI disclaims
        r"\b(as an ai|i'm an ai|i cannot|i'm sorry to hear|unfortunately)\b": 3.0,
        # Formal closures
        r"\b(best regards|sincerely yours|warmest regards|respectfully)\b": 2.5,
        # Corporate language
        r"\b(furthermore|moreover|in conclusion|in summary|to summarize)\b": 2.0,
        # Professional structure
        r"\b(i understand|allow me to|permit me to|kindly|regarding)\b": 1.5,
        # Formal action words
        r"\b(elucidate|facilitate|endeavor|notwithstanding)\b": 3.0,
    }

    # Casual patterns with weights
    CASUAL_PATTERNS = {
        # Casual expressions
        r"\b(yeah|yep|nah|just|kinda|sorta|gonna|wanna|gotta)\b": 1.5,
        # Slang and informal
        r"[aeiou]{2,}|[.!?]{2,}": 0.5,  # Elongation or multiple punctuation
        # Text speak
        r"\b(u|ur|wtf|lol|omg|tbh|ngl|idk)\b": 1.0,
        # Short fragments
        r"^[a-z][a-z0-9]{0,3}$": 1.0,  # Very short messages
    }

    @staticmethod
    def _count_patterns(text: str, patterns: Dict[str, float]) -> float:
        """Count pattern matches and calculate score."""
        score = 0.0
        text_lower = text.lower()
        for pattern, weight in patterns.items():
            matches = len(re.findall(pattern, text_lower))
            score += matches * weight
        return score

    @staticmethod
    def get_llm_score(text: str) -> float:
        """Score how LLM-like the text is (0-100)."""
        llm_score = ResponseValidator._count_patterns(text, ResponseValidator.LLM_PATTERNS)
        casual_score = ResponseValidator._count_patterns(text, ResponseValidator.CASUAL_PATTERNS)

        # Normalize
        if llm_score + casual_score == 0:
            return 25.0  # Middle ground

        return (llm_score / (llm_score + casual_score)) * 100

    @staticmethod
    def is_llm_sounding(text: str, threshold: float = 60.0) -> bool:
        """Check if response sounds like LLM (True = LLM-like, False = human-like)."""
        score = ResponseValidator.get_llm_score(text)
        return score > threshold

    @staticmethod
    def check_length_validity(
        text: str,
        min_words: int = 2,
        max_words: int = 50,
    ) -> Tuple[bool, str]:
        """Validate response length."""
        word_count = len(text.split())

        if word_count < min_words:
            return False, f"Too short ({word_count} words, min {min_words})"
        if word_count > max_words:
            return False, f"Too long ({word_count} words, max {max_words})"

        return True, "Length OK"

    @staticmethod
    def check_coherence(text: str) -> Tuple[bool, float]:
        """Check if response is coherent."""
        if not text or len(text.strip()) < 2:
            return False, 0.0

        # Check for random character sequences
        words = text.split()
        if len(words) > 1:
            word_lengths = [len(w) for w in words]
            avg_length = sum(word_lengths) / len(word_lengths)
            if avg_length > 25:  # Average word too long = likely gibberish
                return False, 0.2

        # Check for repeated patterns
        char_freq = Counter(text.lower())
        most_common_freq = char_freq.most_common(1)[0][1] if char_freq else 0
        repetition_ratio = most_common_freq / len(text) if text else 0

        if repetition_ratio > 0.3:  # More than 30% one character
            return False, 0.3

        return True, 0.9

    @staticmethod
    def check_personality_consistency(
        text: str,
        personality_profile: Dict,
        user_message: str = "",
    ) -> Tuple[bool, str]:
        """Check if response matches personality profile."""
        if not personality_profile:
            return True, "No profile to validate against"

        profile = personality_profile
        issues = []

        # Check emoji usage
        emoji_count = sum(1 for c in text if ord(c) > 0x1F000)
        expected_emoji_rate = profile.get('emoji_patterns', {}).get('emoji_usage_percentage', 20) / 100
        actual_emoji_rate = emoji_count / len(text) if text else 0

        if expected_emoji_rate > 0.1 and emoji_count == 0:
            issues.append("Missing emojis (personality uses them frequently)")

        # Check formality level
        formality = profile.get('response_characteristics', {}).get('formality', 'low')
        if formality == 'low' and ResponseValidator.get_llm_score(text) > 70:
            issues.append("Too formal (personality usually casual)")

        # Check length against profile
        avg_words = profile.get('sentence_structure', {}).get('avg_words_per_message', 10)
        actual_words = len(text.split())
        if abs(actual_words - avg_words) > avg_words * 0.7:
            issues.append(f"Length mismatch (expected ~{avg_words}, got {actual_words})")

        if issues:
            return False, "; ".join(issues)

        return True, "Consistent with personality"

    @staticmethod
    def check_language_consistency(
        text: str,
        expected_language: str = "english",
    ) -> Tuple[bool, str]:
        """Check language consistency."""
        has_hindi = any(0x0900 <= ord(c) <= 0x097F for c in text)
        has_english = any(ord(c) < 128 and c.isalpha() for c in text)

        if expected_language == "hindi" and not has_hindi:
            return False, "Expected Hindi, got English"
        elif expected_language == "english" and has_hindi and not has_english:
            return False, "Expected English, got Hindi"
        elif expected_language == "hinglish":
            if not (has_hindi or has_english):
                return False, "Expected Hinglish mix"

        return True, f"Language consistent ({expected_language})"

    @staticmethod
    def validate_full_response(
        text: str,
        personality_profile: Dict = None,
        user_message: str = "",
        min_words: int = 2,
        max_words: int = 50,
        language: str = "english",
    ) -> Dict:
        """Comprehensive response validation."""
        results = {
            "is_valid": True,
            "llm_score": ResponseValidator.get_llm_score(text),
            "is_llm_sounding": ResponseValidator.is_llm_sounding(text),
            "coherent": False,
            "coherence_score": 0.0,
            "length_valid": False,
            "length_message": "",
            "personality_consistent": False,
            "personality_message": "",
            "language_consistent": False,
            "language_message": "",
            "issues": [],
        }

        # Coherence check
        coherent, score = ResponseValidator.check_coherence(text)
        results["coherent"] = coherent
        results["coherence_score"] = score

        # Length check
        length_valid, msg = ResponseValidator.check_length_validity(text, min_words, max_words)
        results["length_valid"] = length_valid
        results["length_message"] = msg

        # Personality check
        personality_ok, msg = ResponseValidator.check_personality_consistency(
            text, personality_profile, user_message
        )
        results["personality_consistent"] = personality_ok
        results["personality_message"] = msg

        # Language check
        lang_ok, msg = ResponseValidator.check_language_consistency(text, language)
        results["language_consistent"] = lang_ok
        results["language_message"] = msg

        # Aggregate validity
        results["is_valid"] = (
            coherent
            and length_valid
            and not results["is_llm_sounding"]
            and personality_ok
            and lang_ok
        )

        # Collect issues
        if not coherent:
            results["issues"].append("Not coherent")
        if not length_valid:
            results["issues"].append(msg)
        if results["is_llm_sounding"]:
            results["issues"].append(f"LLM-like tone (score: {results['llm_score']:.1f})")
        if not personality_ok:
            results["issues"].append(msg)
        if not lang_ok:
            results["issues"].append(msg)

        return results


class ResponseImprover:
    """Improves responses based on validation feedback."""

    @staticmethod
    def reduce_formality(text: str) -> str:
        """Make text less formal."""
        replacements = {
            r"\bhello there\b": "hi",
            r"\bgood morning\b": "hey",
            r"\bthank you for\b": "thanks for",
            r"\bi appreciate\b": "thanks for",
            r"\bsorry to hear\b": "sucks to hear",
            r"\bi understand\b": "yeah i get it",
            r"\bfurthermore\b": "also",
            r"\bin conclusion\b": "so basically",
            r"\bregarding\b": "about",
            r"\bplease let me know\b": "let me know",
        }

        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    @staticmethod
    def add_casual_flavor(text: str, personality_profile: Dict = None) -> str:
        """Add casual elements like emojis or slang."""
        if not personality_profile:
            return text

        # Add emojis if expected
        emoji_rate = personality_profile.get('emoji_patterns', {}).get('emoji_usage_percentage', 0) / 100
        top_emojis = personality_profile.get('emoji_patterns', {}).get('top_emojis', [])

        if emoji_rate > 0.1 and top_emojis and not any(ord(c) > 0x1F000 for c in text):
            text = f"{text} {top_emojis[0]}"

        # Add casual markers if personality is informal
        formality = personality_profile.get('response_characteristics', {}).get('formality', 'medium')
        if formality == 'low' and not any(w in text.lower() for w in ['yeah', 'lol', 'omg']):
            # Don't add if it would be weird
            if not text.lower().endswith('?'):
                # Safe to add
                pass

        return text

    @staticmethod
    def adjust_length(text: str, target_words: int, tolerance: float = 0.3) -> str:
        """Adjust response length."""
        words = text.split()
        current_words = len(words)
        min_words = int(target_words * (1 - tolerance))
        max_words = int(target_words * (1 + tolerance))

        if min_words <= current_words <= max_words:
            return text

        if current_words < min_words:
            # Too short - add more
            sentences = re.split(r'[.!?]+', text)
            last_sentence = sentences[-1].strip() if sentences[-1].strip() else sentences[-2].strip() if len(sentences) > 1 else text
            follow_ups = ["right?", "you know?", "honestly", "though"]
            selected = follow_ups[current_words % len(follow_ups)]
            return f"{text} {selected}".strip()

        # Too long - truncate at sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = ""
        for sentence in sentences:
            candidate = f"{result} {sentence}".strip()
            if len(candidate.split()) <= max_words:
                result = candidate
            else:
                break

        if not result:
            result = " ".join(words[:max_words])

        return result
