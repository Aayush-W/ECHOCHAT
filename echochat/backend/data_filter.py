"""
Training Data Quality Filter: Removes contaminated or inappropriate training examples.

Detects and filters:
- Formal email templates
- File paths and document references
- System messages and metadata
- Extremely short/long examples
- Off-topic or spam-like content
"""

import re
import logging
from typing import List, Dict, Tuple
from collections import Counter

logger = logging.getLogger("echochat.data_filter")


class TrainingDataFilter:
    """Filters training data to improve model quality."""

    # Patterns that indicate formal email templates
    EMAIL_TEMPLATE_PATTERNS = [
        r"^(subject|to|from|cc|bcc|date):\s*",  # Email headers
        r"dear\s+\w+",  # Formal greeting
        r"(yours|sincerely|regards|respectfully)\s*,?\s*$",  # Formal closure
        r"^(thank you|thanks for|i hope|i appreciate|furthermore|moreover)",  # Formal openers
        r"please (find attached|see attached|refer to|note that|contact)",  # Formal language
        r"^(respectfully|sincerely|warmest regards)",  # Professional signatures
    ]

    # Patterns for document/file references
    FILE_PATTERNS = [
        r"\.pdf|\.doc|\.xlsx|\.ppt|\.txt|\.csv",  # File extensions
        r"(file|document|attachment|linked|drive|link|url)",  # File references
        r"https?://",  # URLs
        r"(page|sheet|cell|row|column|table)",  # Document structure
    ]

    # System/metadata patterns
    SYSTEM_PATTERNS = [
        r"^(\d+/\d+/\d+|Security code)",  # Dates or security messages
        r"(message|chat|notification|alert|system)",  # System references
        r"media (omitted|attached|deleted)",  # WhatsApp media messages
    ]

    # Characteristics of quality training examples
    MIN_LENGTH = 3  # Minimum characters
    MAX_LENGTH = 500  # Maximum characters
    MIN_WORDS = 1  # Minimum meaningful words
    MAX_WORDS = 100  # Maximum words (avoid essays)

    @staticmethod
    def _check_pattern_match(text: str, patterns: List[str]) -> Tuple[bool, float]:
        """
        Check if text matches any patterns.
        Returns (is_match, match_score)
        """
        text_lower = text.lower()
        matches = 0
        total_patterns = len(patterns)

        for pattern in patterns:
            if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
                matches += 1

        match_score = matches / total_patterns if total_patterns > 0 else 0
        is_match = matches > 0

        return is_match, match_score

    @staticmethod
    def _calculate_contamination_score(text: str) -> float:
        """
        Calculate how 'contaminated' a training example is (0-1).
        Higher = more contaminated.
        """
        _, email_score = TrainingDataFilter._check_pattern_match(
            text, TrainingDataFilter.EMAIL_TEMPLATE_PATTERNS
        )
        _, file_score = TrainingDataFilter._check_pattern_match(
            text, TrainingDataFilter.FILE_PATTERNS
        )
        _, system_score = TrainingDataFilter._check_pattern_match(
            text, TrainingDataFilter.SYSTEM_PATTERNS
        )

        # Weighted average
        contamination = (email_score * 0.5 + file_score * 0.3 + system_score * 0.2)
        return min(contamination, 1.0)

    @staticmethod
    def _calculate_quality_score(
        input_text: str,
        output_text: str,
    ) -> float:
        """
        Calculate quality score for a training pair (0-1).
        Higher = better quality.
        """
        quality = 1.0

        # Check lengths
        input_len = len(input_text)
        output_len = len(output_text)

        if input_len < TrainingDataFilter.MIN_LENGTH or input_len > TrainingDataFilter.MAX_LENGTH:
            quality -= 0.2
        if output_len < TrainingDataFilter.MIN_LENGTH or output_len > TrainingDataFilter.MAX_LENGTH:
            quality -= 0.2

        # Check for reasonable word counts
        input_words = len(input_text.split())
        output_words = len(output_text.split())

        if input_words < TrainingDataFilter.MIN_WORDS:
            quality -= 0.1
        if output_words < TrainingDataFilter.MIN_WORDS:
            quality -= 0.1
        if input_words > TrainingDataFilter.MAX_WORDS or output_words > TrainingDataFilter.MAX_WORDS:
            quality -= 0.15

        # Check for diversity (not all same character)
        if input_len > 0:
            char_freq = Counter(input_text)
            most_common_freq = char_freq.most_common(1)[0][1]
            if most_common_freq / input_len > 0.5:  # >50% same character
                quality -= 0.3

        if output_len > 0:
            char_freq = Counter(output_text)
            most_common_freq = char_freq.most_common(1)[0][1]
            if most_common_freq / output_len > 0.5:
                quality -= 0.3

        # Check for repetition (same phrase repeated)
        if input_text == output_text:  # Exact copy is bad
            quality -= 0.4

        return max(quality, 0.0)

    @staticmethod
    def _is_appropriate(
        input_text: str,
        output_text: str,
    ) -> bool:
        """
        Check if training pair is contextually appropriate.
        """
        # Too similar to others isn't good for diversity
        if input_text.lower() == output_text.lower():
            logger.debug("Filtered: Input and output identical")
            return False

        # Check for gibberish
        def has_reasonable_chars(text: str) -> bool:
            text_clean = re.sub(r'[^a-zA-Z0-9\s\u0900-\u097F]', '', text)
            if len(text_clean) < len(text) * 0.5:  # More than 50% special chars
                return False
            return True

        if not has_reasonable_chars(input_text) or not has_reasonable_chars(output_text):
            logger.debug("Filtered: Contains excessive special characters")
            return False

        # Check for spam-like patterns (repeated punctuation)
        if re.search(r'(.)\1{3,}', input_text) or re.search(r'(.)\1{3,}', output_text):
            logger.debug("Filtered: Excessive character repetition")
            return False

        return True

    @staticmethod
    def evaluate_training_pair(
        input_text: str,
        output_text: str,
        verbose: bool = False,
    ) -> Dict:
        """
        Evaluate a training pair and return detailed assessment.
        """
        result = {
            "input": input_text,
            "output": output_text,
            "quality_score": 0.0,
            "contamination_score": 0.0,
            "is_appropriate": False,
            "overall_valid": False,
            "reasons": [],
        }

        # Calculate contamination
        result["contamination_score"] = (
            TrainingDataFilter._calculate_contamination_score(input_text) +
            TrainingDataFilter._calculate_contamination_score(output_text)
        ) / 2

        # Calculate quality
        result["quality_score"] = TrainingDataFilter._calculate_quality_score(
            input_text, output_text
        )

        # Check appropriateness
        result["is_appropriate"] = TrainingDataFilter._is_appropriate(input_text, output_text)

        # Determine overall validity
        if result["contamination_score"] > 0.6:
            result["reasons"].append(
                f"High contamination ({result['contamination_score']:.2f})"
            )
        if result["quality_score"] < 0.4:
            result["reasons"].append(
                f"Low quality ({result['quality_score']:.2f})"
            )
        if not result["is_appropriate"]:
            result["reasons"].append("Not contextually appropriate")

        result["overall_valid"] = (
            result["contamination_score"] < 0.6
            and result["quality_score"] >= 0.4
            and result["is_appropriate"]
        )

        if verbose and result["reasons"]:
            logger.info(f"Filtered pair: {'; '.join(result['reasons'])}")

        return result

    @staticmethod
    def filter_training_data(
        training_pairs: List[Dict],
        min_quality: float = 0.4,
        max_contamination: float = 0.6,
        verbose: bool = False,
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Filter training data and return valid and invalid pairs.

        Returns:
            (valid_pairs, invalid_pairs, statistics)
        """
        valid_pairs = []
        invalid_pairs = []
        stats = {
            "total_input": len(training_pairs),
            "valid_output": 0,
            "filtered_count": 0,
            "avg_quality_before": 0.0,
            "avg_quality_after": 0.0,
            "avg_contamination": 0.0,
            "filters_applied": {
                "low_quality": 0,
                "high_contamination": 0,
                "not_appropriate": 0,
            },
        }

        quality_scores = []
        contamination_scores = []

        for pair in training_pairs:
            input_text = pair.get("input", "").strip()
            output_text = pair.get("output", "").strip()

            if not input_text or not output_text:
                stats["filters_applied"]["not_appropriate"] += 1
                invalid_pairs.append({**pair, "reason": "Empty input or output"})
                continue

            eval_result = TrainingDataFilter.evaluate_training_pair(
                input_text, output_text, verbose=verbose
            )

            quality_scores.append(eval_result["quality_score"])
            contamination_scores.append(eval_result["contamination_score"])

            if not eval_result["overall_valid"]:
                invalid_pairs.append({**pair, **eval_result})
                stats["filtered_count"] += 1

                if eval_result["quality_score"] < min_quality:
                    stats["filters_applied"]["low_quality"] += 1
                if eval_result["contamination_score"] > max_contamination:
                    stats["filters_applied"]["high_contamination"] += 1
                if not eval_result["is_appropriate"]:
                    stats["filters_applied"]["not_appropriate"] += 1
            else:
                valid_pairs.append(pair)
                stats["valid_output"] += 1

        # Calculate statistics
        if quality_scores:
            stats["avg_quality_before"] = sum(quality_scores) / len(quality_scores)
            if valid_pairs:
                valid_quality = [
                    TrainingDataFilter._calculate_quality_score(
                        p.get("input", ""),
                        p.get("output", ""),
                    )
                    for p in valid_pairs
                ]
                stats["avg_quality_after"] = sum(valid_quality) / len(valid_quality)

        if contamination_scores:
            stats["avg_contamination"] = sum(contamination_scores) / len(contamination_scores)

        if verbose:
            logger.info(f"Training data filter statistics: {stats}")

        return valid_pairs, invalid_pairs, stats
