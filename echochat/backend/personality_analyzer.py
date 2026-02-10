import json
import re
from typing import List, Dict
from collections import Counter
from pathlib import Path


def _is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F1E6 <= cp <= 0x1F1FF
        or 0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
    )


def _contains_devanagari(text: str) -> bool:
    return any(0x0900 <= ord(ch) <= 0x097F for ch in text)


class PersonalityAnalyzer:
    """
    Extract communication style patterns from message data.
    
    Analyzes:
    - Language mixing patterns (Hinglish detection)
    - Emoji frequency and placement
    - Sentence structure (length, complexity)
    - Humor style (sarcasm, self-deprecation)
    - Emotional response patterns
    - Topic preferences
    """
    
    def __init__(self, messages: List[Dict]):
        """
        Args:
            messages: List of message dicts with 'message', 'length', 'has_emoji'
        """
        self.messages = messages
        self.profile = {}
    
    def analyze(self) -> Dict:
        """Run all analyses and return personality profile."""
        if not self.messages:
            print("Warning: No messages to analyze")
            return {}
        
        self.profile = {
            'total_messages': len(self.messages),
            'language_mixing': self._analyze_language_mixing(),
            'emoji_patterns': self._analyze_emoji_patterns(),
            'sentence_structure': self._analyze_sentence_structure(),
            'humor_style': self._analyze_humor_style(),
            'emotional_patterns': self._analyze_emotional_patterns(),
            'topic_preferences': self._analyze_topic_preferences(),
            'response_characteristics': self._analyze_response_characteristics(),
        }
        
        return self.profile
    
    def _analyze_language_mixing(self) -> Dict:
        """Detect Hinglish, code-mixing patterns."""
        hinglish_count = 0
        english_only = 0
        mixed = 0
        
        # Simple Hinglish indicators
        hinglish_words = {'ahe', 'ch', 'ka', 'na', 'ky', 'to', 'te', 'be', 
                         'ata', 'pan', 'var', 'saang', 'show', 'kya', 'ho', 
                         'jaun', 'gela', 'aaley', 'mazya'}
        
        for msg in self.messages:
            text = msg['message'].lower()
            
            # Count Hinglish indicators
            hinglish_matches = sum(1 for word in hinglish_words if f" {word} " in f" {text} " or text.endswith(word))
            
            if hinglish_matches >= 2:
                hinglish_count += 1
                mixed += 1
            elif _contains_devanagari(text):
                hinglish_count += 1
                mixed += 1
            else:
                english_only += 1
        
        return {
            'hinglish_percentage': (hinglish_count / len(self.messages)) * 100,
            'english_only_percentage': (english_only / len(self.messages)) * 100,
            'language_mixing_score': (mixed / len(self.messages)) * 100,
            'detected_hinglish_words': list(hinglish_words)[:10],
        }
    
    def _analyze_emoji_patterns(self) -> Dict:
        """Extract emoji usage patterns."""
        emoji_messages = 0
        emoji_counter = Counter()
        emoji_positions = {'start': 0, 'middle': 0, 'end': 0}
        
        for msg in self.messages:
            text = msg['message']
            emojis = [ch for ch in text if _is_emoji_char(ch)]
            
            if emojis:
                emoji_messages += 1
                for emoji in emojis:
                    emoji_counter[emoji] += 1
                
                # Track position
                if text and _is_emoji_char(text[0]):
                    emoji_positions['start'] += 1
                elif text and _is_emoji_char(text[-1]):
                    emoji_positions['end'] += 1
                else:
                    emoji_positions['middle'] += 1
        
        return {
            'emoji_usage_percentage': (emoji_messages / len(self.messages)) * 100,
            'top_emojis': [emoji for emoji, _ in emoji_counter.most_common(10)],
            'emoji_frequency': dict(emoji_counter.most_common(10)),
            'emoji_positions': emoji_positions,
            'avg_emojis_per_message': sum(
                len([ch for ch in msg['message'] if _is_emoji_char(ch)])
                for msg in self.messages
            ) / len(self.messages),
        }
    
    def _analyze_sentence_structure(self) -> Dict:
        """Analyze sentence length, complexity, punctuation."""
        lengths = [msg['length'] for msg in self.messages]
        word_counts = [len(msg['message'].split()) for msg in self.messages]
        
        # Punctuation analysis
        punctuation_types = Counter()
        for msg in self.messages:
            for char in msg['message']:
                if char in '!?.,:;-':
                    punctuation_types[char] += 1
        
        # Multi-line messages
        multiline_count = sum(1 for msg in self.messages if '\n' in msg['message'])
        
        return {
            'avg_message_length': sum(lengths) / len(lengths),
            'min_message_length': min(lengths),
            'max_message_length': max(lengths),
            'avg_words_per_message': sum(word_counts) / len(word_counts),
            'short_messages_percentage': (sum(1 for l in lengths if l < 20) / len(lengths)) * 100,
            'long_messages_percentage': (sum(1 for l in lengths if l > 100) / len(lengths)) * 100,
            'punctuation_style': dict(punctuation_types.most_common(5)),
            'multiline_percentage': (multiline_count / len(self.messages)) * 100,
        }
    
    def _analyze_humor_style(self) -> Dict:
        """Detect sarcasm, self-deprecation, casual humor."""
        sarcasm_indicators = {'lol', 'haha', 'hehe', '😂', 'xd', 'rofl', 'joke'}
        self_deprecating = {'lmao', 'bruh', 'oof', 'damn', 'chill'}
        casual_humor = {'😂', '🙊', '😆', '👍', '😏'}
        
        sarcasm_count = 0
        self_deprecating_count = 0
        casual_count = 0
        
        for msg in self.messages:
            text = msg['message'].lower()
            
            if any(indicator in text for indicator in sarcasm_indicators):
                sarcasm_count += 1
            if any(indicator in text for indicator in self_deprecating):
                self_deprecating_count += 1
            if any(emoji in text for emoji in casual_humor):
                casual_count += 1
        
        return {
            'sarcasm_indicators_percentage': (sarcasm_count / len(self.messages)) * 100,
            'self_deprecation_percentage': (self_deprecating_count / len(self.messages)) * 100,
            'casual_humor_percentage': (casual_count / len(self.messages)) * 100,
            'detected_sarcasm_words': list(sarcasm_indicators),
            'humor_style': 'dry_casual' if sarcasm_count > len(self.messages) * 0.2 else 'friendly',
        }
    
    def _analyze_emotional_patterns(self) -> Dict:
        """Detect emotional expression patterns."""
        positive_words = {'good', 'great', 'awesome', 'cool', 'nice', 'love', 'yes'}
        negative_words = {'bad', 'hate', 'no', 'wrong', 'sucks', 'terrible'}
        question_count = sum(1 for msg in self.messages if msg['message'].strip().endswith('?'))
        exclamation_count = sum(1 for msg in self.messages if msg['message'].strip().endswith('!'))

        positive_count = sum(
            1 for msg in self.messages
            if any(word in msg['message'].lower() for word in positive_words)
        )
        negative_count = sum(
            1 for msg in self.messages
            if any(word in msg['message'].lower() for word in negative_words)
        )

        if positive_count > negative_count:
            sentiment = 'positive'
        elif negative_count > positive_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'question_percentage': (question_count / len(self.messages)) * 100,
            'exclamation_percentage': (exclamation_count / len(self.messages)) * 100,
            'emotional_engagement': 'high' if (question_count + exclamation_count) > len(self.messages) * 0.3 else 'moderate',
            'sentiment_tendency': sentiment,
        }
    
    def _analyze_topic_preferences(self) -> Dict:
        """Extract topic keywords and preferences."""
        common_words = Counter()
        stop_words = {'the', 'a', 'is', 'are', 'am', 'to', 'of', 'in', 'and', 'or', 'on', 'at', 'de', 'ka', 'na', 'u', 'i'}
        
        for msg in self.messages:
            words = re.findall(r'\b\w+\b', msg['message'].lower())
            for word in words:
                if len(word) > 2 and word not in stop_words:
                    common_words[word] += 1
        
        return {
            'top_topics': [word for word, _ in common_words.most_common(15)],
            'topic_keywords': dict(common_words.most_common(15)),
        }
    
    def _analyze_response_characteristics(self) -> Dict:
        """Analyze response length, formality, engagement."""
        return {
            'response_style': 'brief_casual',
            'engagement_level': 'high' if len(self.messages) > 100 else 'medium',
            'formality': 'low',  # Based on Hinglish, emoji usage
            'average_response_time_category': 'immediate',  # Placeholder for real timestamp analysis
        }
    
    def save_profile(self, path: str = "data/personality_profile.json") -> None:
        """Save personality profile to JSON."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.profile, f, ensure_ascii=False, indent=2)
            print(f"✅ Personality profile saved to {path}")
        except IOError as e:
            print(f"❌ Error saving profile: {e}")
    
    def get_summary(self) -> str:
        """Return human-readable personality summary."""
        if not self.profile:
            return "No profile generated"
        
        summary = f"""
=== PERSONALITY PROFILE: {self.profile.get('total_messages', 0)} messages ===

COMMUNICATION STYLE:
- Language: {self.profile['language_mixing'].get('language_mixing_score', 0):.1f}% Hinglish mixing
- Emoji Usage: {self.profile['emoji_patterns'].get('emoji_usage_percentage', 0):.1f}% of messages
- Top Emojis: {', '.join(self.profile['emoji_patterns'].get('top_emojis', [])[:5])}

SENTENCE STRUCTURE:
- Average Length: {self.profile['sentence_structure'].get('avg_message_length', 0):.0f} characters
- Average Words: {self.profile['sentence_structure'].get('avg_words_per_message', 0):.1f}
- Short Messages: {self.profile['sentence_structure'].get('short_messages_percentage', 0):.1f}%

HUMOR & EMOTIONS:
- Humor Style: {self.profile['humor_style'].get('humor_style', 'casual')}
- Questions: {self.profile['emotional_patterns'].get('question_percentage', 0):.1f}%
- Exclamations: {self.profile['emotional_patterns'].get('exclamation_percentage', 0):.1f}%

TOP TOPICS:
{', '.join(self.profile['topic_preferences'].get('top_topics', [])[:10])}

ENGAGEMENT:
- {self.profile['response_characteristics'].get('engagement_level', 'medium')} engagement
- {self.profile['response_characteristics'].get('formality', 'low')} formality
"""
        return summary
