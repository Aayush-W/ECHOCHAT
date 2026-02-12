import json
import re
from typing import List, Dict, Tuple
from collections import Counter
from pathlib import Path
import string


def _is_emoji_char(ch: str) -> bool:
    """
    Detect emoji characters with extended Unicode ranges.
    Includes: emoticons, symbols, pictographs, supplementary planes.
    """
    cp = ord(ch)
    return (
        0x1F1E6 <= cp <= 0x1F1FF  # Regional Indicators
        or 0x1F300 <= cp <= 0x1FAFF  # Emoticons, symbols
        or 0x2600 <= cp <= 0x27BF   # Miscellaneous Symbols
        or 0x1F900 <= cp <= 0x1F9FF  # Supplementary Multilingual Plane
        or 0x1FA00 <= cp <= 0x1FA6F  # Chess Symbols
        or 0x200D in [ord(ch)]       # Zero-width joiner
        or 0xFF00 <= cp <= 0xFFEF    # Halfwidth and Fullwidth Forms
    )


def _contains_devanagari(text: str) -> bool:
    """Detect Devanagari script (Hindi, Marathi, etc.)"""
    return any(0x0900 <= ord(ch) <= 0x097F for ch in text)


def _sanitize_message(text: str) -> str:
    """Remove noise: URLs, mentions, hashtags, extra whitespace."""
    if not text or not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove mentions (@user)
    text = re.sub(r'@\S+', '', text)
    # Remove hashtags but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_words(text: str) -> List[str]:
    """Extract words with proper tokenization."""
    if not text:
        return []
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def _get_word_boundaries(text: str, word: str) -> Tuple[bool, bool]:
    """Check if word exists as complete token (not substring)."""
    pattern = r'\b' + re.escape(word) + r'\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


class PersonalityAnalyzer:
    """
    Extract communication style patterns from message data with quality filtering.
    
    Analyzes:
    - Language mixing patterns (Hinglish detection with confidence scoring)
    - Emoji frequency, placement, and categorization
    - Sentence structure (length, complexity, punctuation patterns)
    - Humor style detection with context awareness
    - Emotional response patterns with sentiment analysis
    - Topic preferences with noise filtering
    - Response characteristics based on actual data
    
    Features:
    - Input validation and sanitization
    - Confidence thresholds to exclude garbage results
    - Quality flags for low-confidence detections
    - Message filtering to remove spam/noise
    """
    
    # Confidence thresholds
    MIN_MESSAGES_FOR_ANALYSIS = 10
    MIN_PATTERN_FREQUENCY = 0.05  # At least 5% of messages
    MIN_CONFIDENCE_SCORE = 0.6    # 60% confidence threshold
    
    # Improved Hinglish word list with better indicators
    HINGLISH_WORDS = {
        # Common Hinglish patterns
        'kya', 'haan', 'nahi', 'bhai', 'yaar', 'bro', 'arre', 'acha',
        'chalega', 'karenge', 'bolthe', 'dekho', 'jaao', 'gela', 
        'kadam', 'samay', 'maza', 'scene', 'logic', 'vibe', 'boss',
        # Marathi patterns  
        'ahe', 'aahe', 'hote', 'kaay', 'pan', 'var', 'sangu',
        # Common code-mixing
        'ok', 'na', 'kk', 'kkk', 'done', 'busy', 'sleep', 'tired'
    }
    
    # More comprehensive stop words (excluding single letters)
    STOP_WORDS = {
        'the', 'a', 'is', 'are', 'am', 'to', 'of', 'in', 'and', 'or', 
        'on', 'at', 'be', 'by', 'for', 'with', 'as', 'was', 'were',
        'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'just', 'all', 'each', 'every', 'some', 'any', 'no', 'nor',
        'de', 'ka', 'na', 'se', 'tak',  # Hindi particles
        'abt', 'tho', 'so', 'but', 'bc', 'bro'  # Common slang
    }
    
    def __init__(self, messages: List[Dict]):
        """
        Args:
            messages: List of message dicts with 'message', 'length', optional 'timestamp'
        """
        self.messages = messages if messages else []
        self.profile = {}
        self.metadata = {}
        self._validate_messages()
    
    def _validate_messages(self) -> None:
        """Validate message structure and filter invalid entries."""
        valid_messages = []
        for msg in self.messages:
            if not isinstance(msg, dict):
                continue
            if not isinstance(msg.get('message'), str) or not msg['message'].strip():
                continue
            valid_messages.append(msg)
        
        self.messages = valid_messages
        self.metadata['valid_messages'] = len(self.messages)
        self.metadata['invalid_messages'] = len(self.messages) - len(valid_messages)
    
    def analyze(self) -> Dict:
        """Run all analyses and return personality profile with quality metrics."""
        if len(self.messages) < self.MIN_MESSAGES_FOR_ANALYSIS:
            return {
                'error': f'Insufficient data ({len(self.messages)} messages)',
                'min_required': self.MIN_MESSAGES_FOR_ANALYSIS,
                'quality_flag': 'LOW_DATA_VOLUME'
            }
        
        self.profile = {
            'total_messages': len(self.messages),
            'language_mixing': self._analyze_language_mixing(),
            'emoji_patterns': self._analyze_emoji_patterns(),
            'sentence_structure': self._analyze_sentence_structure(),
            'humor_style': self._analyze_humor_style(),
            'emotional_patterns': self._analyze_emotional_patterns(),
            'topic_preferences': self._analyze_topic_preferences(),
            'response_characteristics': self._analyze_response_characteristics(),
            'quality_metrics': self._calculate_quality_metrics(),
        }
        
        return self.profile
    
    def _analyze_language_mixing(self) -> Dict:
        """
        Detect Hinglish and code-mixing patterns with improved word boundary detection.
        Returns confidence scores for reliability.
        """
        hinglish_count = 0
        english_only = 0
        mixed = 0
        detected_words = Counter()
        
        for msg in self.messages:
            text = msg['message'].lower()
            
            # Check for Devanagari script
            if _contains_devanagari(text):
                hinglish_count += 1
                mixed += 1
                continue
            
            # Check for Hinglish word indicators with proper boundaries
            hinglish_matches = 0
            for word in self.HINGLISH_WORDS:
                if _get_word_boundaries(text, word):
                    hinglish_matches += 1
                    detected_words[word] += 1
            
            # Require at least 2 indicators or known pattern
            if hinglish_matches >= 2:
                hinglish_count += 1
                mixed += 1
            else:
                english_only += 1
        
        total = len(self.messages)
        hinglish_pct = (hinglish_count / total) * 100 if total > 0 else 0
        
        # Calculate confidence score
        confidence = min(1.0, (hinglish_count / max(5, total)) * 0.5 + 
                        (len(detected_words) / len(self.HINGLISH_WORDS)) * 0.5)
        
        return {
            'hinglish_percentage': hinglish_pct,
            'english_only_percentage': (english_only / total) * 100,
            'language_mixing_score': (mixed / total) * 100,
            'detected_hinglish_words': list(detected_words.most_common(10)),
            'confidence_score': confidence,
            'quality_flag': 'HIGH' if confidence > self.MIN_CONFIDENCE_SCORE else 'LOW'
        }
    
    def _analyze_emoji_patterns(self) -> Dict:
        """
        Extract emoji usage patterns with position tracking.
        Fixed: Properly count messages with multiple emoji positions.
        """
        emoji_messages = 0
        emoji_counter = Counter()
        emoji_positions = {'start': 0, 'middle': 0, 'end': 0}
        emojis_per_message = []
        
        for msg in self.messages:
            text = msg['message']
            emojis = [ch for ch in text if _is_emoji_char(ch)]
            
            if emojis:
                emoji_messages += 1
                emojis_per_message.append(len(emojis))
                
                # Count all emojis in Counter (fixed from before)
                for emoji in emojis:
                    emoji_counter[emoji] += 1
                
                # Check positions - FIX: now correctly identifies all positions
                has_start = text and _is_emoji_char(text[0])
                has_end = text and _is_emoji_char(text[-1])
                has_middle = any(_is_emoji_char(ch) for ch in text[1:-1] if text) if len(text) > 1 else False
                
                if has_start:
                    emoji_positions['start'] += 1
                if has_end:
                    emoji_positions['end'] += 1
                if has_middle:
                    emoji_positions['middle'] += 1
        
        total = len(self.messages)
        emoji_usage_pct = (emoji_messages / total) * 100 if total > 0 else 0
        
        return {
            'emoji_usage_percentage': emoji_usage_pct,
            'top_emojis': [emoji for emoji, _ in emoji_counter.most_common(10)],
            'emoji_frequency': dict(emoji_counter.most_common(10)),
            'emoji_positions': emoji_positions,
            'avg_emojis_per_message': sum(emojis_per_message) / len(emojis_per_message) if emojis_per_message else 0,
            'emoji_usage_intensity': self._classify_emoji_usage(emoji_usage_pct),
        }
    
    def _classify_emoji_usage(self, percentage: float) -> str:
        """Classify emoji usage intensity."""
        if percentage == 0:
            return 'none'
        elif percentage < 10:
            return 'rare'
        elif percentage < 30:
            return 'occasional'
        elif percentage < 60:
            return 'frequent'
        else:
            return 'heavy'
    
    def _analyze_sentence_structure(self) -> Dict:
        """
        Analyze sentence length, complexity, punctuation with refined metrics.
        """
        lengths = [msg.get('length', len(msg['message'])) for msg in self.messages]
        word_counts = [len(_extract_words(msg['message'])) for msg in self.messages]
        
        # Punctuation analysis with improved detection
        punctuation_types = Counter()
        exclamation_count = 0
        question_count = 0
        
        for msg in self.messages:
            text = msg['message'].strip()
            
            if text.endswith('!'):
                exclamation_count += 1
            elif text.endswith('?'):
                question_count += 1
            
            for char in text:
                if char in '!?.,:;-…':
                    punctuation_types[char] += 1
        
        # Multi-line messages
        multiline_count = sum(1 for msg in self.messages if '\n' in msg['message'])
        
        total = len(self.messages)
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        
        return {
            'avg_message_length': avg_length,
            'min_message_length': min(lengths) if lengths else 0,
            'max_message_length': max(lengths) if lengths else 0,
            'median_message_length': sorted(lengths)[len(lengths)//2] if lengths else 0,
            'avg_words_per_message': avg_words,
            'short_messages_percentage': (sum(1 for l in lengths if l < 20) / total) * 100,
            'medium_messages_percentage': (sum(1 for l in lengths if 20 <= l <= 100) / total) * 100,
            'long_messages_percentage': (sum(1 for l in lengths if l > 100) / total) * 100,
            'punctuation_style': dict(punctuation_types.most_common(5)),
            'exclamation_percentage': (exclamation_count / total) * 100,
            'question_percentage': (question_count / total) * 100,
            'multiline_percentage': (multiline_count / total) * 100,
            'message_length_category': self._classify_message_length(avg_length),
        }
    
    def _classify_message_length(self, avg: float) -> str:
        """Classify typical message length."""
        if avg < 20:
            return 'very_brief'
        elif avg < 50:
            return 'brief'
        elif avg < 100:
            return 'medium'
        elif avg < 200:
            return 'long'
        else:
            return 'very_long'
    
    def _analyze_humor_style(self) -> Dict:
        """
        Detect humor style with improved pattern matching and confidence scoring.
        """
        sarcasm_patterns = {'lol', 'haha', 'hehe', '😂', 'xd', 'rofl', 'literally', 'yeah right', 'sure'}
        self_deprecating_patterns = {'oof', 'damn', 'fail', 'broke', 'stupid me'}
        casual_humor_emojis = {'😂', '🙊', '😆', '👍', '😏', '🤦', '💀'}
        
        sarcasm_count = 0
        self_deprecating_count = 0
        casual_count = 0
        
        for msg in self.messages:
            text = msg['message'].lower()
            
            # Sarcasm detection with boundaries
            if any(_get_word_boundaries(text, word) for word in sarcasm_patterns if len(word) > 2):
                sarcasm_count += 1
            if any(word in text for word in ['lol', 'haha', 'hehe', 'xd']):
                sarcasm_count += 1
            
            # Self-deprecation
            if any(_get_word_boundaries(text, word) for word in self_deprecating_patterns):
                self_deprecating_count += 1
            
            # Casual humor emojis
            if any(emoji in text for emoji in casual_humor_emojis):
                casual_count += 1
        
        total = len(self.messages)
        sarcasm_pct = (sarcasm_count / total) * 100 if total > 0 else 0
        self_deprecating_pct = (self_deprecating_count / total) * 100 if total > 0 else 0
        casual_pct = (casual_count / total) * 100 if total > 0 else 0
        
        # Determine primary humor style
        humor_style = self._classify_humor_style(sarcasm_pct, self_deprecating_pct, casual_pct)
        
        return {
            'sarcasm_indicators_percentage': sarcasm_pct,
            'self_deprecation_percentage': self_deprecating_pct,
            'casual_humor_percentage': casual_pct,
            'detected_humor_patterns': {
                'sarcasm': sarcasm_count,
                'self_deprecating': self_deprecating_count,
                'casual': casual_count
            },
            'humor_style': humor_style,
            'humor_confidence': max(sarcasm_pct, self_deprecating_pct, casual_pct) / 100,
        }
    
    def _classify_humor_style(self, sarcasm: float, deprecating: float, casual: float) -> str:
        """Classify primary humor style."""
        if sarcasm > 20:
            return 'sarcastic'
        elif deprecating > 15:
            return 'self_deprecating'
        elif casual > 15:
            return 'casual'
        else:
            return 'minimal_humor'
    
    def _analyze_emotional_patterns(self) -> Dict:
        """
        Analyze emotional expression with context-aware sentiment detection.
        Improved to handle negation and emoji sentiment.
        """
        positive_words = {'good', 'great', 'awesome', 'cool', 'nice', 'love', 'yes', 'happy', 'thanks', 'perfect', 'amazing'}
        negative_words = {'bad', 'hate', 'no', 'wrong', 'sucks', 'terrible', 'sad', 'angry', 'hate', 'awful', 'worst'}
        positive_emojis = {'😊', '😂', '❤️', '😍', '🥰', '👍', '💯', '🔥'}
        negative_emojis = {'😢', '😞', '😡', '😤', '😭', '😠', '💔', '😒'}
        
        question_count = sum(1 for msg in self.messages if msg['message'].strip().endswith('?'))
        exclamation_count = sum(1 for msg in self.messages if msg['message'].strip().endswith('!'))
        
        positive_count = 0
        negative_count = 0
        
        for msg in self.messages:
            text = msg['message'].lower()
            
            # Simple negation handling
            has_negation = any(word in text for word in ['not', 'never', 'dont', "don't", 'no', 'neither'])
            
            # Count words
            positive_matches = sum(1 for word in positive_words if _get_word_boundaries(text, word))
            negative_matches = sum(1 for word in negative_words if _get_word_boundaries(text, word))
            
            # Adjust for negation
            if has_negation:
                positive_matches, negative_matches = negative_matches, positive_matches
            
            if positive_matches > 0:
                positive_count += 1
            if negative_matches > 0:
                negative_count += 1
            
            # Check emojis
            if any(emoji in text for emoji in positive_emojis):
                positive_count += 0.5
            if any(emoji in text for emoji in negative_emojis):
                negative_count += 0.5
        
        total = len(self.messages)
        positive_pct = (positive_count / total) * 100 if total > 0 else 0
        negative_pct = (negative_count / total) * 100 if total > 0 else 0
        
        # Sentiment determination
        if positive_pct > negative_pct + 5:
            sentiment = 'positive'
        elif negative_pct > positive_pct + 5:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        engagement = 'very_high' if (question_count + exclamation_count) > total * 0.5 else \
                    'high' if (question_count + exclamation_count) > total * 0.3 else \
                    'moderate'
        
        return {
            'question_percentage': (question_count / total) * 100,
            'exclamation_percentage': (exclamation_count / total) * 100,
            'emotional_engagement': engagement,
            'sentiment_tendency': sentiment,
            'positive_percentage': positive_pct,
            'negative_percentage': negative_pct,
            'sentiment_confidence': abs(positive_pct - negative_pct) / 100,
        }
    
    def _analyze_topic_preferences(self) -> Dict:
        """
        Extract topic keywords with improved filtering.
        Removes noise, filters by frequency, and excludes low-quality results.
        """
        word_counter = Counter()
        
        for msg in self.messages:
            # Sanitize to remove URLs, mentions, etc.
            sanitized = _sanitize_message(msg['message'])
            words = _extract_words(sanitized)
            
            for word in words:
                # Length filter and stop words filter
                if len(word) > 2 and word not in self.STOP_WORDS:
                    word_counter[word] += 1
        
        total = len(self.messages)
        min_frequency = max(2, int(total * 0.02))  # At least 2% frequency
        
        # Filter by frequency threshold
        filtered_topics = [
            (word, count) for word, count in word_counter.most_common(20)
            if count >= min_frequency
        ]
        
        # Only return if quality threshold met
        if len(filtered_topics) < 5:
            return {
                'top_topics': [],
                'topic_keywords': {},
                'quality_flag': 'INSUFFICIENT_DATA'
            }
        
        return {
            'top_topics': [word for word, _ in filtered_topics[:10]],
            'topic_keywords': dict(filtered_topics),
            'topic_count': len(filtered_topics),
            'quality_flag': 'HIGH'
        }
    
    def _analyze_response_characteristics(self) -> Dict:
        """
        Analyze actual response characteristics from data.
        Fixed: Previously returned hardcoded values, now truly analyzes.
        """
        lengths = [msg.get('length', len(msg['message'])) for msg in self.messages]
        
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        
        # Determine message length category
        if avg_length < 30:
            response_style = 'brief_casual'
        elif avg_length < 80:
            response_style = 'moderate'
        else:
            response_style = 'detailed'
        
        # Determine formality from language patterns
        hinglish_pct = self.profile.get('language_mixing', {}).get('hinglish_percentage', 0)
        emoji_pct = self.profile.get('emoji_patterns', {}).get('emoji_usage_percentage', 0)
        
        if hinglish_pct > 50 and emoji_pct > 40:
            formality = 'very_casual'
        elif hinglish_pct > 30 or emoji_pct > 30:
            formality = 'casual'
        elif hinglish_pct < 10 and emoji_pct < 10:
            formality = 'formal'
        else:
            formality = 'semi_formal'
        
        # Engagement level from message frequency and structure
        question_pct = self.profile.get('emotional_patterns', {}).get('question_percentage', 0)
        exclamation_pct = self.profile.get('emotional_patterns', {}).get('exclamation_percentage', 0)
        
        if (question_pct + exclamation_pct) > 50:
            engagement = 'very_high'
        elif (question_pct + exclamation_pct) > 30:
            engagement = 'high'
        elif (question_pct + exclamation_pct) > 10:
            engagement = 'moderate'
        else:
            engagement = 'low'
        
        return {
            'response_style': response_style,
            'engagement_level': engagement,
            'formality': formality,
            'avg_message_length': avg_length,
            'message_length_category': self._classify_message_length(avg_length),
            'typical_response_length': self._get_length_description(avg_length),
        }
    
    def _get_length_description(self, avg: float) -> str:
        if avg < 10:
            return 'single_word_to_few_words'
        elif avg < 30:
            return 'short_4_6_words'
        elif avg < 60:
            return 'medium_7_15_words'
        elif avg < 150:
            return 'long_16_plus_words'
        else:
            return 'very_long_detailed'
    
    def _calculate_quality_metrics(self) -> Dict:
        """
        Calculate overall quality metrics and confidence scores.
        Flag low-quality or unreliable results.
        """
        total = len(self.messages)
        
        # Data quality metrics
        data_quality = 'EXCELLENT' if total > 100 else \
                      'GOOD' if total > 50 else \
                      'FAIR' if total > 20 else \
                      'POOR'
        
        # Check for garbage/spam
        spam_indicators = 0
        for msg in self.messages:
            text = msg['message']
            # Check for repeated characters (spam indicator)
            if re.search(r'(\w)\1{5,}', text):
                spam_indicators += 1
            # Check for too many special characters
            if len(re.findall(r'[^a-zA-Z0-9\s]', text)) / max(1, len(text)) > 0.5:
                spam_indicators += 1
        
        spam_percentage = (spam_indicators / total) * 100 if total > 0 else 0
        
        return {
            'data_quality': data_quality,
            'total_messages_analyzed': total,
            'spam_messages_percentage': spam_percentage,
            'quality_flag': 'WARNING' if spam_percentage > 10 else 'OK',
            'reliability': 'HIGH' if total > 50 and spam_percentage < 5 else 'MEDIUM' if total > 20 else 'LOW'
        }
    
    def save_profile(self, path: str = "data/personality_profile.json") -> None:
        """Save personality profile to JSON with metadata."""
        try:
            output = {
                'timestamp': str(Path.cwd()),
                'analysis_metadata': self.metadata,
                'personality_profile': self.profile,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"✅ Personality profile saved to {path}")
        except IOError as e:
            print(f"❌ Error saving profile: {e}")
    
    def get_summary(self) -> str:
        """Return human-readable personality summary with quality flags."""
        if not self.profile:
            return "No profile generated"
        
        quality = self.profile.get('quality_metrics', {})
        lang = self.profile.get('language_mixing', {})
        emoji = self.profile.get('emoji_patterns', {})
        structure = self.profile.get('sentence_structure', {})
        humor = self.profile.get('humor_style', {})
        emotion = self.profile.get('emotional_patterns', {})
        topics = self.profile.get('topic_preferences', {})
        response = self.profile.get('response_characteristics', {})
        
        summary = f"""
{'='*60}
PERSONALITY PROFILE ANALYSIS
{'='*60}
Messages Analyzed: {self.profile.get('total_messages', 0)} | Quality: {quality.get('data_quality', 'UNKNOWN')}
{'⚠️  WARNING: High spam content!' if quality.get('spam_messages_percentage', 0) > 10 else '✅ Data quality good'}
{'⚠️  WARNING: Low confidence results - consider more messages' if quality.get('reliability') == 'LOW' else ''}

{'='*60}
COMMUNICATION STYLE:
{'='*60}
Language: {lang.get('language_mixing_score', 0):.1f}% Hinglish mixing (Confidence: {lang.get('confidence_score', 0):.1f})
Emoji Usage: {emoji.get('emoji_usage_intensity', 'none')} ({emoji.get('emoji_usage_percentage', 0):.1f}%)
Top Emojis: {', '.join(emoji.get('top_emojis', [])[:5]) or 'None'}

{'='*60}
SENTENCE STRUCTURE:
{'='*60}
Message Length: {response.get('message_length_category', 'unknown')}
Average: {structure.get('avg_message_length', 0):.0f} characters | {structure.get('avg_words_per_message', 0):.1f} words
Questions: {emotion.get('question_percentage', 0):.1f}% | Exclamations: {emotion.get('exclamation_percentage', 0):.1f}%
Punctuation: {', '.join(f"{k}({v})" for k, v in list(structure.get('punctuation_style', {}).items())[:3]) or 'minimal'}

{'='*60}
HUMOR & EMOTION:
{'='*60}
Humor Style: {humor.get('humor_style', 'minimal_humor').replace('_', ' ').title()}
Sentiment: {emotion.get('sentiment_tendency', 'neutral').title()}
Engagement: {response.get('engagement_level', 'unknown').replace('_', ' ').title()}

{'='*60}
TOP TOPICS:
{'='*60}
{', '.join(topics.get('top_topics', [])[:10]) or 'Insufficient data'}

{'='*60}
RELIABILITY: {quality.get('reliability', 'UNKNOWN')}
{'='*60}
"""
        return summary
