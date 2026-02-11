from typing import Dict, List, Optional, Tuple
from pathlib import Path
import requests
import json

from .config import (
    LLM_API_ENDPOINT,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    SYSTEM_PROMPT,
    STYLE_PROMPT_INJECTION,
    MEMORY_PROMPT_INJECTION,
    LOCAL_INFERENCE_ENABLED,
    LOCAL_BASE_MODEL,
    LOCAL_ADAPTER_PATH,
    LOCAL_DEVICE_MAP,
)
from .text_filter import classify_intent, contains_banned_output

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    PeftModel = None

from .response_validator import ResponseValidator, ResponseImprover
from .response_refiner import ResponseRefiner
from .db_manager import DatabaseManager

# Initialize database for logging
_db_manager = None
try:
    _db_manager = DatabaseManager()
except Exception as e:
    pass


class Responder:
    """
    LLM-based response generator via Ollama API.
    
    Workflow:
    1. Take user message
    2. Retrieve personality style
    3. Retrieve relevant memories
    4. Construct prompt with context
    5. Query LLM
    6. Apply safety filters
    7. Return response
    """
    
    def __init__(
        self,
        personality_profile: Dict = None,
        memory_store = None,
        example_store = None,
        persona_pack: Dict = None,
        model_name: str = LLM_MODEL_NAME,
        api_endpoint: str = LLM_API_ENDPOINT,
        local_adapter_path: Optional[str] = None,
        local_base_model: Optional[str] = None,
        use_local_inference: Optional[bool] = None,
    ):
        """
        Args:
            personality_profile: From PersonalityAnalyzer
            memory_store: MemoryStore instance
            model_name: Ollama model (e.g., "mistral", "llama2")
            api_endpoint: Ollama API URL
        """
        self.personality_profile = personality_profile or {}
        self.memory_store = memory_store
        self.example_store = example_store
        self.persona_pack = persona_pack or {}
        self.model_name = model_name
        self.api_endpoint = api_endpoint

        self.local_adapter_path = local_adapter_path or LOCAL_ADAPTER_PATH
        self.local_base_model = local_base_model or LOCAL_BASE_MODEL
        if use_local_inference is None:
            use_local_inference = LOCAL_INFERENCE_ENABLED
        self.use_local_inference = bool(use_local_inference and self.local_adapter_path)
        if self.use_local_inference and not Path(self.local_adapter_path).exists():
            print(f"Local adapter not found: {self.local_adapter_path}")
            self.use_local_inference = False

        self._local_pipeline_cache: Dict[Tuple[str, str], Tuple] = {}
        self._refiner = ResponseRefiner(
            memory_store=self.memory_store,
            example_store=self.example_store,
        )

        if not self.use_local_inference:
            # Record whether Ollama is reachable so we can provide a fallback
            try:
                self._ollama_available = bool(self._test_connection())
            except Exception:
                self._ollama_available = False
    
    def _test_connection(self) -> bool:
        """Test if Ollama API is running."""
        try:
            response = requests.get(f"{self.api_endpoint}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"Connected to Ollama at {self.api_endpoint}")
                return True
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to Ollama at {self.api_endpoint}")
            print("   Start Ollama with: ollama serve")
            print(f"   Or pull model: ollama pull {self.model_name}")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def generate_response(
        self,
        user_message: str,
        include_memories: bool = True,
        verbose: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Generate response to user message.
        
        Args:
            user_message: Input from user
            include_memories: Whether to retrieve context
            verbose: Print prompt for debugging
        
        Returns:
            {
                'response': generated text,
                'reasoning': internal thinking,
                'memories_used': list of retrieved memories,
                'model': model name,
                'success': bool
            }
        """
        
        intent = classify_intent(user_message)
        language_mode = self._detect_language_mode(user_message)

        # Retrieve memories if available
        memories_context = ""
        memories_used = []
        style_examples: List[str] = []
        examples_context = ""
        
        if self.memory_store:
            style_examples = self.memory_store.sample_style_examples(count=4, max_length=120)

        use_memories = include_memories
        if len(user_message.split()) <= 4:
            use_memories = False

        if use_memories and self.memory_store:
            search_results = self.memory_store.search_with_context(
                user_message,
                top_k=4,
                max_length=240,
                intent=intent,
            )
            memories_context = self.memory_store.format_context(
                search_results,
                include_meta=False,
                max_chars=220,
            )
            memories_used = [result['message']['text'] for result in search_results]

        example_results: List[Dict] = []
        if self.example_store:
            example_results = self.example_store.search(
                user_message,
                top_k=2,
                similarity_threshold=0.3,
                max_input_length=200,
                max_output_length=200,
                intent=intent,
            )
            examples_context = self.example_store.format_examples(
                example_results,
                max_chars_input=140,
                max_chars_output=140,
            )

        if intent == "chat" and len(user_message.split()) <= 4:
            fast_reply = self._fast_chat_reply(user_message, example_results)
            if fast_reply:
                return {
                    'response': fast_reply,
                    'reasoning': None,
                    'memories_used': memories_used,
                    'model': self.model_name,
                    'success': True,
                }
        
        attempt = 0
        response_text = ""
        lint_failed = False
        banned_failed = False
        while attempt < 2:
            extra_constraints = (
                self._llm_lint_constraints(avoid_file_terms=(intent != "file"))
                if attempt == 1
                else ""
            )

            # Build prompt
            prompt = self._build_prompt(
                user_message,
                memories_context,
                style_examples,
                examples_context,
                extra_constraints,
            )

            if verbose:
                print("\n" + "="*60)
                print("PROMPT SENT TO LLM:")
                print("="*60)
                print(prompt)
                print("="*60 + "\n")

            # Query LLM or provide a graceful fallback when Ollama/local model is unavailable
            try:
                if self.use_local_inference:
                    try:
                        response_text = self._query_local(prompt)
                    except Exception as exc:
                        print(f"Local inference failed, falling back to Ollama: {exc}")
                        if getattr(self, "_ollama_available", True):
                            response_text = self._query_ollama(prompt)
                        else:
                            response_text = ""
                else:
                    if not getattr(self, "_ollama_available", True):
                        # Ollama not reachable - produce a safe fallback using examples/memories
                        if example_results and example_results[0].get("output"):
                            response_text = example_results[0].get("output", "")
                        elif memories_used:
                            response_text = memories_used[0]
                        else:
                            mode = self._detect_language_mode(user_message)
                            response_text = "haan bolo" if mode in ("hinglish", "hindi") else "yeah?"
                    else:
                        response_text = self._query_ollama(prompt)
            except Exception as e:
                print(f"Error generating response: {e}")
                return {
                    'response': f"Sorry, I encountered an error: {str(e)}",
                    'reasoning': None,
                    'memories_used': [],
                    'model': self.model_name,
                    'success': False,
                }

            # Apply safety filters + refine
            response_text = self._apply_safety_filters(response_text)
            response_text = self._refine_response(user_message, response_text, example_results)
            response_text, _ = self._refiner.refine(
                user_message=user_message,
                response=response_text,
                intent=intent,
                example_results=example_results,
                language_mode=language_mode,
            )

            validator_result = ResponseValidator.validate_full_response(
                response_text,
                personality_profile=self.personality_profile,
                user_message=user_message,
                min_words=2,
                max_words=50,
                language=language_mode,
            )

            lint_failed = validator_result["is_llm_sounding"]
            banned_failed = contains_banned_output(response_text) and intent != "file"

            # Log response quality if database is available
            if session_id and _db_manager:
                try:
                    _db_manager.log_response_quality(
                        session_id=session_id,
                        user_message=user_message,
                        ai_response=response_text,
                        is_llm_sounding=lint_failed,
                    )
                except Exception:
                    pass

            if not lint_failed and not banned_failed:
                break

            # prepare for retry
            attempt += 1

            # Improve response if it failed validation
            if attempt == 1 and (lint_failed or not validator_result["is_valid"]):
                if lint_failed:
                    response_text = ResponseImprover.reduce_formality(response_text)
                if not validator_result["length_valid"]:
                    min_words, max_words = self._target_word_range()
                    response_text = ResponseImprover.adjust_length(
                        response_text,
                        int(min_words + (max_words - min_words) / 2),
                    )

        return {
            'response': response_text,
            'reasoning': self._extract_reasoning(response_text),
            'memories_used': memories_used,
            'model': self.model_name,
            'success': True,
        }
        
    
    def _build_prompt(
        self,
        user_message: str,
        memories_context: str,
        style_examples: List[str],
        examples_context: str,
        extra_constraints: str,
    ) -> str:
        """Construct full prompt with context injection."""
        
        # Inject personality style
        style_context = STYLE_PROMPT_INJECTION
        style_profile = self._format_style_profile()
        style_examples_block = self._format_style_examples(style_examples)
        persona_block = self._format_persona_pack()
        
        # Inject memory context
        memory_injection = ""
        if memories_context:
            memory_injection = MEMORY_PROMPT_INJECTION.format(
                memories_context=memories_context
            )

        examples_injection = ""
        if examples_context:
            examples_injection = (
                "EXAMPLE PAIRS (for meaning + tone, do NOT copy):\n"
                f"{examples_context}"
            )

        min_words, max_words = self._target_word_range()
        language_mode = self._detect_language_mode(user_message)
        response_targets = (
            "RESPONSE TARGETS:\n"
            f"- Language: {language_mode}\n"
            f"- Length: {min_words}-{max_words} words\n"
            "- Keep it natural and casual."
        )
        
        # Build full prompt
        sections = [
            SYSTEM_PROMPT,
            style_context,
            style_profile,
            style_examples_block,
            persona_block,
            response_targets,
            extra_constraints,
            examples_injection,
            memory_injection,
            "==== CURRENT CONVERSATION ====",
            f"User: {user_message}",
            "---",
            "Assistant:",
        ]

        prompt = "\n\n".join(section for section in sections if section)
        
        return prompt

    def _llm_lint_constraints(self, avoid_file_terms: bool = False) -> str:
        rules = [
            "ANTI-LLM STYLE RULES:",
            "- No formal tone, no corporate words, no bullet lists.",
            "- No greetings like 'Hello there' or 'Hope you are well'.",
            "- Do not mention AI, memory, or context.",
            "- Keep it short, casual, and chatty.",
        ]
        if avoid_file_terms:
            rules.append("- Do not mention files, attachments, or deleted messages.")
        return "\n".join(rules)

    def _fast_chat_reply(self, user_message: str, example_results: List[Dict]) -> str:
        if self._is_greeting(user_message):
            return self._greeting_template(user_message)

        if example_results:
            top = example_results[0]
            similarity = top.get("similarity", 0)
            overlap = self._token_overlap(user_message, top.get("input", ""))
            if similarity >= 0.55 and overlap >= 0.2:
                candidate = top.get("output", "").strip()
                if candidate and not contains_banned_output(candidate):
                    return self._refine_response(user_message, candidate, example_results)

        return self._short_ack_template(user_message)

    def _greeting_template(self, user_message: str) -> str:
        mode = self._detect_language_mode(user_message)
        lowered = user_message.lower()
        wants_bhai = "bhai" in lowered or "bro" in lowered
        preferred = self.persona_pack.get("preferred_greeting")
        if preferred:
            reply = preferred.strip()
            if reply:
                if wants_bhai and "bhai" not in reply.lower():
                    reply = f"{reply} bhai"
                return reply

        if mode == "hinglish":
            options = [
                "mast hu{bhai} tu kaisa hai?",
                "theek hu{bhai}, tu bata?",
                "sahi hu{bhai}. tu kaisa hai?",
            ]
        elif mode == "hindi":
            options = [
                "main theek hu, tum kaise ho?",
                "sab theek hai, tum batao?",
            ]
        else:
            options = [
                "I'm good. You?",
                "All good here. You?",
            ]

        idx = abs(hash(user_message)) % len(options)
        reply = options[idx]
        bhai = " bhai" if wants_bhai and "bhai" not in reply else ""
        reply = reply.format(bhai=bhai).strip()
        return reply

    def _short_ack_template(self, user_message: str) -> str:
        mode = self._detect_language_mode(user_message)
        if mode == "hinglish":
            return "haan bol bhai"
        if mode == "hindi":
            return "haan bolo"
        return "yeah?"

    def _is_greeting(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        greetings = {
            "hi", "hello", "hey", "yo", "sup", "hru", "bro", "bhai", "yaar",
            "kya haal", "kya hal", "kya scene", "kya chal", "kaisa", "kaisi",
            "kaise", "kaisa hai", "kaisi hai", "kaise ho", "kaisa ho",
        }
        if any(phrase in lowered for phrase in greetings):
            return True
        tokens = lowered.split()
        return len(tokens) <= 3 and any(tok in greetings for tok in tokens)

    def _token_overlap(self, a: str, b: str) -> float:
        a_tokens = {tok for tok in a.lower().split() if tok}
        b_tokens = {tok for tok in b.lower().split() if tok}
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)

    def _is_llm_sounding(self, text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return False

        llm_phrases = [
            "as an ai",
            "i'm here to help",
            "hope you're doing well",
            "certainly",
            "in conclusion",
            "furthermore",
            "moreover",
            "it's important to",
            "please let me know",
            "happy to help",
            "let me know if you have",
            "if you have any questions",
            "i can assist with",
            "i understand your",
            "i'm sorry to hear",
            "thank you for sharing",
        ]
        if any(phrase in lowered for phrase in llm_phrases):
            return True

        if len(text.split()) >= 30:
            return True

        if "\n- " in text or "\n1." in text:
            return True

        # Too much punctuation or overly formal structure
        if text.count(",") >= 4 and text.count("!") == 0:
            return True

        return False

    def _is_emoji_char(self, ch: str) -> bool:
        cp = ord(ch)
        return (
            0x1F1E6 <= cp <= 0x1F1FF
            or 0x1F300 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
        )

    def _format_style_profile(self) -> str:
        profile = self.personality_profile or {}
        total = profile.get('total_messages')
        lines = []
        if total:
            lines.append(f"STYLE PROFILE (from {total} messages):")
        else:
            lines.append("STYLE PROFILE:")

        language = profile.get('language_mixing', {})
        mix_score = language.get('language_mixing_score')
        if mix_score is None:
            language_desc = "Match the user's language, keep it casual."
        elif mix_score < 5:
            language_desc = "Mostly English with rare Hinglish words."
        elif mix_score < 20:
            language_desc = "Light Hinglish mix with mostly English."
        elif mix_score < 50:
            language_desc = "Balanced Hinglish mix."
        else:
            language_desc = "Mostly Hinglish."
        lines.append(f"- Language: {language_desc}")

        emoji = profile.get('emoji_patterns', {})
        emoji_pct = emoji.get('emoji_usage_percentage')
        top_emojis = [
            e for e in emoji.get('top_emojis', [])
            if any(self._is_emoji_char(ch) for ch in e)
        ]
        if emoji_pct is None:
            lines.append("- Emojis: mirror typical usage, avoid overdoing it.")
        else:
            if emoji_pct < 10:
                emoji_freq = "rare"
            elif emoji_pct < 30:
                emoji_freq = "occasional"
            elif emoji_pct < 60:
                emoji_freq = "frequent"
            else:
                emoji_freq = "very frequent"
            if top_emojis:
                lines.append(f"- Emojis: {emoji_freq}. Common: {', '.join(top_emojis[:5])}")
            else:
                lines.append(f"- Emojis: {emoji_freq}.")

        sentence = profile.get('sentence_structure', {})
        avg_words = sentence.get('avg_words_per_message')
        short_pct = sentence.get('short_messages_percentage')
        if avg_words is None and short_pct is None:
            lines.append("- Length: keep it short and chatty.")
        else:
            if (avg_words is not None and avg_words < 7) or (short_pct is not None and short_pct > 45):
                length_desc = "Prefer short replies, fragments are ok."
            elif avg_words is not None and avg_words < 12:
                length_desc = "Medium length replies, avoid long explanations."
            else:
                length_desc = "Longer replies are ok, but avoid essays."
            lines.append(f"- Length: {length_desc}")

        humor = profile.get('humor_style', {}).get('humor_style')
        if humor:
            lines.append(f"- Humor: {humor.replace('_', ' ')}")

        engagement = profile.get('response_characteristics', {})
        formality = engagement.get('formality')
        engagement_level = engagement.get('engagement_level')
        if formality or engagement_level:
            line = "- Tone:"
            if engagement_level:
                line += f" {engagement_level} engagement"
            if formality:
                line += f", {formality} formality"
            lines.append(line)

        lines.append("- Typing: casual chat spelling, small typos are ok. Do not over-correct.")

        return "\n".join(lines)

    def _format_style_examples(self, style_examples: List[str]) -> str:
        if not style_examples:
            return ""
        lines = ["STYLE EXAMPLES (mimic tone and spelling, do not copy content):"]
        for example in style_examples:
            lines.append(f"- {example}")
        return "\n".join(lines)

    def _load_local_pipeline(self):
        if AutoTokenizer is None or AutoModelForCausalLM is None or PeftModel is None or torch is None:
            raise RuntimeError("Local inference dependencies are not installed.")

        adapter_path = self.local_adapter_path
        base_model = self.local_base_model
        if not adapter_path or not base_model:
            raise RuntimeError("Local adapter path or base model is not configured.")

        key = (base_model, adapter_path)
        if key in self._local_pipeline_cache:
            return self._local_pipeline_cache[key]

        device_map = LOCAL_DEVICE_MAP
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        self._local_pipeline_cache[key] = (tokenizer, model)
        return tokenizer, model

    def _query_local(self, prompt: str) -> str:
        tokenizer, model = self._load_local_pipeline()
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                do_sample=True,
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def _detect_language_mode(self, user_message: str) -> str:
        text = user_message.lower()
        if any(0x0900 <= ord(ch) <= 0x097F for ch in user_message):
            return "hindi"
        hinglish_markers = {
            "kya", "kaise", "kaisa", "kaisi", "bhai", "yaar", "haan", "nahi",
            "kyu", "kyon", "acha", "accha", "theek", "thik", "mast", "sahi",
            "hai", "ho", "hu", "tum", "tu", "aap", "mera", "meri", "apna",
        }
        if any(word in text.split() for word in hinglish_markers):
            return "hinglish"
        preferred = self.persona_pack.get("preferred_language")
        if preferred in ("hinglish", "hindi"):
            tokens = [tok for tok in text.split() if tok]
            if len(tokens) <= 3:
                return preferred
        return "english"

    def _target_word_range(self) -> tuple:
        profile = self.personality_profile or {}
        avg_words = None
        if self.persona_pack:
            avg_words = self.persona_pack.get("avg_words_per_message")
        length_pref = None
        if self.persona_pack:
            length_pref = self.persona_pack.get("preferred_reply_length")
        if length_pref:
            length_map = {"short": 6, "medium": 12, "long": 20}
            avg_words = length_map.get(length_pref, avg_words)
        if not avg_words:
            avg_words = (
                profile.get('sentence_structure', {}).get('avg_words_per_message')
            )
        if not avg_words:
            avg_words = 10
        avg_words = max(4, min(40, int(round(avg_words))))
        min_words = max(3, int(round(avg_words * 0.6)))
        max_words = max(min_words + 2, int(round(avg_words * 1.4)))
        return min_words, max_words

    def _trim_to_max_words(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        trimmed = " ".join(words[:max_words])
        return trimmed.rstrip(" ,.;:!?") + "..."

    def _refine_response(self, user_message: str, response: str, example_results: List[Dict]) -> str:
        text = response.strip()

        # Remove meta artifacts and memory mentions
        lines = []
        for line in text.splitlines():
            lower = line.lower()
            if "reference to past conversation" in lower:
                continue
            if "past conversation" in lower:
                continue
            if "memory" in lower and ("reference" in lower or "context" in lower):
                continue
            if "as an ai" in lower or "i am an ai" in lower:
                continue
            lines.append(line)
        text = " ".join(line.strip() for line in lines if line.strip())
        if not text:
            mode = self._detect_language_mode(user_message)
            if mode == "hinglish":
                text = "haan bolo"
            elif mode == "hindi":
                text = "haan bolo"
            else:
                text = "yeah?"

        min_words, max_words = self._target_word_range()

        text = self._apply_persona_style(text, user_message)
        words = text.split()

        if len(words) > max_words:
            # Prefer trimming by sentence boundaries
            parts = []
            current = []
            for token in words:
                current.append(token)
                if token.endswith((".", "!", "?")):
                    parts.append(" ".join(current))
                    current = []
            if current:
                parts.append(" ".join(current))

            new_text = ""
            for part in parts:
                candidate = (new_text + " " + part).strip()
                if len(candidate.split()) <= max_words:
                    new_text = candidate
                else:
                    break
            text = new_text if new_text else self._trim_to_max_words(text, max_words)

        if len(text.split()) < min_words:
            mode = self._detect_language_mode(user_message)
            if mode == "hinglish":
                follow_ups = ["tu kaisa hai?", "aur kya scene?", "sab theek?"]
            elif mode == "hindi":
                follow_ups = ["tum kaise ho?", "aur kya chal raha hai?"]
            else:
                follow_ups = ["you good?", "how's it going?"]

            if not text.endswith("?"):
                text = (text + " " + follow_ups[0]).strip()

        if example_results:
            top_example = example_results[0]
            if top_example.get("similarity", 0) >= 0.75:
                if len(text.split()) > max_words:
                    text = top_example.get("output", text)
                if not text:
                    text = top_example.get("output", text)

        return text

    def _apply_persona_style(self, text: str, user_message: str) -> str:
        if not self.persona_pack:
            return text

        mode = self._detect_language_mode(user_message)
        lowered = text.lower()
        stop_tokens = {"ok", "okay", "hlo", "yep", "ho", "hn", "aahe", "na", "ha"}

        # Inject common ending if response is short and doesn't already end with it.
        endings = [
            e for e in self.persona_pack.get("common_endings", [])
            if e and "http" not in e and e not in stop_tokens
        ]
        if endings and len(text.split()) <= 8:
            candidate = endings[0]
            if candidate and not lowered.endswith(candidate):
                text = f"{text} {candidate}".strip()

        # Emoji handling
        emoji_rate = self.persona_pack.get("emoji_rate", 0)
        top_emojis = self.persona_pack.get("top_emojis", [])
        if emoji_rate and emoji_rate > 0.25 and top_emojis:
            if not any(self._is_emoji_char(ch) for ch in text):
                text = f"{text} {top_emojis[0]}".strip()

        # Ensure Hinglish flavor for hinglish mode
        if mode == "hinglish":
            slang_words = [
                s for s in self.persona_pack.get("slang_words", [])
                if s and "http" not in s and s not in stop_tokens
            ]
            if slang_words:
                marker = slang_words[0]
                if marker and marker not in lowered:
                    text = f"{marker} {text}".strip()

        return text

    def _format_persona_pack(self) -> str:
        if not self.persona_pack:
            return ""

        stop_tokens = {"ok", "okay", "hlo", "yep", "ho", "hn", "aahe", "na", "ha"}
        starters = [
            s for s in self.persona_pack.get("common_starters", [])
            if s and "http" not in s and s not in stop_tokens
        ]
        endings = [
            s for s in self.persona_pack.get("common_endings", [])
            if s and "http" not in s and s not in stop_tokens
        ]
        slang = [
            s for s in self.persona_pack.get("slang_words", [])
            if s and "http" not in s and s not in stop_tokens
        ]
        short_samples = self.persona_pack.get("short_reply_samples", [])
        emojis = self.persona_pack.get("top_emojis", [])

        lines = ["PERSONA PACK (use as style cues only):"]
        preferred_language = self.persona_pack.get("preferred_language")
        preferred_greeting = self.persona_pack.get("preferred_greeting")
        preferred_length = self.persona_pack.get("preferred_reply_length")
        tone_notes = self.persona_pack.get("tone_notes")
        avoid_topics = self.persona_pack.get("avoid_topics", [])
        emoji_rate_hint = self.persona_pack.get("emoji_rate_hint")
        if preferred_language:
            lines.append(f"- Preferred language: {preferred_language}")
        if preferred_length:
            lines.append(f"- Reply length: {preferred_length}")
        if preferred_greeting:
            lines.append(f"- Greeting style: {preferred_greeting}")
        if starters:
            lines.append(f"- Common openers: {', '.join(starters[:5])}")
        if endings:
            lines.append(f"- Common endings: {', '.join(endings[:5])}")
        if slang:
            lines.append(f"- Common slang: {', '.join(slang[:8])}")
        if emojis:
            lines.append(f"- Common emojis: {', '.join(emojis[:5])}")
        if emoji_rate_hint:
            lines.append(f"- Emoji usage hint: {emoji_rate_hint}")
        if avoid_topics:
            lines.append(f"- Avoid topics: {', '.join(avoid_topics[:8])}")
        if tone_notes:
            lines.append(f"- Style boundaries: {tone_notes}")
        if short_samples:
            lines.append("- Short reply examples:")
            for sample in short_samples[:5]:
                lines.append(f"  - {sample}")
        return "\n".join(lines)
    
    def _query_ollama(self, prompt: str) -> str:
        """Send prompt to Ollama and get response."""
        
        url = f"{self.api_endpoint}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "num_predict": LLM_MAX_TOKENS,
            "stream": False,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
        
        except requests.exceptions.Timeout:
            raise Exception("LLM request timed out (>30s)")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama API")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Ollama error: {e.response.text}")
        except json.JSONDecodeError:
            raise Exception("Invalid response from Ollama")
    
    def _apply_safety_filters(self, response: str) -> str:
        """Apply safety and ethics rules."""
        
        # Rule 1: Never claim to be real person
        if "i am the real" in response.lower() or "i'm actually" in response.lower():
            response = response.replace("I am the real", "I'm simulating")
            response = response.replace("I'm actually", "I'm simulating")
        
        # Rule 2: Don't promise permanence
        permanence_keywords = ["forever", "always be here", "never leave"]
        for keyword in permanence_keywords:
            if keyword.lower() in response.lower():
                response += "\n\n(Note: I'm an AI simulation, not a permanent presence.)"
        
        # Rule 3: Check for harmful content
        harmful_keywords = ["suicide", "self-harm", "explicit sexual"]
        for keyword in harmful_keywords:
            if keyword.lower() in response.lower():
                response = "I can't engage with that topic. Please reach out to someone you trust."
        
        return response.strip()
    
    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract thinking/reasoning from response if available."""
        # Placeholder: LLMs typically output reasoning in special tokens
        # For now, return None
        return None
    
    def get_debug_info(self) -> Dict:
        """Get debugging information."""
        return {
            'model': self.model_name,
            'api_endpoint': self.api_endpoint,
            'personality_profile_loaded': bool(self.personality_profile),
            'memory_store_loaded': bool(self.memory_store),
            'memory_store_stats': self.memory_store.get_stats() if self.memory_store else None,
        }
