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

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    PeftModel = None


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

        if not self.use_local_inference:
            self._test_connection()
    
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
            search_results = self.memory_store.search(
                user_message,
                top_k=4,
                max_length=240,
            )
            memories_context = self.memory_store.format_context(
                search_results,
                include_meta=False,
                max_chars=220,
            )
            memories_used = [result['message']['text'] for result in search_results]

        if self.example_store:
            example_results = self.example_store.search(
                user_message,
                top_k=2,
                similarity_threshold=0.22,
                max_input_length=200,
                max_output_length=200,
            )
            examples_context = self.example_store.format_examples(
                example_results,
                max_chars_input=140,
                max_chars_output=140,
            )
        
        # Build prompt
        prompt = self._build_prompt(
            user_message,
            memories_context,
            style_examples,
            examples_context,
        )
        
        if verbose:
            print("\n" + "="*60)
            print("PROMPT SENT TO LLM:")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")
        
        # Query LLM
        try:
            if self.use_local_inference:
                try:
                    response_text = self._query_local(prompt)
                except Exception as exc:
                    print(f"Local inference failed, falling back to Ollama: {exc}")
                    response_text = self._query_ollama(prompt)
            else:
                response_text = self._query_ollama(prompt)
            
            # Apply safety filters
            response_text = self._apply_safety_filters(response_text)
            response_text = self._refine_response(user_message, response_text)
            
            return {
                'response': response_text,
                'reasoning': self._extract_reasoning(response_text),
                'memories_used': memories_used,
                'model': self.model_name,
                'success': True,
            }
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                'response': f"Sorry, I encountered an error: {str(e)}",
                'reasoning': None,
                'memories_used': [],
                'model': self.model_name,
                'success': False,
            }
    
    def _build_prompt(
        self,
        user_message: str,
        memories_context: str,
        style_examples: List[str],
        examples_context: str,
    ) -> str:
        """Construct full prompt with context injection."""
        
        # Inject personality style
        style_context = STYLE_PROMPT_INJECTION
        style_profile = self._format_style_profile()
        style_examples_block = self._format_style_examples(style_examples)
        
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
            response_targets,
            examples_injection,
            memory_injection,
            "==== CURRENT CONVERSATION ====",
            f"User: {user_message}",
            "---",
            "Assistant:",
        ]

        prompt = "\n\n".join(section for section in sections if section)
        
        return prompt

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
        return "english"

    def _target_word_range(self) -> tuple:
        profile = self.personality_profile or {}
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

    def _refine_response(self, user_message: str, response: str) -> str:
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

        return text
    
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
