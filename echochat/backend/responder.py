from typing import Dict, List, Optional
import requests
import json

from backend.config import (
    LLM_API_ENDPOINT,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    SYSTEM_PROMPT,
    STYLE_PROMPT_INJECTION,
    MEMORY_PROMPT_INJECTION,
)


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
        model_name: str = LLM_MODEL_NAME,
        api_endpoint: str = LLM_API_ENDPOINT,
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
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if Ollama API is running."""
        try:
            response = requests.get(f"{self.api_endpoint}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {self.api_endpoint}")
                return True
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {self.api_endpoint}")
            print("   Start Ollama with: ollama serve")
            print(f"   Or pull model: ollama pull {self.model_name}")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
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
        
        if include_memories and self.memory_store:
            search_results = self.memory_store.search(user_message, top_k=5)
            memories_context = self.memory_store.format_context(search_results)
            memories_used = [result['message']['text'] for result in search_results]
        
        # Build prompt
        prompt = self._build_prompt(
            user_message,
            memories_context,
        )
        
        if verbose:
            print("\n" + "="*60)
            print("PROMPT SENT TO LLM:")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")
        
        # Query LLM
        try:
            response_text = self._query_ollama(prompt)
            
            # Apply safety filters
            response_text = self._apply_safety_filters(response_text)
            
            return {
                'response': response_text,
                'reasoning': self._extract_reasoning(response_text),
                'memories_used': memories_used,
                'model': self.model_name,
                'success': True,
            }
        
        except Exception as e:
            print(f"❌ Error generating response: {e}")
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
    ) -> str:
        """Construct full prompt with context injection."""
        
        # Inject personality style
        style_context = STYLE_PROMPT_INJECTION
        
        # Inject memory context
        memory_injection = MEMORY_PROMPT_INJECTION.format(
            memories_context=memories_context if memories_context else "No relevant past messages."
        )
        
        # Build full prompt
        prompt = f"""{SYSTEM_PROMPT}

{style_context}

{memory_injection}

==== CURRENT CONVERSATION ====

User: {user_message}

{self.personality_profile.get('total_messages', '')}

---

{self.model_name.upper()} (respond naturally):"""
        
        return prompt
    
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