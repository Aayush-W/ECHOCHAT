import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .text_filter import contains_banned_output, is_blocked


class ResponseRefiner:
    """
    Relevance-focused refinement agent.

    Steps:
    1) Assess relevance + drift between input and response
    2) If low relevance, consult vector DB (memory store) + examples
    3) Choose a safer, more on-topic response
    """

    GENERIC_RESPONSES = {
        "yeah?",
        "ok",
        "okay",
        "idk",
        "not sure",
        "haan bolo",
        "haan bol bhai",
        "hmm",
        "hmmm",
        "kya",
    }

    def __init__(self, memory_store=None, example_store=None):
        self.memory_store = memory_store
        self.example_store = example_store

    def _get_embedder(self):
        if self.memory_store and getattr(self.memory_store, "model", None):
            return self.memory_store.model
        if self.example_store and getattr(self.example_store, "model", None):
            return self.example_store.model
        return None

    def _lexical_similarity(self, a: str, b: str) -> float:
        a_tokens = {tok for tok in re.findall(r"\b\w+\b", (a or "").lower()) if tok}
        b_tokens = {tok for tok in re.findall(r"\b\w+\b", (b or "").lower()) if tok}
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)

    def _semantic_similarity(self, a: str, b: str) -> float:
        embedder = self._get_embedder()
        if embedder is None:
            return self._lexical_similarity(a, b)
        try:
            vecs = embedder.encode([a, b])
            v1, v2 = vecs[0], vecs[1]
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1e-12
            return float(np.dot(v1, v2) / denom)
        except Exception:
            return self._lexical_similarity(a, b)

    def assess(self, user_message: str, response: str) -> Dict:
        relevance = self._semantic_similarity(user_message, response)
        drift = max(0.0, 1.0 - relevance)
        lower = (response or "").strip().lower()
        generic = lower in self.GENERIC_RESPONSES
        too_short = len((response or "").split()) < 2
        blocked = is_blocked(response) or contains_banned_output(response)
        return {
            "relevance": relevance,
            "drift": drift,
            "generic": generic,
            "too_short": too_short,
            "blocked": blocked,
        }

    def _fallback_prompt(self, language_mode: str) -> str:
        if language_mode == "hinglish":
            return "thoda aur detail de?"
        if language_mode == "hindi":
            return "thoda aur detail do?"
        return "can you share a bit more detail?"

    def _select_example(self, example_results: List[Dict]) -> Optional[str]:
        if not example_results:
            return None
        best = example_results[0]
        similarity = best.get("similarity", 0.0)
        if similarity >= 0.4:
            output = (best.get("output") or "").strip()
            if output and not contains_banned_output(output):
                return output
        return None

    def _select_memory(self, query: str, intent: str) -> Optional[str]:
        if not self.memory_store:
            return None
        try:
            results = self.memory_store.search(
                query,
                top_k=3,
                similarity_threshold=0.35,
                max_length=240,
                intent=intent,
            )
        except Exception:
            results = []
        if not results:
            return None
        for result in results:
            msg = result.get("message", {})
            text = (msg.get("text") or "").strip()
            if text and not contains_banned_output(text):
                return text
        return None

    def refine(
        self,
        user_message: str,
        response: str,
        intent: str,
        example_results: Optional[List[Dict]] = None,
        language_mode: str = "english",
    ) -> Tuple[str, Dict]:
        example_results = example_results or []
        metrics = self.assess(user_message, response)

        relevance = metrics["relevance"]
        drift = metrics["drift"]
        generic = metrics["generic"] or metrics["too_short"]
        blocked = metrics["blocked"]

        if intent == "info":
            needs_refine = blocked or generic or relevance < 0.18 or drift > 0.82
        else:
            needs_refine = blocked or (generic and relevance < 0.12) or drift > 0.9

        if not needs_refine:
            return response, metrics

        candidate = self._select_example(example_results)
        chosen = "example"
        if not candidate:
            candidate = self._select_memory(user_message, intent)
            chosen = "memory"

        if not candidate:
            candidate = self._fallback_prompt(language_mode)
            chosen = "fallback"

        metrics["refined"] = True
        metrics["chosen"] = chosen
        return candidate, metrics
