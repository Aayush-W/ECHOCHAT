import json
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .text_filter import is_blocked, is_file_related


class ExampleStore:
    """
    Retrieve example input/output pairs from training data using semantic similarity.
    """

    _shared_model = None
    _shared_model_name = None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        training_data_path: str = "data/training_data.jsonl",
    ):
        self.model_name = model_name
        self.training_data_path = training_data_path
        self.examples: List[Dict] = []
        self.embeddings = None
        self.model = None
        self.embeddings_available = SentenceTransformer is not None
        self._bm25_doc_freqs: List[Dict[str, int]] = []
        self._bm25_idf: Dict[str, float] = {}
        self._bm25_doc_len: List[int] = []
        self._bm25_avgdl: float = 0.0

        if self.embeddings_available:
            try:
                if (
                    ExampleStore._shared_model is not None
                    and ExampleStore._shared_model_name == model_name
                ):
                    self.model = ExampleStore._shared_model
                else:
                    self.model = SentenceTransformer(model_name)
                    ExampleStore._shared_model = self.model
                    ExampleStore._shared_model_name = model_name
            except Exception:
                self.embeddings_available = False

        self.load_examples()

    def load_examples(self) -> None:
        path = Path(self.training_data_path)
        if not path.exists():
            self.examples = []
            return

        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                input_text = (obj.get("input") or "").strip()
                output_text = (obj.get("output") or "").strip()
                if not input_text or not output_text:
                    continue
                if "http://" in input_text or "https://" in input_text:
                    continue
                if "http://" in output_text or "https://" in output_text:
                    continue
                if is_blocked(input_text) or is_blocked(output_text):
                    continue
                examples.append(
                    {
                        "input": input_text,
                        "output": output_text,
                    }
                )

        self.examples = examples
        self._build_bm25_index()
        if self.embeddings_available and self.model and self.examples:
            self._generate_embeddings()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _build_bm25_index(self) -> None:
        self._bm25_doc_freqs = []
        self._bm25_idf = {}
        self._bm25_doc_len = []
        self._bm25_avgdl = 0.0

        if not self.examples:
            return

        df = Counter()
        for ex in self.examples:
            tokens = self._tokenize(ex["input"])
            freqs = Counter(tokens)
            self._bm25_doc_freqs.append(freqs)
            doc_len = sum(freqs.values())
            self._bm25_doc_len.append(doc_len)
            for token in freqs.keys():
                df[token] += 1

        total_docs = len(self.examples)
        self._bm25_avgdl = sum(self._bm25_doc_len) / max(total_docs, 1)
        for token, freq in df.items():
            self._bm25_idf[token] = np.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)

    def _bm25_scores(self, query: str) -> List[float]:
        if not self._bm25_doc_freqs:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return [0.0] * len(self._bm25_doc_freqs)

        k1 = 1.5
        b = 0.75
        scores = []
        for freqs, doc_len in zip(self._bm25_doc_freqs, self._bm25_doc_len):
            score = 0.0
            for token in tokens:
                idf = self._bm25_idf.get(token, 0.0)
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + k1 * (1 - b + b * (doc_len / max(self._bm25_avgdl, 1e-6)))
                score += idf * ((tf * (k1 + 1)) / denom)
            scores.append(score)
        return scores

    def _generate_embeddings(self) -> None:
        if not self.embeddings_available or not self.model or not self.examples:
            return
        inputs = [ex["input"] for ex in self.examples]
        try:
            self.embeddings = self.model.encode(inputs, show_progress_bar=False)
        except Exception:
            self.embeddings = None

    def _lexical_similarity(self, a: str, b: str) -> float:
        a_tokens = {tok for tok in a.lower().split() if tok}
        b_tokens = {tok for tok in b.lower().split() if tok}
        if not a_tokens or not b_tokens:
            return 0.0
        intersection = a_tokens & b_tokens
        union = a_tokens | b_tokens
        return len(intersection) / max(len(union), 1)

    def search(
        self,
        query: str,
        top_k: int = 2,
        similarity_threshold: float = 0.25,
        max_input_length: Optional[int] = 200,
        max_output_length: Optional[int] = 220,
        intent: Optional[str] = None,
    ) -> List[Dict]:
        if not self.examples:
            return []

        def _passes_intent(ex: Dict) -> bool:
            if not intent:
                return True
            input_text = ex.get("input", "")
            output_text = ex.get("output", "")
            if intent == "chat":
                return (not is_file_related(input_text)) and (not is_file_related(output_text))
            return True

        bm25_scores = self._bm25_scores(query)
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0

        if self.embeddings_available and self.model and self.embeddings is not None:
            query_embedding = self.model.encode([query])[0]
            denom = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            denom = np.where(denom == 0, 1e-12, denom)
            similarities = np.dot(self.embeddings, query_embedding) / denom

            max_sim = float(np.max(similarities)) if len(similarities) else 0.0
            combined = []
            for idx, sim in enumerate(similarities):
                sim_norm = (float(sim) / max_sim) if max_sim > 0 else 0.0
                bm25_norm = (bm25_scores[idx] / max_bm25) if max_bm25 > 0 else 0.0
                score = (0.65 * sim_norm) + (0.35 * bm25_norm)
                combined.append(score)

            sorted_indices = np.argsort(combined)[::-1]
            results = []
            for idx in sorted_indices:
                if len(results) >= top_k:
                    break
                score = float(similarities[idx])
                combined_score = float(combined[idx])
                if combined_score < similarity_threshold:
                    continue
                if score < similarity_threshold:
                    if max_bm25 == 0:
                        continue
                example = self.examples[idx]
                if not _passes_intent(example):
                    continue
                if max_input_length and len(example["input"]) > max_input_length:
                    continue
                if max_output_length and len(example["output"]) > max_output_length:
                    continue
                results.append(
                    {
                        "input": example["input"],
                        "output": example["output"],
                        "similarity": combined_score,
                    }
                )
            return results

        scored = []
        for ex in self.examples:
            if not _passes_intent(ex):
                continue
            if max_input_length and len(ex["input"]) > max_input_length:
                continue
            if max_output_length and len(ex["output"]) > max_output_length:
                continue
            score = self._lexical_similarity(query, ex["input"])
            if score >= similarity_threshold:
                scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, ex in scored[:top_k]:
            results.append(
                {
                    "input": ex["input"],
                    "output": ex["output"],
                    "similarity": score,
                }
            )
        return results

    def format_examples(
        self,
        results: List[Dict],
        max_chars_input: int = 140,
        max_chars_output: int = 140,
    ) -> str:
        if not results:
            return ""

        lines = []
        for result in results:
            input_text = result.get("input", "").strip()
            output_text = result.get("output", "").strip()
            if max_chars_input and len(input_text) > max_chars_input:
                input_text = textwrap.shorten(input_text, width=max_chars_input, placeholder="...")
            if max_chars_output and len(output_text) > max_chars_output:
                output_text = textwrap.shorten(output_text, width=max_chars_output, placeholder="...")
            lines.append(f"- User: {input_text}\n  Assistant: {output_text}")

        return "\n".join(lines)
