import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


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
                examples.append(
                    {
                        "input": input_text,
                        "output": output_text,
                    }
                )

        self.examples = examples
        if self.embeddings_available and self.model and self.examples:
            self._generate_embeddings()

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
    ) -> List[Dict]:
        if not self.examples:
            return []

        if self.embeddings_available and self.model and self.embeddings is not None:
            query_embedding = self.model.encode([query])[0]
            denom = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            denom = np.where(denom == 0, 1e-12, denom)
            similarities = np.dot(self.embeddings, query_embedding) / denom

            sorted_indices = np.argsort(similarities)[::-1]
            results = []
            for idx in sorted_indices:
                if len(results) >= top_k:
                    break
                score = float(similarities[idx])
                if score < similarity_threshold:
                    continue
                example = self.examples[idx]
                if max_input_length and len(example["input"]) > max_input_length:
                    continue
                if max_output_length and len(example["output"]) > max_output_length:
                    continue
                results.append(
                    {
                        "input": example["input"],
                        "output": example["output"],
                        "similarity": score,
                    }
                )
            return results

        scored = []
        for ex in self.examples:
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
