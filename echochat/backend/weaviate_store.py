import json
from typing import Dict, List

import numpy as np
import requests


class WeaviateVectorStore:
    """
    Lightweight Weaviate wrapper using HTTP APIs directly.

    This avoids tight coupling to a specific weaviate-client version.
    The class stores external vectors (`vectorizer: none`) and provides
    simple upsert + nearVector search.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        index_name: str = "EchoChatMemory",
    ):
        self.url = url.rstrip("/")
        self.index_name = index_name
        self.session = requests.Session()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        response = self.session.get(f"{self.url}/v1/schema", timeout=8)
        response.raise_for_status()

        classes = [c.get("class") for c in response.json().get("classes", [])]
        if self.index_name in classes:
            return

        class_obj = {
            "class": self.index_name,
            "vectorizer": "none",
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "meta", "dataType": ["text"]},
            ],
        }
        create_response = self.session.post(
            f"{self.url}/v1/schema",
            json=class_obj,
            timeout=8,
        )
        create_response.raise_for_status()

    def _clear_class_objects(self) -> None:
        response = self.session.delete(
            f"{self.url}/v1/objects",
            params={"class": self.index_name},
            timeout=20,
        )
        # Weaviate may return 404/422 depending on state. Ignore these.
        if response.status_code not in (200, 204, 404, 422):
            response.raise_for_status()

    def upsert_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict]) -> None:
        if embeddings is None or len(metadatas) == 0:
            return

        # Rebuild object set for deterministic mapping with metadata order.
        self._clear_class_objects()

        for emb, meta in zip(embeddings, metadatas):
            payload = {
                "class": self.index_name,
                "properties": {
                    "text": meta.get("text", ""),
                    "meta": json.dumps(meta, ensure_ascii=False),
                },
                "vector": emb.tolist(),
            }
            response = self.session.post(
                f"{self.url}/v1/objects",
                json=payload,
                timeout=15,
            )
            response.raise_for_status()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        if query_vector is None:
            return []

        query = f"""
query ($vector: [Float!]!, $limit: Int!) {{
  Get {{
    {self.index_name}(nearVector: {{vector: $vector}}, limit: $limit) {{
      text
      meta
      _additional {{ certainty distance }}
    }}
  }}
}}
"""

        variables = {
            "vector": query_vector.tolist(),
            "limit": int(top_k),
        }
        response = self.session.post(
            f"{self.url}/v1/graphql",
            json={"query": query, "variables": variables},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        items = payload.get("data", {}).get("Get", {}).get(self.index_name, []) or []
        results: List[Dict] = []

        for item in items:
            meta_raw = item.get("meta")
            metadata: Dict
            try:
                metadata = json.loads(meta_raw) if meta_raw else {}
            except Exception:
                metadata = {"text": item.get("text", "")}

            additional = item.get("_additional", {}) or {}
            certainty = additional.get("certainty")
            if certainty is None:
                distance = additional.get("distance")
                # Convert distance (smaller is better) to pseudo-similarity.
                certainty = 1.0 - float(distance) if distance is not None else 0.0

            results.append(
                {
                    "metadata": metadata,
                    "score": float(certainty),
                }
            )

        return results
