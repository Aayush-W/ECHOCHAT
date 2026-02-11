import json
from typing import List, Dict, Optional

import numpy as np

try:
    import weaviate
except Exception:
    weaviate = None


class WeaviateVectorStore:
    """Simple Weaviate client wrapper for storing and searching embeddings.

    This class is optional (imports may fail). It stores each message as an
    object with `text` and `meta` (JSON) properties and accepts externally
    computed vectors (vectorizer='none').
    """

    def __init__(self, url: str = "http://localhost:8080", index_name: str = "EchoChatMemory", embedding_dim: int = 384):
        if weaviate is None:
            raise ImportError("weaviate-client not installed")

        self.url = url
        self.index_name = index_name
        self.embedding_dim = embedding_dim

        self.client = weaviate.Client(url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        schema = self.client.schema.get()
        classes = [c.get('class') for c in schema.get('classes', [])]
        if self.index_name in classes:
            return

        class_obj = {
            'class': self.index_name,
            'vectorizer': 'none',
            'moduleConfig': {},
            'properties': [
                {'name': 'text', 'dataType': ['text']},
                {'name': 'meta', 'dataType': ['text']},
            ],
        }
        self.client.schema.create_class(class_obj)

    def upsert_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict]) -> None:
        """Upsert many objects with supplied vectors.

        embeddings: (N, D) numpy array
        metadatas: list of message dicts (length N)
        """
        if embeddings is None or len(metadatas) == 0:
            return

        with self.client.batch as batch:
            for emb, meta in zip(embeddings, metadatas):
                obj = {
                    'text': meta.get('text', ''),
                    'meta': json.dumps(meta, ensure_ascii=False),
                }
                batch.add_data_object(obj, self.index_name, vector=emb.tolist())

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search the Weaviate index with a vector.

        Returns list of {'metadata': <msg dict>, 'score': float}
        """
        if weaviate is None:
            return []

        if query_vector is None:
            return []

        try:
            qv = query_vector.tolist()
            res = (
                self.client.query
                import json
                from typing import List, Dict

                import numpy as np
                import requests


                class WeaviateVectorStore:
                    """Lightweight Weaviate wrapper using HTTP/GraphQL.

                    This implementation avoids depending on a specific `weaviate-client`
                    version and talks to the Weaviate HTTP API directly. It creates a class
                    (if missing) and upserts objects one-by-one. Search uses GraphQL
                    `nearVector` queries.
                    """

                    def __init__(self, url: str = "http://localhost:8080", index_name: str = "EchoChatMemory"):
                        self.url = url.rstrip("/")
                        self.index_name = index_name
                        self.session = requests.Session()
                        # Ensure the class exists (no-op if server unreachable)
                        try:
                            self._ensure_schema()
                        except Exception:
                            pass

                    def _ensure_schema(self) -> None:
                        r = self.session.get(f"{self.url}/v1/schema")
                        r.raise_for_status()
                        classes = [c.get("class") for c in r.json().get("classes", [])]
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
                        r = self.session.post(f"{self.url}/v1/schema", json=class_obj)
                        r.raise_for_status()

                    def upsert_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict]) -> None:
                        if embeddings is None or len(metadatas) == 0:
                            return

                        for emb, meta in zip(embeddings, metadatas):
                            try:
                                props = {"text": meta.get("text", ""), "meta": json.dumps(meta, ensure_ascii=False)}
                                payload = {"class": self.index_name, "properties": props, "vector": emb.tolist()}
                                r = self.session.post(f"{self.url}/v1/objects", json=payload, timeout=10)
                                # ignore individual failures; migration will log
                            except Exception:
                                continue

                    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
                        if query_vector is None:
                            return []

                        try:
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
                            variables = {"vector": query_vector.tolist(), "limit": top_k}
                            resp = self.session.post(f"{self.url}/v1/graphql", json={"query": query, "variables": variables}, timeout=10)
                            resp.raise_for_status()
                            data = resp.json()
                            results = []
                            for item in (data.get("data", {}).get("Get", {}).get(self.index_name, []) or []):
                                meta_text = item.get("meta")
                                try:
                                    metadata = json.loads(meta_text) if meta_text else {}
                                except Exception:
                                    metadata = {"text": item.get("text", "")}

                                add = item.get("_additional", {})
                                score = add.get("certainty") if add else None
                                if score is None:
                                    score = add.get("distance") if add else 0.0
                                results.append({"metadata": metadata, "score": float(score) if score is not None else 0.0})

                            return results
                        except Exception:
                            return []
