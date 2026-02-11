"""
Faiss Vector Store wrapper for persistent vector search.

Provides:
- build_index(embeddings, metadatas)
- search(query_embedding, top_k)
- save/load index and metadata

Falls back gracefully if `faiss` is not available.
"""
from pathlib import Path
import json
import numpy as np

try:
    import faiss
except Exception:
    faiss = None


class FaissVectorStore:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index_file = self.index_path.with_suffix('.index')
        self.meta_file = self.index_path.with_suffix('.meta.json')
        self.index = None
        self.metadatas = []

    def available(self) -> bool:
        return faiss is not None

    def build_index(self, embeddings: np.ndarray, metadatas: list) -> None:
        if not self.available():
            return
        if embeddings is None or len(embeddings) == 0:
            return
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        # normalize for inner product as cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        self.index = index
        self.metadatas = list(metadatas)
        # persist
        self._save()

    def _save(self):
        if not self.available() or self.index is None:
            return
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        if not self.available():
            return False
        if not self.index_file.exists() or not self.meta_file.exists():
            return False
        try:
            self.index = faiss.read_index(str(self.index_file))
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self.metadatas = json.load(f)
            return True
        except Exception:
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Return list of (metadata, score) for top_k nearest neighbors."""
        if not self.available() or self.index is None:
            return []
        if query_embedding.ndim == 1:
            q = query_embedding.reshape(1, -1).astype('float32')
        else:
            q = query_embedding.astype('float32')
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append({'metadata': self.metadatas[idx], 'score': float(score)})
        return results
