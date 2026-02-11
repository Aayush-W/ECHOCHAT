import json
import re
import textwrap
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from .weaviate_store import WeaviateVectorStore
except Exception:
    WeaviateVectorStore = None

try:
    from .vector_store import FaissVectorStore
except Exception:
    FaissVectorStore = None

from .text_filter import is_blocked, is_file_related, is_chat_safe


class MemoryStore:
    """
    Vector-based semantic memory system.

    Stores and retrieves past messages using semantic similarity.

    Capabilities:
    - Load past messages
    - Generate embeddings for each message
    - Semantic search (find similar messages)
    - Context retrieval for prompt injection
    - Memory decay/aging (optional)
    """

    _shared_model = None
    _shared_model_name = None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        memory_data_path: str = "data/memory_data.json",
    ):
        """
        Args:
            model_name: HuggingFace embedding model
            memory_data_path: Path to memory data JSON
        """
        self.model_name = model_name
        self.memory_data_path = memory_data_path
        self.messages: List[Dict] = []
        self.embeddings = None
        self.model = None
        self.embeddings_available = SentenceTransformer is not None
        self._sorted_indices: List[int] = []
        self._index_to_pos: Dict[int, int] = {}
        self._text_to_indices: Dict[str, List[int]] = {}
        self._normalized_texts: List[str] = []
        self._text_ts_to_index: Dict[tuple, int] = {}

        if self.embeddings_available:
            try:
                if (
                    MemoryStore._shared_model is not None
                    and MemoryStore._shared_model_name == model_name
                ):
                    self.model = MemoryStore._shared_model
                else:
                    self.model = SentenceTransformer(model_name)
                    MemoryStore._shared_model = self.model
                    MemoryStore._shared_model_name = model_name
                print(f"Loaded embedding model: {model_name}")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                self.embeddings_available = False
        else:
            print("sentence-transformers not installed. Embeddings disabled.")

        self.load_memories()
        # initialize persistent vector store if available
        try:
            self._vector_store = None
            # prefer Weaviate when available (managed vector DB)
            if WeaviateVectorStore is not None:
                try:
                    self._vector_store = WeaviateVectorStore(index_name=Path(self.memory_data_path).stem)
                except Exception:
                    self._vector_store = None
            elif FaissVectorStore is not None:
                emb_index_path = Path(self.memory_data_path).with_suffix('.faiss')
                self._vector_store = FaissVectorStore(str(emb_index_path))
                # Try loading existing index (Faiss has a load method)
                try:
                    if not self._vector_store.load():
                        pass
                except Exception:
                    pass
        except Exception:
            self._vector_store = None

    def load_memories(self) -> None:
        """Load memory data from JSON file."""
        try:
            with open(self.memory_data_path, 'r', encoding='utf-8') as f:
                self.messages = json.load(f)
            print(f"Loaded {len(self.messages)} memories")
            self._build_indices()

            # Attempt to load cached embeddings from disk for this memory file
            try:
                emb_path = Path(self.memory_data_path).with_suffix('.emb.npy')
                meta_path = Path(self.memory_data_path).with_suffix('.emb.meta.json')
                if emb_path.exists() and self.embeddings_available and self.model:
                    try:
                        arr = np.load(str(emb_path))
                        self.embeddings = arr
                        # verify dimension
                        if getattr(self.embeddings, 'ndim', 0) == 1:
                            self.embeddings = None
                        else:
                            print(f"Loaded embeddings cache: {emb_path}")
                    except Exception:
                        self.embeddings = None
                # If not loaded, generate and persist
                if self.embeddings_available and self.model and self.embeddings is None:
                    self._generate_embeddings()
                    try:
                        if self.embeddings is not None:
                            np.save(str(emb_path), self.embeddings)
                            meta = {"model_name": self.model_name}
                            with open(meta_path, 'w', encoding='utf-8') as mf:
                                json.dump(meta, mf)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Embedding cache handling error: {e}")
        except FileNotFoundError:
            print(f"Memory file not found: {self.memory_data_path}")
            self.messages = []
            self._build_indices()
        except json.JSONDecodeError as e:
            print(f"Memory file is not valid JSON: {e}")
            self.messages = []
            self._build_indices()

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _build_indices(self) -> None:
        self._normalized_texts = [self._normalize_text(msg.get("text", "")) for msg in self.messages]
        self._sorted_indices = sorted(
            range(len(self.messages)),
            key=lambda i: self.messages[i].get("timestamp", ""),
        )
        self._index_to_pos = {idx: pos for pos, idx in enumerate(self._sorted_indices)}
        self._text_to_indices = {}
        self._text_ts_to_index = {}
        for idx, norm in enumerate(self._normalized_texts):
            if not norm:
                continue
            self._text_to_indices.setdefault(norm, []).append(idx)
            ts = self.messages[idx].get("timestamp", "")
            self._text_ts_to_index[(ts, norm)] = idx
        for norm, idxs in self._text_to_indices.items():
            idxs.sort(key=lambda i: self.messages[i].get("timestamp", ""))

    def _index_for_message(self, msg: Dict) -> Optional[int]:
        if not isinstance(msg, dict):
            return None
        norm = self._normalize_text(msg.get("text", ""))
        ts = msg.get("timestamp", "")
        idx = self._text_ts_to_index.get((ts, norm))
        if idx is not None:
            return idx
        if not norm:
            return None
        indices = self._text_to_indices.get(norm)
        if not indices:
            return None
        return indices[0]

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all messages."""
        if not self.embeddings_available or not self.model:
            print("Embeddings not available")
            return

        if not self.messages:
            self.embeddings = None
            return

        texts = [msg.get('text', '') for msg in self.messages]
        print(f"Generating embeddings for {len(texts)} messages...")

        try:
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"Generated {len(self.embeddings)} embeddings")
            # Persist embeddings to disk alongside memory file
            try:
                emb_path = Path(self.memory_data_path).with_suffix('.emb.npy')
                np.save(str(emb_path), self.embeddings)
                # Push embeddings to vector store if available
                try:
                    vs = getattr(self, '_vector_store', None)
                    if vs is not None:
                        if hasattr(vs, 'upsert_embeddings'):
                            vs.upsert_embeddings(self.embeddings, self.messages)
                        elif hasattr(vs, 'build_index'):
                            vs.build_index(self.embeddings, self.messages)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception as e:
            print(f"Error generating embeddings: {e}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        max_length: Optional[int] = None,
        intent: Optional[str] = None,
    ) -> List[Dict]:
        """
        Semantic search for similar messages.

        Args:
            query: User input or context
            top_k: Number of top results
            similarity_threshold: Minimum cosine similarity

        Returns:
            List of most relevant memories
        """
        if not self.embeddings_available or not self.model or self.embeddings is None:
            print("Embeddings unavailable. Returning recent messages.")
            return self.get_recent_messages(top_k, max_length=max_length, intent=intent)

        def _passes_intent(text: str) -> bool:
            if not intent:
                return True
            if intent == "chat":
                return is_chat_safe(text)
            if intent == "file":
                return not is_blocked(text)
            return not is_blocked(text)

        try:
            # Prefer vector store search (Weaviate preferred, Faiss fallback)
            if self._vector_store:
                qe = self.model.encode([query])[0]
                try:
                    results = self._vector_store.search(qe, top_k=top_k)
                except Exception:
                    results = []

                out = []
                for r in results:
                    msg = r.get('metadata')
                    score = r.get('score', 0.0)
                    text = msg.get('text', '') if isinstance(msg, dict) else ''
                    if not _passes_intent(text):
                        continue
                    msg_length = msg.get('length', len(text)) if isinstance(msg, dict) else len(text)
                    if max_length is not None and msg_length > max_length:
                        continue
                    if score is not None and score < similarity_threshold:
                        continue
                    out.append({
                        'message': msg,
                        'similarity': float(score),
                        'index': self._index_for_message(msg),
                    })
                if out:
                    return out

            # Fallback: numpy-based cosine similarity
            query_embedding = self.model.encode([query])[0]

            denom = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            denom = np.where(denom == 0, 1e-12, denom)
            similarities = np.dot(self.embeddings, query_embedding) / denom

            sorted_indices = np.argsort(similarities)[::-1]

            results = []
            for idx in sorted_indices:
                if len(results) >= top_k:
                    break
                if similarities[idx] < similarity_threshold:
                    continue

                msg = self.messages[idx]
                text = msg.get('text', '')
                if not _passes_intent(text):
                    continue
                msg_length = msg.get('length', len(text))
                if max_length is not None and msg_length > max_length:
                    continue

                results.append({
                    'message': msg,
                    'similarity': float(similarities[idx]),
                    'index': idx,
                })

            if results:
                return results
            return self.get_recent_messages(top_k, max_length=max_length, intent=intent)

        except Exception as e:
            print(f"Error during search: {e}")
            return self.get_recent_messages(top_k, max_length=max_length, intent=intent)

    def get_recent_messages(
        self,
        count: int = 5,
        max_length: Optional[int] = None,
        intent: Optional[str] = None,
    ) -> List[Dict]:
        """Get most recent messages (fallback)."""
        if not self.messages:
            return []

        def _passes_intent(text: str) -> bool:
            if not intent:
                return True
            if intent == "chat":
                return is_chat_safe(text)
            if intent == "file":
                return not is_blocked(text)
            return not is_blocked(text)

        recent_indices = sorted(
            range(len(self.messages)),
            key=lambda i: self.messages[i].get('timestamp', ''),
            reverse=True,
        )
        recent = [(idx, self.messages[idx]) for idx in recent_indices]

        if max_length is not None:
            filtered = []
            for idx, msg in recent:
                text = msg.get('text', '')
                if not _passes_intent(text):
                    continue
                msg_length = msg.get('length', len(text))
                if msg_length <= max_length:
                    filtered.append((idx, msg))
            recent = filtered
        else:
            recent = [(idx, msg) for idx, msg in recent if _passes_intent(msg.get('text', ''))]

        recent = recent[:count]

        return [{'message': msg, 'similarity': 0.0, 'index': idx} for idx, msg in recent]

    def search_with_context(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.3,
        max_length: Optional[int] = None,
        intent: Optional[str] = None,
        neighbor_window: int = 2,
        max_duplicates: int = 2,
        max_total: int = 8,
    ) -> List[Dict]:
        base_results = self.search(
            query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            max_length=max_length,
            intent=intent,
        )
        if not base_results:
            return []

        def _passes_intent(text: str) -> bool:
            if not intent:
                return True
            if intent == "chat":
                return is_chat_safe(text)
            if intent == "file":
                return not is_blocked(text)
            return not is_blocked(text)

        base_indices = [res.get("index") for res in base_results if res.get("index") is not None]
        selected = set(base_indices)

        for idx in list(base_indices):
            pos = self._index_to_pos.get(idx)
            if pos is None:
                continue
            for offset in range(1, neighbor_window + 1):
                if pos - offset >= 0:
                    selected.add(self._sorted_indices[pos - offset])
                if pos + offset < len(self._sorted_indices):
                    selected.add(self._sorted_indices[pos + offset])

            norm = self._normalized_texts[idx] if idx < len(self._normalized_texts) else ""
            if norm and norm in self._text_to_indices:
                dup_indices = self._text_to_indices[norm]
                if len(dup_indices) > 1:
                    try:
                        dup_pos = dup_indices.index(idx)
                    except ValueError:
                        dup_pos = None
                    if dup_pos is not None:
                        for offset in range(1, max_duplicates + 1):
                            if dup_pos - offset >= 0:
                                selected.add(dup_indices[dup_pos - offset])
                            if dup_pos + offset < len(dup_indices):
                                selected.add(dup_indices[dup_pos + offset])

        similarity_map = {res.get("index"): res.get("similarity", 0.0) for res in base_results}
        ordered = sorted(
            selected,
            key=lambda i: self.messages[i].get("timestamp", ""),
        )

        results: List[Dict] = []
        for idx in ordered:
            msg = self.messages[idx]
            text = msg.get("text", "")
            if not text:
                continue
            if max_length is not None:
                msg_length = msg.get("length", len(text))
                if msg_length > max_length:
                    continue
            if not _passes_intent(text):
                continue
            results.append({
                'message': msg,
                'similarity': float(similarity_map.get(idx, 0.0)),
                'index': idx,
            })
            if len(results) >= max_total:
                break
        return results

    def format_context(
        self,
        search_results: List[Dict],
        include_meta: bool = False,
        max_chars: int = 220,
    ) -> str:
        """Format search results as context string for prompt."""
        if not search_results:
            return ""

        context_lines = []
        for result in search_results:
            msg = result['message']
            text = msg.get('text', '').strip()
            if not text:
                continue

            if max_chars is not None and len(text) > max_chars:
                text = textwrap.shorten(text, width=max_chars, placeholder="...")

            if include_meta:
                timestamp = msg.get('timestamp', 'unknown')
                context_lines.append(f"- ({timestamp}) {text}")
            else:
                context_lines.append(f"- {text}")

        return '\n'.join(context_lines)

    def sample_style_examples(self, count: int = 4, max_length: int = 120) -> List[str]:
        """Pick short, style-rich messages to use as tone examples."""
        if not self.messages:
            return []

        candidates = []
        for msg in self.messages:
            text = msg.get('text', '').strip()
            if not text:
                continue
            if is_blocked(text) or is_file_related(text):
                continue
            if len(text) < 3 or len(text) > max_length:
                continue
            if "\n" in text:
                continue
            if "http://" in text or "https://" in text:
                continue
            if re.search(r"\b\d{7,}\b", text):
                continue

            score = 0
            if msg.get('has_emoji'):
                score += 2
            if re.search(r"\b(ya|haan|nahi|kya|bhai|bro|lol|hehe|xd|hmm|ok|okay)\b", text.lower()):
                score += 1
            if "?" in text or "!" in text:
                score += 1
            if len(text.split()) <= 6:
                score += 1

            candidates.append((score, len(text), text))

        if not candidates:
            return []

        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        examples = []
        for _, _, text in candidates:
            if text not in examples:
                examples.append(text)
            if len(examples) >= count:
                break

        return examples

    def add_memory(self, message: Dict) -> None:
        """Add a new message to memory."""
        self.messages.append(message)

        if self.embeddings_available and self.model:
            self._generate_embeddings()

    def save_memories(self, path: str = None) -> None:
        """Save updated memories to JSON."""
        if path is None:
            path = self.memory_data_path

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(self.messages)} memories to {path}")
        except IOError as e:
            print(f"Error saving memories: {e}")

    def get_stats(self) -> Dict:
        """Get memory store statistics."""
        return {
            'total_memories': len(self.messages),
            'embedding_model': self.model_name,
            'embeddings_available': self.embeddings_available and self.model is not None,
            'embeddings_generated': self.embeddings is not None,
        }
