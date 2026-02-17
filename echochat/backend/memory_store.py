import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Set

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

from .text_filter import is_blocked, is_chat_safe, is_file_related


class MemoryStore:
    """
    Vector-based semantic memory with intent-aware hybrid reranking.

    Retrieval combines:
    - semantic similarity (sentence-transformer embeddings)
    - lexical overlap (keyword coverage)

    This reduces false positives from pure vector-nearest matches while keeping
    recall for context-heavy prompts.
    """

    _shared_model = None
    _shared_model_name = None
    _LEXICAL_STOPWORDS = {
        "a", "an", "the", "is", "am", "are", "was", "were", "to", "of", "for", "in",
        "on", "at", "and", "or", "it", "this", "that", "i", "me", "my", "you", "your",
        "we", "our", "they", "their", "do", "does", "did", "have", "has", "had", "be",
        "kya", "kaise", "kaisa", "kaisi", "hai", "ho", "hu", "na", "haan", "nahi",
        "aur", "tu", "tum", "main", "mai", "bhai", "bro", "yaar", "kasa", "kay",
        "ki", "ka", "ke", "ch", "cha",
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        memory_data_path: str = "data/memory_data.json",
    ):
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
        self._vector_store = None
        self._vector_signature: Optional[tuple] = None

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

        self._init_vector_store()
        self.load_memories()

    def _init_vector_store(self) -> None:
        try:
            # Prefer managed DB when available, fallback to local FAISS.
            if WeaviateVectorStore is not None:
                try:
                    self._vector_store = WeaviateVectorStore(
                        index_name=Path(self.memory_data_path).stem
                    )
                    return
                except Exception:
                    self._vector_store = None

            if FaissVectorStore is not None:
                emb_index_path = Path(self.memory_data_path).with_suffix(".faiss")
                self._vector_store = FaissVectorStore(str(emb_index_path))
                try:
                    self._vector_store.load()
                except Exception:
                    pass
        except Exception:
            self._vector_store = None

    def load_memories(self) -> None:
        try:
            with open(self.memory_data_path, "r", encoding="utf-8") as f:
                self.messages = json.load(f)
            print(f"Loaded {len(self.messages)} memories")
            self._build_indices()

            emb_path = Path(self.memory_data_path).with_suffix(".emb.npy")
            meta_path = Path(self.memory_data_path).with_suffix(".emb.meta.json")

            if emb_path.exists() and self.embeddings_available and self.model:
                try:
                    arr = np.load(str(emb_path))
                    if getattr(arr, "ndim", 0) >= 2 and len(arr) == len(self.messages):
                        self.embeddings = arr
                        print(f"Loaded embeddings cache: {emb_path}")
                    else:
                        self.embeddings = None
                except Exception:
                    self.embeddings = None

            if self.embeddings_available and self.model and self.embeddings is None:
                self._generate_embeddings()
                try:
                    if self.embeddings is not None:
                        np.save(str(emb_path), self.embeddings)
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            json.dump({"model_name": self.model_name}, mf)
                except Exception:
                    pass

            self._sync_vector_store()
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

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", (text or "").lower())

    def _query_terms(self, query: str) -> Set[str]:
        return self._content_terms(query)

    def _content_terms(self, text: str) -> Set[str]:
        tokens = self._tokenize(text)
        return {
            tok for tok in tokens
            if tok and len(tok) > 1 and tok not in self._LEXICAL_STOPWORDS
        }

    def _lexical_score(self, query_terms: Set[str], text: str, query_lower: str) -> float:
        if not query_terms:
            return 0.0

        text_tokens = set(self._tokenize(text))
        if not text_tokens:
            return 0.0

        intersection = query_terms & text_tokens
        if not intersection:
            return 0.0

        coverage = len(intersection) / max(len(query_terms), 1)
        jaccard = len(intersection) / max(len(query_terms | text_tokens), 1)
        score = (0.75 * coverage) + (0.25 * jaccard)

        if query_lower and query_lower in (text or "").lower():
            score = max(score, 0.95)

        return float(score)

    def _passes_intent(self, text: str, intent: Optional[str]) -> bool:
        if not intent:
            return True
        if intent == "chat":
            return is_chat_safe(text)
        if intent == "file":
            return not is_blocked(text)
        return not is_blocked(text)

    def _passes_relevance_gate(
        self,
        semantic_score: float,
        lexical_score: float,
        similarity_threshold: float,
        intent: Optional[str],
        query_term_count: int,
    ) -> bool:
        if intent == "info":
            min_sem = max(similarity_threshold, 0.36)
            min_lex = 0.06 if query_term_count >= 2 else 0.0
        elif intent == "file":
            min_sem = max(similarity_threshold, 0.34)
            min_lex = 0.04 if query_term_count >= 2 else 0.0
        else:
            min_sem = max(similarity_threshold, 0.30)
            min_lex = 0.0

        if semantic_score < min_sem:
            return False
        if lexical_score < min_lex:
            return False
        return True

    def _hybrid_score(self, semantic_score: float, lexical_score: float) -> float:
        return float((0.8 * semantic_score) + (0.2 * lexical_score))

    def _lexical_only_search(
        self,
        query: str,
        top_k: int,
        max_length: Optional[int],
        intent: Optional[str],
    ) -> List[Dict]:
        query_lower = query.strip().lower()
        query_terms = self._query_terms(query)
        if not query_terms:
            return []

        lexical_hits: List[Dict] = []
        for idx, msg in enumerate(self.messages):
            text = msg.get("text", "")
            if not text:
                continue
            if not self._passes_intent(text, intent):
                continue
            msg_length = msg.get("length", len(text))
            if max_length is not None and msg_length > max_length:
                continue

            lexical_score = self._lexical_score(query_terms, text, query_lower)
            if intent == "info":
                if lexical_score < 0.18:
                    continue
            elif lexical_score < 0.12:
                continue

            lexical_hits.append(
                {
                    "message": msg,
                    "similarity": lexical_score,
                    "lexical_score": lexical_score,
                    "hybrid_score": lexical_score,
                    "index": idx,
                }
            )

        lexical_hits.sort(
            key=lambda x: (x.get("hybrid_score", 0.0), x.get("similarity", 0.0)),
            reverse=True,
        )
        return lexical_hits[:top_k]

    def _build_indices(self) -> None:
        self._normalized_texts = [
            self._normalize_text(msg.get("text", ""))
            for msg in self.messages
        ]
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
        if not self.embeddings_available or not self.model:
            print("Embeddings not available")
            return

        if not self.messages:
            self.embeddings = None
            return

        texts = [msg.get("text", "") for msg in self.messages]
        print(f"Generating embeddings for {len(texts)} messages...")
        try:
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"Generated {len(self.embeddings)} embeddings")
            self._sync_vector_store()
        except Exception as e:
            print(f"Error generating embeddings: {e}")

    def _sync_vector_store(self) -> None:
        vs = getattr(self, "_vector_store", None)
        if vs is None or self.embeddings is None or not self.messages:
            return

        signature = (
            len(self.messages),
            self.model_name,
            self.messages[0].get("timestamp", "") if self.messages else "",
            self.messages[-1].get("timestamp", "") if self.messages else "",
        )
        if signature == self._vector_signature:
            return

        try:
            if hasattr(vs, "build_index"):
                # Faiss path
                needs_rebuild = True
                try:
                    meta_len = len(getattr(vs, "metadatas", []) or [])
                    needs_rebuild = meta_len != len(self.messages)
                except Exception:
                    needs_rebuild = True
                if needs_rebuild:
                    vs.build_index(self.embeddings.astype("float32"), self.messages)
            elif hasattr(vs, "upsert_embeddings"):
                # Weaviate path
                vs.upsert_embeddings(self.embeddings.astype("float32"), self.messages)
            self._vector_signature = signature
        except Exception:
            pass

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        max_length: Optional[int] = None,
        intent: Optional[str] = None,
        allow_recent_fallback: bool = False,
    ) -> List[Dict]:
        """
        Hybrid semantic + lexical search.
        """
        if not query or not query.strip():
            return []

        if not self.embeddings_available or not self.model or self.embeddings is None:
            hits = self._lexical_only_search(
                query=query,
                top_k=top_k,
                max_length=max_length,
                intent=intent,
            )
            if hits:
                return hits
            if allow_recent_fallback:
                return self.get_recent_messages(top_k, max_length=max_length, intent=intent)
            return []

        try:
            query_embedding = self.model.encode([query])[0]
            query_lower = query.strip().lower()
            query_terms = self._query_terms(query)
            query_term_count = len(query_terms)

            # Optional vector DB shortlist.
            vector_scores: Dict[int, float] = {}
            if self._vector_store is not None:
                try:
                    short_k = min(max(top_k * 8, 32), len(self.messages))
                    vector_hits = self._vector_store.search(query_embedding, top_k=short_k)
                    for hit in vector_hits:
                        msg = hit.get("metadata")
                        if not isinstance(msg, dict):
                            continue
                        idx = self._index_for_message(msg)
                        if idx is None:
                            continue
                        score = float(hit.get("score", 0.0) or 0.0)
                        prev = vector_scores.get(idx)
                        if prev is None or score > prev:
                            vector_scores[idx] = score
                except Exception:
                    vector_scores = {}

            denom = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            denom = np.where(denom == 0, 1e-12, denom)
            similarities = np.dot(self.embeddings, query_embedding) / denom

            shortlist_size = min(max(top_k * 12, 64), len(self.messages))
            semantic_shortlist = np.argsort(similarities)[::-1][:shortlist_size]
            candidate_indices = set(int(i) for i in semantic_shortlist)
            candidate_indices.update(vector_scores.keys())

            ranked: List[Dict] = []
            for idx in candidate_indices:
                msg = self.messages[idx]
                text = msg.get("text", "")
                if not text:
                    continue
                if not self._passes_intent(text, intent):
                    continue

                msg_length = msg.get("length", len(text))
                if max_length is not None and msg_length > max_length:
                    continue

                semantic_score = float(similarities[idx])
                if idx in vector_scores:
                    semantic_score = max(semantic_score, vector_scores[idx])

                lexical_score = self._lexical_score(query_terms, text, query_lower)
                if not self._passes_relevance_gate(
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                    similarity_threshold=similarity_threshold,
                    intent=intent,
                    query_term_count=query_term_count,
                ):
                    continue

                ranked.append(
                    {
                        "message": msg,
                        "similarity": semantic_score,
                        "lexical_score": lexical_score,
                        "hybrid_score": self._hybrid_score(semantic_score, lexical_score),
                        "index": idx,
                    }
                )

            if ranked:
                ranked.sort(
                    key=lambda x: (
                        x.get("hybrid_score", 0.0),
                        x.get("similarity", 0.0),
                        x.get("lexical_score", 0.0),
                    ),
                    reverse=True,
                )
                return ranked[:top_k]

            if allow_recent_fallback:
                return self.get_recent_messages(top_k, max_length=max_length, intent=intent)
            return []
        except Exception as e:
            print(f"Error during search: {e}")
            if allow_recent_fallback:
                return self.get_recent_messages(top_k, max_length=max_length, intent=intent)
            return []

    def get_recent_messages(
        self,
        count: int = 5,
        max_length: Optional[int] = None,
        intent: Optional[str] = None,
    ) -> List[Dict]:
        if not self.messages:
            return []

        recent_indices = sorted(
            range(len(self.messages)),
            key=lambda i: self.messages[i].get("timestamp", ""),
            reverse=True,
        )
        recent = [(idx, self.messages[idx]) for idx in recent_indices]

        if max_length is not None:
            filtered = []
            for idx, msg in recent:
                text = msg.get("text", "")
                if not self._passes_intent(text, intent):
                    continue
                msg_length = msg.get("length", len(text))
                if msg_length <= max_length:
                    filtered.append((idx, msg))
            recent = filtered
        else:
            recent = [
                (idx, msg)
                for idx, msg in recent
                if self._passes_intent(msg.get("text", ""), intent)
            ]

        recent = recent[:count]
        return [{"message": msg, "similarity": 0.0, "index": idx} for idx, msg in recent]

    def search_with_context(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.3,
        max_length: Optional[int] = None,
        intent: Optional[str] = None,
        neighbor_window: int = 1,
        max_duplicates: int = 1,
        max_total: int = 6,
    ) -> List[Dict]:
        base_results = self.search(
            query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            max_length=max_length,
            intent=intent,
            allow_recent_fallback=False,
        )
        if not base_results:
            return []

        query_terms = self._query_terms(query)
        query_lower = query.strip().lower()
        selected = {
            res.get("index")
            for res in base_results
            if res.get("index") is not None
        }
        selected = set(list(selected)[:top_k])

        relevance_map = {
            res.get("index"): float(res.get("hybrid_score", res.get("similarity", 0.0)))
            for res in base_results
            if res.get("index") is not None
        }

        for idx in list(selected):
            pos = self._index_to_pos.get(idx)
            if pos is None:
                continue

            anchor_text = self.messages[idx].get("text", "")
            anchor_terms = self._content_terms(anchor_text)
            min_neighbor_lex = 0.08 if intent == "info" else 0.05
            min_neighbor_bridge = 0.3 if intent == "info" else 0.2
            for offset in range(1, neighbor_window + 1):
                if pos - offset >= 0:
                    prev_idx = self._sorted_indices[pos - offset]
                    prev_text = self.messages[prev_idx].get("text", "")
                    prev_lex = self._lexical_score(query_terms, prev_text, query_lower)
                    prev_bridge = (
                        len(anchor_terms & self._content_terms(prev_text)) / max(len(anchor_terms), 1)
                        if anchor_terms else 0.0
                    )
                    if prev_lex >= min_neighbor_lex or prev_bridge >= min_neighbor_bridge:
                        selected.add(prev_idx)

                if pos + offset < len(self._sorted_indices):
                    next_idx = self._sorted_indices[pos + offset]
                    next_text = self.messages[next_idx].get("text", "")
                    next_lex = self._lexical_score(query_terms, next_text, query_lower)
                    next_bridge = (
                        len(anchor_terms & self._content_terms(next_text)) / max(len(anchor_terms), 1)
                        if anchor_terms else 0.0
                    )
                    if next_lex >= min_neighbor_lex or next_bridge >= min_neighbor_bridge:
                        selected.add(next_idx)

            # Duplicate neighbors are useful only when base relevance is already strong.
            norm = self._normalized_texts[idx] if idx < len(self._normalized_texts) else ""
            if (
                max_duplicates > 0
                and norm
                and norm in self._text_to_indices
                and relevance_map.get(idx, 0.0) >= 0.5
            ):
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
            if not self._passes_intent(text, intent):
                continue

            results.append(
                {
                    "message": msg,
                    "similarity": float(relevance_map.get(idx, 0.0)),
                    "index": idx,
                }
            )
            if len(results) >= max_total:
                break

        return results

    def format_context(
        self,
        search_results: List[Dict],
        include_meta: bool = False,
        max_chars: int = 220,
    ) -> str:
        if not search_results:
            return ""

        context_lines = []
        for result in search_results:
            msg = result["message"]
            text = msg.get("text", "").strip()
            if not text:
                continue

            if max_chars is not None and len(text) > max_chars:
                text = textwrap.shorten(text, width=max_chars, placeholder="...")

            if include_meta:
                timestamp = msg.get("timestamp", "unknown")
                context_lines.append(f"- ({timestamp}) {text}")
            else:
                context_lines.append(f"- {text}")

        return "\n".join(context_lines)

    def sample_style_examples(self, count: int = 4, max_length: int = 120) -> List[str]:
        if not self.messages:
            return []

        candidates = []
        for msg in self.messages:
            text = msg.get("text", "").strip()
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
            if msg.get("has_emoji"):
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
        self.messages.append(message)
        self._build_indices()
        if self.embeddings_available and self.model:
            self._generate_embeddings()

    def save_memories(self, path: str = None) -> None:
        if path is None:
            path = self.memory_data_path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(self.messages)} memories to {path}")
        except IOError as e:
            print(f"Error saving memories: {e}")

    def get_stats(self) -> Dict:
        return {
            "total_memories": len(self.messages),
            "embedding_model": self.model_name,
            "embeddings_available": self.embeddings_available and self.model is not None,
            "embeddings_generated": self.embeddings is not None,
            "vector_store_enabled": self._vector_store is not None,
        }
