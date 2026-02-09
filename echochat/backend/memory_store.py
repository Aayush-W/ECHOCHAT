import json
from pathlib import Path
from typing import List, Dict

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


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

        if self.embeddings_available:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"Loaded embedding model: {model_name}")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                self.embeddings_available = False
        else:
            print("sentence-transformers not installed. Embeddings disabled.")

        self.load_memories()

    def load_memories(self) -> None:
        """Load memory data from JSON file."""
        try:
            with open(self.memory_data_path, 'r', encoding='utf-8') as f:
                self.messages = json.load(f)
            print(f"Loaded {len(self.messages)} memories")

            if self.embeddings_available and self.model:
                self._generate_embeddings()
        except FileNotFoundError:
            print(f"Memory file not found: {self.memory_data_path}")
            self.messages = []
        except json.JSONDecodeError as e:
            print(f"Memory file is not valid JSON: {e}")
            self.messages = []

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
        except Exception as e:
            print(f"Error generating embeddings: {e}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3
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
            return self.get_recent_messages(top_k)

        try:
            query_embedding = self.model.encode([query])[0]

            denom = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            denom = np.where(denom == 0, 1e-12, denom)
            similarities = np.dot(self.embeddings, query_embedding) / denom

            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if similarities[idx] >= similarity_threshold:
                    results.append({
                        'message': self.messages[idx],
                        'similarity': float(similarities[idx]),
                    })

            return results

        except Exception as e:
            print(f"Error during search: {e}")
            return self.get_recent_messages(top_k)

    def get_recent_messages(self, count: int = 5) -> List[Dict]:
        """Get most recent messages (fallback)."""
        if not self.messages:
            return []

        recent = sorted(
            self.messages,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:count]

        return [{'message': msg, 'similarity': 0.0} for msg in recent]

    def format_context(self, search_results: List[Dict]) -> str:
        """Format search results as context string for prompt."""
        if not search_results:
            return "No relevant memories found."

        context_lines = []
        for i, result in enumerate(search_results, 1):
            msg = result['message']
            similarity = result['similarity']
            timestamp = msg.get('timestamp', 'unknown')

            context_lines.append(
                f"[{i}] ({timestamp}, similarity: {similarity:.2f})\n"
                f"   Message: {msg.get('text', '')}"
            )

        return '\n'.join(context_lines)

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
