"""
Database Manager: Persistent storage with SQLite for improved scalability.

Handles:
- Session metadata
- Training data quality metrics
- Response quality feedback
- Embeddings cache
- User preferences
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger("echochat.db")


class DatabaseManager:
    def __init__(self, db_path: str = "data/echochat.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    echo_person TEXT NOT NULL,
                    chat_hash TEXT UNIQUE,
                    message_count INTEGER,
                    training_pairs INTEGER,
                    memory_entries INTEGER,
                    personality_profile_path TEXT,
                    training_data_path TEXT,
                    memory_data_path TEXT,
                    embeddings_cache_path TEXT,
                    ollama_model TEXT,
                    local_adapter_path TEXT,
                    local_base_model TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    last_accessed_at TEXT,
                    status TEXT DEFAULT 'ready',
                    metadata JSON
                )
            """)

            # Response quality feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS response_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT,
                    ai_response TEXT,
                    rating INTEGER,
                    feedback TEXT,
                    is_llm_sounding BOOLEAN,
                    contains_sensitive BOOLEAN,
                    created_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Embeddings cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_text TEXT UNIQUE,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    created_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Training data quality table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    input_text TEXT,
                    output_text TEXT,
                    quality_score REAL,
                    is_appropriate BOOLEAN,
                    contamination_score REAL,
                    notes TEXT,
                    created_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Personality analysis history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    personality_profile JSON,
                    analysis_version TEXT,
                    created_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            logger.info("Database initialized successfully")

    def create_session(
        self,
        session_id: str,
        echo_person: str,
        chat_hash: str,
        message_count: int = 0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Create a new session record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.utcnow().isoformat()
                cursor.execute("""
                    INSERT INTO sessions (
                        session_id, echo_person, chat_hash, message_count,
                        created_at, updated_at, last_accessed_at, metadata, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, echo_person, chat_hash, message_count,
                    now, now, now, json.dumps(metadata or {}), "ready"
                ))
                logger.info(f"Created session: {session_id}")
                return True
        except sqlite3.IntegrityError as e:
            logger.warning(f"Session already exists: {session_id}")
            return False

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session metadata."""
        allowed_fields = {
            'echo_person', 'message_count', 'training_pairs', 'memory_entries',
            'personality_profile_path', 'training_data_path', 'memory_data_path',
            'embeddings_cache_path', 'ollama_model', 'local_adapter_path',
            'local_base_model', 'status', 'metadata'
        }

        update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not update_fields:
            return False

        update_fields['updated_at'] = datetime.utcnow().isoformat()
        update_fields['last_accessed_at'] = datetime.utcnow().isoformat()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
                values = list(update_fields.values()) + [session_id]
                cursor.execute(
                    f"UPDATE sessions SET {set_clause} WHERE session_id = ?",
                    values
                )
                logger.info(f"Updated session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            return False

    def log_response_quality(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        is_llm_sounding: bool = False,
        rating: Optional[int] = None,
        feedback: Optional[str] = None,
    ) -> bool:
        """Log response quality metrics for improvement."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO response_quality (
                        session_id, user_message, ai_response,
                        is_llm_sounding, rating, feedback, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, user_message, ai_response,
                    is_llm_sounding, rating, feedback,
                    datetime.utcnow().isoformat()
                ))
                return True
        except Exception as e:
            logger.error(f"Error logging response quality: {e}")
            return False

    def get_quality_metrics(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Get quality metrics for continuous improvement."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM response_quality
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def cache_embeddings(
        self,
        session_id: str,
        message_text: str,
        embedding: bytes,
        embedding_dim: int,
    ) -> bool:
        """Cache embeddings for faster lookup."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings_cache (
                        session_id, message_text, embedding, embedding_dim, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id, message_text, embedding, embedding_dim,
                    datetime.utcnow().isoformat()
                ))
                return True
        except Exception as e:
            logger.error(f"Error caching embeddings: {e}")
            return False

    def get_cached_embedding(self, message_text: str) -> Optional[Tuple]:
        """Get cached embedding if available."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT embedding, embedding_dim FROM embeddings_cache
                WHERE message_text = ?
            """, (message_text,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def log_training_quality(
        self,
        session_id: str,
        input_text: str,
        output_text: str,
        quality_score: float,
        is_appropriate: bool,
        contamination_score: float = 0.0,
        notes: Optional[str] = None,
    ) -> bool:
        """Log training data quality metrics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO training_quality (
                        session_id, input_text, output_text,
                        quality_score, is_appropriate, contamination_score,
                        notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, input_text, output_text,
                    quality_score, is_appropriate, contamination_score,
                    notes, datetime.utcnow().isoformat()
                ))
                return True
        except Exception as e:
            logger.error(f"Error logging training quality: {e}")
            return False

    def get_training_quality_stats(self, session_id: str) -> Dict:
        """Get training data quality statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(CASE WHEN is_appropriate THEN 1 ELSE 0 END) as appropriate_count,
                    AVG(quality_score) as avg_quality,
                    AVG(contamination_score) as avg_contamination
                FROM training_quality
                WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return {}

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Clean up sessions older than specified days."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM sessions
                    WHERE last_accessed_at < datetime('now', '-' || ? || ' days')
                    AND status != 'pinned'
                """, (days,))
                deleted = cursor.rowcount
                logger.info(f"Cleaned up {deleted} old sessions")
                return deleted
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0
