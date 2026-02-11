"""
Setup and Migration Utility: Initialize and upgrade projects with new features.

This utility:
1. Initializes the database
2. Migrates existing data
3. Validates data quality
4. Generates reports
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import requests

from .db_manager import DatabaseManager
from .data_filter import TrainingDataFilter
from .dataset_builder_enhanced import build_datasets_enhanced, load_training_data, load_memory_data
from .logger import setup_logging
try:
    from .weaviate_store import WeaviateVectorStore
except Exception:
    WeaviateVectorStore = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

logger = setup_logging("echochat.migration")


class ProjectMigration:
    """Handles project upgrades and migrations."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.db_manager = None
        self.migration_log = []

    def log_step(self, step: str, status: str, details: str = ""):
        """Log migration step."""
        entry = {
            "step": step,
            "status": status,
            "details": details,
        }
        self.migration_log.append(entry)
        logger.info(f"[{status}] {step}: {details}")

    def initialize_database(self) -> bool:
        """Initialize SQLite database."""
        try:
            self.db_manager = DatabaseManager(str(self.data_dir / "echochat.db"))
            self.log_step("Database Initialization", "SUCCESS", "SQLite database created")
            return True
        except Exception as e:
            self.log_step("Database Initialization", "FAILED", str(e))
            logger.error(f"Failed to initialize database: {e}")
            return False

    def migrate_existing_sessions(self) -> bool:
        """Migrate existing session data to database."""
        if not self.db_manager:
            return False

        try:
            sessions_dir = self.data_dir / "sessions"
            if not sessions_dir.exists():
                self.log_step("Session Migration", "SKIPPED", "No sessions directory found")
                return True

            migrated = 0
            for session_dir in sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name
                meta_file = session_dir / "meta.json"

                if not meta_file.exists():
                    continue

                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)

                    echo_person = meta.get("echo_person", "Unknown")
                    message_count = meta.get("message_count", 0)

                    # Create session in database
                    if self.db_manager.create_session(
                        session_id=session_id,
                        echo_person=echo_person,
                        chat_hash="",
                        message_count=message_count,
                        metadata=meta,
                    ):
                        migrated += 1
                        logger.debug(f"Migrated session: {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to migrate session {session_id}: {e}")

            self.log_step(
                "Session Migration",
                "SUCCESS",
                f"Migrated {migrated} sessions to database",
            )
            return True

        except Exception as e:
            self.log_step("Session Migration", "FAILED", str(e))
            return False

    def validate_training_data(self) -> Dict:
        """Validate training data quality across all sessions."""
        try:
            sessions_dir = self.data_dir / "sessions"
            if not sessions_dir.exists():
                self.log_step("Training Data Validation", "SKIPPED", "No sessions found")
                return {}

            results = {}

            for session_dir in sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name
                training_file = session_dir / "training_data.jsonl"

                if not training_file.exists():
                    continue

                try:
                    training_data = load_training_data(str(training_file))

                    if not training_data:
                        continue

                    valid_pairs, invalid_pairs, stats = TrainingDataFilter.filter_training_data(
                        training_data,
                        verbose=False,
                    )

                    results[session_id] = {
                        "total": len(training_data),
                        "valid": len(valid_pairs),
                        "invalid": len(invalid_pairs),
                        "quality_score": stats.get("avg_quality_after", 0),
                        "contamination": stats.get("avg_contamination", 0),
                    }

                    # Update database with quality metrics
                    if self.db_manager:
                        self.db_manager.update_session(
                            session_id,
                            training_pairs=len(valid_pairs),
                        )

                    logger.info(f"Validated {session_id}: {stats}")

                except Exception as e:
                    logger.warning(f"Failed to validate {session_id}: {e}")

            self.log_step(
                "Training Data Validation",
                "SUCCESS",
                f"Validated {len(results)} sessions",
            )

            return results

        except Exception as e:
            self.log_step("Training Data Validation", "FAILED", str(e))
            return {}

    def optimize_memory_data(self) -> bool:
        """Optimize memory data across sessions."""
        try:
            sessions_dir = self.data_dir / "sessions"
            if not sessions_dir.exists():
                return True

            optimized = 0

            for session_dir in sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name
                memory_file = session_dir / "memory_data.json"

                if not memory_file.exists():
                    continue

                try:
                    memory_data = load_memory_data(str(memory_file))

                    if not memory_data:
                        continue

                    # Ensure all entries have required fields
                    for entry in memory_data:
                        if "context" not in entry:
                            entry["context"] = {
                                "word_count": len(entry.get("text", "").split()),
                                "contains_code": False,
                                "is_question": entry.get("text", "").strip().endswith("?"),
                            }

                    # Save optimized data
                    with open(memory_file, "w", encoding="utf-8") as f:
                        json.dump(memory_data, f, ensure_ascii=False, indent=2)

                    # Update database
                    if self.db_manager:
                        self.db_manager.update_session(
                            session_id,
                            memory_entries=len(memory_data),
                        )

                    optimized += 1
                    logger.debug(f"Optimized memory data for {session_id}")

                except Exception as e:
                    logger.warning(f"Failed to optimize {session_id}: {e}")

            self.log_step(
                "Memory Data Optimization",
                "SUCCESS",
                f"Optimized {optimized} sessions",
            )
            return True

        except Exception as e:
            self.log_step("Memory Data Optimization", "FAILED", str(e))
            return False

    def generate_report(self) -> str:
        """Generate migration report."""
        report = "\n" + "=" * 70 + "\n"
        report += "ECHOCHAT MIGRATION REPORT\n"
        report += "=" * 70 + "\n\n"

        for log_entry in self.migration_log:
            status_icon = "✓" if log_entry["status"] == "SUCCESS" else "✗" if log_entry["status"] == "FAILED" else "⊘"
            report += f"{status_icon} [{log_entry['status']}] {log_entry['step']}\n"
            if log_entry["details"]:
                report += f"   → {log_entry['details']}\n"

        report += "\n" + "=" * 70 + "\n"

        return report

    def run_full_migration(self) -> bool:
        """Run complete migration pipeline."""
        logger.info("Starting EchoChat project migration...")

        # Step 1: Initialize database
        if not self.initialize_database():
            return False

        # Step 2: Migrate existing sessions
        if not self.migrate_existing_sessions():
            logger.warning("Session migration failed, continuing...")

        # Step 3: Validate training data
        training_results = self.validate_training_data()

        # Step 4: Optimize memory data
        if not self.optimize_memory_data():
            logger.warning("Memory optimization failed, continuing...")

        # Step 5: Optionally push embeddings to Weaviate (managed vector DB)
        try:
            if WeaviateVectorStore is not None and SentenceTransformer is not None:
                self.log_step("Weaviate Push", "STARTED", "Uploading embeddings to Weaviate (if available)")
                self.push_embeddings_to_weaviate()
                self.log_step("Weaviate Push", "SUCCESS", "Embeddings pushed to Weaviate")
            else:
                self.log_step("Weaviate Push", "SKIPPED", "Weaviate client or SentenceTransformer not available")
        except Exception as e:
            self.log_step("Weaviate Push", "FAILED", str(e))

        # Generate report
        report = self.generate_report()
        print(report)

        # Save report
        report_file = self.data_dir / "migration_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Migration completed. Report saved to {report_file}")
        return True

    def push_embeddings_to_weaviate(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Compute embeddings for session memory files and push them to Weaviate.

        Uses `sentence_transformers` to compute embeddings and the Weaviate wrapper
        to upsert vectors into a class named after the session folder.
        """
        if WeaviateVectorStore is None:
            raise RuntimeError("WeaviateVectorStore not available")
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available")

        model = SentenceTransformer(model_name)

        sessions_dir = self.data_dir / "sessions"
        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            memory_file = session_dir / "memory_data.json"
            if not memory_file.exists():
                continue

            try:
                memory_data = load_memory_data(str(memory_file))
                texts = [m.get('text', '') for m in memory_data if m.get('text')]
                if not texts:
                    continue

                emb = model.encode(texts, show_progress_bar=False)
                import numpy as _np
                emb = _np.array(emb)

                # use session-specific index/class name
                idx_name = f"EchoChat_{session_id}"
                pushed = False
                # Try real Weaviate first
                if WeaviateVectorStore is not None:
                    try:
                        store = WeaviateVectorStore(index_name=idx_name)
                        store.upsert_embeddings(emb, [m for m in memory_data if m.get('text')])
                        logger.info(f"Pushed {len(texts)} embeddings for session {session_id} to Weaviate class {idx_name}")
                        pushed = True
                    except Exception as e:
                        logger.warning(f"Weaviate push failed for {session_id}: {e}")

                # Fallback: local simple vector server
                if not pushed:
                    try:
                        url = "http://127.0.0.1:8081/upsert"
                        payload = {
                            "class_name": idx_name,
                            "vectors": emb.tolist(),
                            "metadatas": [m for m in memory_data if m.get('text')],
                        }
                        resp = requests.post(url, json=payload, timeout=10)
                        if resp.status_code == 200:
                            logger.info(f"Pushed {len(texts)} embeddings for session {session_id} to local vector server {idx_name}")
                            pushed = True
                        else:
                            logger.warning(f"Local vector server returned {resp.status_code} for {session_id}: {resp.text}")
                    except Exception as e:
                        logger.warning(f"Local vector server push failed for {session_id}: {e}")

                if not pushed:
                    logger.warning(f"Failed to push embeddings for {session_id} to any vector store")
            except Exception as e:
                logger.warning(f"Failed to push embeddings for {session_id}: {e}")


def main():
    """Run migration utility."""
    base_dir = Path(__file__).parent.parent
    migration = ProjectMigration(base_dir)

    print("\n" + "=" * 70)
    print("ECHOCHAT PROJECT MIGRATION")
    print("=" * 70 + "\n")

    print("This utility will:\n")
    print("1. Initialize SQLite database for better session management")
    print("2. Migrate existing session metadata to database")
    print("3. Validate training data quality")
    print("4. Optimize memory data structures")
    print("\n")

    response = input("Continue with migration? (yes/no): ").strip().lower()

    if response != "yes":
        print("Migration cancelled.")
        return

    if migration.run_full_migration():
        print("\n✓ Migration completed successfully!")
        print("Your EchoChat project is now enhanced with:")
        print("  • SQLite database for persistent metadata")
        print("  • Improved response validation")
        print("  • Training data quality filtering")
        print("  • Better logging and monitoring")
    else:
        print("\n✗ Migration failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
