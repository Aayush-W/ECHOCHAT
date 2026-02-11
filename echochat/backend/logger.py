"""
Logging Module: Comprehensive logging for debugging and monitoring.

Features:
- Structured logging with levels
- Performance tracking
- Error tracking
- Request tracking
- Session tracking
"""

import logging
import logging.handlers
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

LOG_DIR = Path(__file__).parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class StructuredFormatter(logging.Formatter):
    """Format logs as structured JSON for better analysis."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add custom fields if available
        if hasattr(record, "session_id"):
            log_obj["session_id"] = record.session_id
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
        if hasattr(record, "response_time"):
            log_obj["response_time_ms"] = record.response_time

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up logging for a module."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler - JSON format
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / f"{name.replace('.', '_')}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    fh.setFormatter(StructuredFormatter())
    logger.addHandler(fh)

    # Console handler - simple format
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

    return logger


class PerformanceTracker:
    """Track performance metrics for optimization."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: Dict[str, list] = {}

    def log_response_time(self, operation: str, duration_ms: float):
        """Log operation timing."""
        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(duration_ms)

        # Log if unusually slow
        if duration_ms > 1000:  # More than 1 second
            self.logger.warning(
                f"Slow operation: {operation} took {duration_ms:.0f}ms",
                extra={"response_time": duration_ms},
            )

    def get_stats(self, operation: str) -> Optional[Dict]:
        """Get statistics for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return None

        times = self.metrics[operation]
        return {
            "count": len(times),
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "total_ms": sum(times),
        }

    def clear_stats(self):
        """Clear metrics."""
        self.metrics.clear()


class RequestLogger:
    """Log API requests and responses."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_request(
        self,
        method: str,
        path: str,
        session_id: Optional[str] = None,
        extra: Optional[Dict] = None,
    ):
        """Log incoming request."""
        msg = f"{method} {path}"
        if session_id:
            msg += f" [session: {session_id}]"

        self.logger.info(msg, extra={"session_id": session_id})

    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        session_id: Optional[str] = None,
    ):
        """Log outgoing response."""
        level = logging.WARNING if status_code >= 400 else logging.INFO
        msg = f"{method} {path} -> {status_code} ({duration_ms:.0f}ms)"

        self.logger.log(
            level,
            msg,
            extra={"session_id": session_id, "response_time": duration_ms},
        )

    def log_error(
        self,
        method: str,
        path: str,
        error: Exception,
        session_id: Optional[str] = None,
    ):
        """Log request error."""
        self.logger.error(
            f"{method} {path} - {type(error).__name__}: {str(error)}",
            exc_info=True,
            extra={"session_id": session_id},
        )


class ResponseQualityLogger:
    """Log response quality metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_generation(
        self,
        user_message: str,
        response: str,
        llm_score: float,
        is_valid: bool,
        session_id: Optional[str] = None,
    ):
        """Log response generation metrics."""
        status = "valid" if is_valid else "invalid"
        msg = f"Generated response [{status}] - LLM score: {llm_score:.1f}"

        self.logger.info(
            msg,
            extra={"session_id": session_id},
        )

        if not is_valid:
            self.logger.debug(
                f"User: {user_message[:100]}... -> Response: {response[:100]}...",
                extra={"session_id": session_id},
            )

    def log_validation_failure(
        self,
        response: str,
        issues: list,
        session_id: Optional[str] = None,
    ):
        """Log validation failures."""
        self.logger.warning(
            f"Response validation failed: {'; '.join(issues)}",
            extra={"session_id": session_id},
        )


# Global logger instances
api_logger = setup_logging("echochat.api")
responder_logger = setup_logging("echochat.responder")
db_logger = setup_logging("echochat.db")
filter_logger = setup_logging("echochat.filter")

# Global trackers
perf_tracker = PerformanceTracker(responder_logger)
request_logger = RequestLogger(api_logger)
response_quality_logger = ResponseQualityLogger(responder_logger)


def log_performance(operation: str, duration_ms: float):
    """Log operation performance."""
    perf_tracker.log_response_time(operation, duration_ms)


def get_performance_stats(operation: str) -> Optional[Dict]:
    """Get performance stats for an operation."""
    return perf_tracker.get_stats(operation)
