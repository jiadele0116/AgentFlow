import json
import logging
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """JSON formatter that includes trace_id and structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id

        for key in ("agent_id", "tool_name", "skill_name", "duration_ms"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """Human-readable formatter with optional trace_id."""

    def format(self, record: logging.LogRecord) -> str:
        trace = getattr(record, "trace_id", "")
        if trace:
            record.msg = f"[{trace}] {record.getMessage()}"
            record.args = ()
        return f"{record.levelname} {record.name}: {record.getMessage()}"


def setup_logging(level: str = "INFO", json_format: bool = False):
    """Configure the multi-agent logging system."""
    logger = logging.getLogger("multi_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(SimpleFormatter())

    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the multi_agent namespace."""
    return logging.getLogger(f"multi_agent.{name}")
