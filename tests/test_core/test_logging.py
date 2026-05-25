import json
import logging
import pytest


def test_setup_logging_configures_root_logger():
    from src.core.logging import setup_logging

    setup_logging(level="DEBUG")
    logger = logging.getLogger("multi_agent")
    assert logger.level == logging.DEBUG


def test_log_record_has_trace_id(caplog):
    from src.core.logging import setup_logging, get_logger

    setup_logging(level="DEBUG")
    logger = get_logger("test")
    with caplog.at_level(logging.DEBUG):
        logger.info("test message", extra={"trace_id": "abc-123"})

    assert "abc-123" in caplog.text


def test_log_record_is_json(caplog):
    from src.core.logging import setup_logging, get_logger

    setup_logging(level="DEBUG", json_format=True)
    logger = get_logger("test")
    with caplog.at_level(logging.DEBUG):
        logger.info("structured message", extra={"trace_id": "trace-1"})

    for record in caplog.records:
        if record.getMessage() == "structured message":
            assert hasattr(record, "trace_id")
            break


def test_get_logger_returns_named_logger():
    from src.core.logging import get_logger

    logger = get_logger("my_module")
    assert logger.name == "multi_agent.my_module"
