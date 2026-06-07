"""
日志系统模块 (Logging System)

本模块为多Agent系统提供统一的日志框架，支持两种输出格式：
1. StructuredFormatter: JSON 结构化日志，适合生产环境和日志收集系统
2. SimpleFormatter: 人类可读格式，适合本地开发和调试

日志命名空间层级：
  multi_agent (根)
  ├── multi_agent.core       - 核心模块日志
  ├── multi_agent.llm        - LLM 调用日志
  ├── multi_agent.tools      - 工具执行日志
  ├── multi_agent.rag        - RAG 检索日志
  ├── multi_agent.agents     - Agent 运行日志
  └── multi_agent.mcp        - MCP 通信日志

特殊字段说明：
  - trace_id: 请求追踪ID，用于关联同一任务链的所有日志
  - agent_id: 产生日志的 Agent 标识
  - tool_name: 工具名称（工具执行相关日志）
  - skill_name: 技能名称（技能执行相关日志）
  - duration_ms: 操作耗时（毫秒）
"""
import json
import logging
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """JSON 结构化日志格式化器。

    将日志记录转换为 JSON 格式，便于日志收集系统（如 ELK、Loki）解析。
    输出示例：
        {"timestamp":"2026-05-25T10:30:00+00:00","level":"INFO",
         "logger":"multi_agent.llm","message":"Request completed",
         "trace_id":"abc-123","duration_ms":1523}

    支持的额外字段（通过 record 属性注入）：
        - trace_id: 请求追踪ID
        - agent_id: Agent 标识
        - tool_name: 工具名称
        - skill_name: 技能名称
        - duration_ms: 操作耗时
    """

    def format(self, record: logging.LogRecord) -> str:
        # 构建基础日志条目
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 注入追踪ID（如果存在）
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id

        # 注入业务上下文字段
        for key in ("agent_id", "tool_name", "skill_name", "duration_ms"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # 附加异常堆栈信息（如果存在）
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """人类可读的日志格式化器，适用于本地开发和调试。

    输出格式: LEVEL logger_name: message
    如果存在 trace_id，会将其注入到消息前缀中：[trace_id] message
    """

    def format(self, record: logging.LogRecord) -> str:
        # 如果有 trace_id，将其注入到消息中以便追踪
        trace = getattr(record, "trace_id", "")
        if trace:
            record.msg = f"[{trace}] {record.getMessage()}"
            record.args = ()
        return f"{record.levelname} {record.name}: {record.getMessage()}"


def setup_logging(level: str = "INFO", json_format: bool = False):
    """初始化多Agent系统的日志配置。

    该函数在系统启动时调用一次，配置根日志器 "multi_agent"。
    会清除已有的处理器（防止重复配置），然后添加一个新的 StreamHandler。

    Args:
        level: 日志级别字符串 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        json_format: 是否使用 JSON 格式输出。
                     True = StructuredFormatter（生产环境）
                     False = SimpleFormatter（开发环境）
    """
    logger = logging.getLogger("multi_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 清除已有处理器，避免重复配置
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(SimpleFormatter())

    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的子日志器。

    所有子日志器都在 "multi_agent" 命名空间下，继承根日志器的配置。

    Args:
        name: 日志器名称，通常使用模块名如 "llm"、"tools"、"agents"

    Returns:
        配置好的 Logger 实例

    使用示例：
        logger = get_logger("llm")
        logger.info("LLM request started", extra={"trace_id": "abc-123"})
    """
    return logging.getLogger(f"multi_agent.{name}")
