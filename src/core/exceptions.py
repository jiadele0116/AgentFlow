"""
异常体系模块 (Exception Hierarchy)

定义多Agent系统的统一异常层级结构。所有自定义异常都继承自 MultiAgentError，
便于上层代码统一捕获和处理。

异常层级：
  MultiAgentError (基类)
  ├── AgentError     - Agent 运行时错误（携带 agent_id）
  ├── ToolError      - 工具执行错误（携带 tool_name）
  ├── SkillError     - 技能执行错误（携带 skill_name）
  ├── RAGError       - RAG 检索/索引错误
  ├── MCPError       - MCP 协议通信错误（携带 server_name）
  └── ConfigError    - 配置加载/验证错误

设计原则：
  - 每个异常携带 code（错误码）和 context（上下文信息），便于日志和监控
  - 子类异常通过 **kwargs 透传父类参数，保持接口一致性
  - 异常消息应清晰描述问题，context 中包含调试所需的额外信息
"""
from typing import Any


class MultiAgentError(Exception):
    """多Agent系统的基础异常类。

    所有业务异常都应继承此类，提供统一的错误码和上下文机制。

    Attributes:
        code: 错误码字符串，用于程序化识别错误类型（如 "AGENT_TIMEOUT"）
        context: 错误上下文字典，包含调试信息（如重试次数、输入参数等）
    """

    def __init__(self, message: str, code: str = "", context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}


class AgentError(MultiAgentError):
    """Agent 运行时错误。

    当 Agent 在执行过程中遇到不可恢复的错误时抛出，
    例如：LLM 调用失败、策略执行异常、状态转换错误等。

    Attributes:
        agent_id: 产生错误的 Agent 标识符
    """

def __init__(self, message: str, agent_id: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.agent_id = agent_id


class ToolError(MultiAgentError):
    """工具执行错误。

    当工具执行过程中出现问题时抛出，
    例如：参数验证失败、执行超时、外部服务不可达等。

    Attributes:
        tool_name: 产生错误的工具名称
    """

    def __init__(self, message: str, tool_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class SkillError(MultiAgentError):
    """技能执行错误。

    当技能引擎执行 YAML 定义的技能流程时出现问题，
    例如：模板渲染失败、步骤执行错误、条件判断异常等。

    Attributes:
        skill_name: 产生错误的技能名称
    """

    def __init__(self, message: str, skill_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.skill_name = skill_name


class RAGError(MultiAgentError):
    """RAG 模块错误。

    涵盖 RAG 管道中的各种错误：
    - 文档分块失败
    - 嵌入模型加载/推理错误
    - 向量数据库操作错误
    - 检索管道执行错误
    """


class MCPError(MultiAgentError):
    """MCP 协议通信错误。

    当 MCP 客户端/服务器通信出现问题时抛出，
    例如：请求格式错误、工具调用失败、服务器无响应等。

    Attributes:
        server_name: 产生错误的 MCP 服务器名称
    """

    def __init__(self, message: str, server_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.server_name = server_name


class ConfigError(MultiAgentError):
    """配置相关错误。

    当配置加载或验证过程中出现问题时抛出，
    例如：YAML 格式错误、必需字段缺失、环境变量未设置等。
    """
