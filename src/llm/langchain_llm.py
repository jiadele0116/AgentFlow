"""LangChain LLM 工厂模块。

根据系统配置创建对应的 LangChain ChatModel 实例。
"""
from langchain_core.language_models import BaseChatModel

from src.core.config import Settings
from src.core.logging import get_logger

logger = get_logger("llm.langchain")


def create_langchain_llm(settings: Settings) -> BaseChatModel:
    """根据配置创建 LangChain ChatModel 实例。

    Args:
        settings: 系统配置

    Returns:
        配置好的 LangChain ChatModel

    Raises:
        ValueError: 不支持的 LLM 提供商
    """
    provider = settings.llm.provider.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=settings.llm.model,
            anthropic_api_key=settings.llm.api_key,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
            timeout=settings.llm.timeout,
        )
        logger.info(f"Created ChatAnthropic: {settings.llm.model}")
        return llm

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=getattr(settings.llm, 'base_url', None),
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
        logger.info(f"Created ChatOpenAI: {settings.llm.model}")
        return llm

    elif provider == "qwen":
        from langchain_community.chat_models.tongyi import ChatTongyi
        llm = ChatTongyi(
            model=settings.llm.model,
            dashscope_api_key=settings.llm.api_key,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
        logger.info(f"Created ChatTongyi: {settings.llm.model}")
        return llm

    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")
