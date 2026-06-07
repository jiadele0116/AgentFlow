"""
内置工具注册模块 (Built-in Tool Registration)

提供统一入口注册所有内置工具到工具注册表。

内置工具分组：
  - file_ops: 文件操作（read_file, write_file, search_files）
  - data_ops: 数据处理（parse_json, csv_to_dict, json_to_csv）
  - code_exec: 代码执行（run_python，使用沙箱环境）
  - network: 网络请求（fetch_url）
  - rag_ops: RAG 操作（search_knowledge, index_document）

使用示例：
    registry = ToolRegistry()
    register_all_builtin_tools(registry, workspace_dir="/workspace")
"""
from src.tools.registry import ToolRegistry
from src.tools.builtin.file_ops import register_file_tools
from src.tools.builtin.data_ops import register_data_tools
from src.tools.builtin.code_exec import register_code_tools
from src.tools.builtin.network import register_network_tools
from src.tools.builtin.rag_ops import register_rag_tools


def register_all_builtin_tools(registry: ToolRegistry, workspace_dir: str = ".",
                                retrieval_pipeline=None, document_processor=None):
    """注册所有内置工具到指定的注册表。

    Args:
        registry: 工具注册表实例
        workspace_dir: 工作目录（用于代码执行工具的路径基准）
        retrieval_pipeline: RAG 检索管道实例（用于 search_knowledge 工具）
        document_processor: 文档处理器实例（用于 index_document 工具）
    """
    register_file_tools(registry)
    register_data_tools(registry)
    register_code_tools(registry, workspace_dir=workspace_dir)
    register_network_tools(registry)
    register_rag_tools(registry, retrieval_pipeline=retrieval_pipeline, document_processor=document_processor)
