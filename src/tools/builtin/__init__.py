"""Built-in tool registration."""
from src.tools.registry import ToolRegistry
from src.tools.builtin.file_ops import register_file_tools
from src.tools.builtin.data_ops import register_data_tools
from src.tools.builtin.code_exec import register_code_tools
from src.tools.builtin.network import register_network_tools
from src.tools.builtin.rag_ops import register_rag_tools


def register_all_builtin_tools(registry: ToolRegistry, workspace_dir: str = ".",
                                retrieval_pipeline=None, document_processor=None):
    register_file_tools(registry)
    register_data_tools(registry)
    register_code_tools(registry, workspace_dir=workspace_dir)
    register_network_tools(registry)
    register_rag_tools(registry, retrieval_pipeline=retrieval_pipeline, document_processor=document_processor)
