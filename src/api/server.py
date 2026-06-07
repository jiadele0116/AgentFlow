"""FastAPI 服务模块。

提供 Web 聊天界面、WebSocket 端点和文件上传接口。
"""
from pathlib import Path

from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.chat import ChatManager, handle_websocket_chat
from src.api.upload import save_upload_file
from src.core.logging import get_logger

logger = get_logger("api.server")

STATIC_DIR = Path(__file__).parent / "static"


def create_app(graph, memory=None) -> FastAPI:
    """创建 FastAPI 应用实例。

    Args:
        graph: LangGraph 编译好的 StateGraph 实例
        memory: 记忆管理器实例（可选）

    Returns:
        配置好的 FastAPI 应用
    """
    app = FastAPI(title="Multi-Agent System", version="0.1.0")
    chat_manager = ChatManager(graph, memory=memory)

    # 挂载静态文件
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """返回聊天页面。"""
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @app.get("/guide", response_class=HTMLResponse)
    async def guide():
        """返回技术文档页面。"""
        html_path = STATIC_DIR / "docs.html"
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health():
        """健康检查端点。"""
        return {"status": "ok"}

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """文件上传接口。

        Args:
            file: 上传的文件

        Returns:
            包含文件信息的 JSON
        """
        try:
            result = await save_upload_file(file)
            logger.info(f"File uploaded: {result['filename']}")
            return {"status": "ok", "file": result}
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return {"status": "error", "message": "上传失败"}

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """WebSocket 聊天端点。"""
        await handle_websocket_chat(websocket, chat_manager)

    return app
