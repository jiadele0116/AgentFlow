"""文件上传处理模块。

提供文件上传、保存和读取功能。
"""
import os
import uuid
from pathlib import Path

from fastapi import UploadFile
from src.core.logging import get_logger

logger = get_logger("api.upload")

# 上传目录
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "workspace" / "uploads"

# 支持的文件类型
ALLOWED_EXTENSIONS = {
    # 文档
    ".txt", ".md", ".pdf", ".doc", ".docx", ".csv", ".json", ".yaml", ".yml",
    # 代码
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".html", ".css",
    # 图片
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    # 数据
    ".xml", ".sql", ".log",
}

# 最大文件大小 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


def get_upload_dir() -> Path:
    """获取上传目录，不存在则创建。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


async def save_upload_file(file: UploadFile) -> dict:
    """保存上传的文件。

    Args:
        file: FastAPI UploadFile 对象

    Returns:
        包含文件信息的字典 {filename, path, size, type}
    """
    # 检查文件扩展名
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {ext}")

    # 读取文件内容
    content = await file.read()
    size = len(content)

    # 检查文件大小
    if size > MAX_FILE_SIZE:
        raise ValueError(f"文件太大: {size} bytes (最大 {MAX_FILE_SIZE})")

    # 生成唯一文件名
    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    upload_dir = get_upload_dir()
    file_path = upload_dir / unique_name

    # 保存文件
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(f"File saved: {file_path} ({size} bytes)")

    return {
        "filename": file.filename,
        "path": str(file_path),
        "size": size,
        "type": ext,
    }


def read_file_content(file_path: str) -> str:
    """读取文件内容（文本文件）。

    Args:
        file_path: 文件路径

    Returns:
        文件文本内容
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 图片文件返回描述
    ext = path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
        return f"[图片文件: {path.name}, 大小: {path.stat().st_size} bytes]"

    # 文本文件读取内容
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="gbk")
        except:
            return f"[二进制文件: {path.name}, 大小: {path.stat().st_size} bytes]"
