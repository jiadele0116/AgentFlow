"""FastAPI 服务模块。

提供 Web 聊天界面、WebSocket 端点和文件上传接口。
"""
from pathlib import Path

from fastapi import FastAPI, WebSocket, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
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

    # ── 文件上传：GET 显示上传页面，POST 处理上传 ──

    @app.get("/upload", response_class=HTMLResponse)
    async def upload_page():
        """文件上传页面（浏览器直接访问时显示）。"""
        return HTMLResponse(content=_UPLOAD_HTML)

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """文件上传接口（表单提交）。

        接收 multipart/form-data 格式的文件，保存到 data/workspace/uploads/。

        Args:
            file: 上传的文件（字段名必须是 "file"）

        Returns:
            JSON: {"status": "ok", "file": {filename, path, size, type}}
                  或 {"status": "error", "message": "..."}
        """
        try:
            result = await save_upload_file(file)
            logger.info(f"文件上传成功: {result['filename']} ({result['size']} 字节)")
            return {"status": "ok", "file": result}
        except ValueError as e:
            logger.warning(f"文件上传被拒绝: {e}")
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            return JSONResponse(status_code=500, content={"status": "error", "message": f"上传失败: {e}"})

    @app.post("/upload/multiple")
    async def upload_multiple(files: list[UploadFile] = File(...)):
        """批量文件上传接口。

        Args:
            files: 上传的文件列表

        Returns:
            JSON: {"status": "ok", "files": [...], "errors": [...]}
        """
        results = []
        errors = []
        for file in files:
            try:
                result = await save_upload_file(file)
                results.append(result)
                logger.info(f"文件上传成功: {result['filename']}")
            except ValueError as e:
                errors.append({"filename": file.filename, "error": str(e)})
            except Exception as e:
                errors.append({"filename": file.filename, "error": f"上传失败: {e}"})
        return {"status": "ok", "files": results, "errors": errors}

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """WebSocket 聊天端点。"""
        await handle_websocket_chat(websocket, chat_manager)

    return app


# ── 上传页面 HTML（内联，不依赖外部文件） ──

_UPLOAD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>文件上传 - Multi-Agent System</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    background: #f8fafc; color: #1e293b;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; padding: 20px;
  }
  .container {
    background: #fff; border-radius: 16px; padding: 40px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06); max-width: 520px; width: 100%;
  }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 6px; }
  .subtitle { font-size: 13px; color: #64748b; margin-bottom: 24px; }
  .drop-zone {
    border: 2px dashed #cbd5e1; border-radius: 12px; padding: 48px 24px;
    text-align: center; cursor: pointer; transition: all 0.2s;
    background: #fafbfc;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: #6366f1; background: #eef2ff;
  }
  .drop-zone .icon { font-size: 36px; margin-bottom: 8px; }
  .drop-zone .text { font-size: 14px; font-weight: 500; color: #334155; }
  .drop-zone .hint { font-size: 12px; color: #94a3b8; margin-top: 4px; }
  .drop-zone input { display: none; }
  .file-list { margin-top: 16px; }
  .file-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; background: #f1f5f9; border-radius: 8px;
    margin-bottom: 6px; font-size: 13px;
  }
  .file-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-item .size { color: #64748b; font-size: 12px; }
  .file-item .status { font-size: 12px; font-weight: 500; }
  .file-item .status.ok { color: #10b981; }
  .file-item .status.err { color: #ef4444; }
  .file-item .remove {
    background: none; border: none; cursor: pointer; color: #94a3b8;
    font-size: 16px; padding: 0 4px;
  }
  .file-item .remove:hover { color: #ef4444; }
  .upload-all {
    margin-top: 16px; width: 100%; padding: 10px;
    background: #6366f1; color: #fff; border: none; border-radius: 8px;
    font-size: 14px; font-weight: 500; cursor: pointer; transition: background 0.2s;
  }
  .upload-all:hover { background: #4f46e5; }
  .upload-all:disabled { background: #c7d2fe; cursor: not-allowed; }
  .result { margin-top: 12px; padding: 10px; border-radius: 8px; font-size: 13px; }
  .result.ok { background: #d1fae5; color: #065f46; }
  .result.err { background: #fee2e2; color: #991b1b; }
  .back { display: inline-block; margin-top: 16px; color: #6366f1; font-size: 13px; text-decoration: none; }
  .back:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="container">
  <h1>📎 文件上传</h1>
  <p class="subtitle">支持文档、代码、图片等格式，单文件最大 10MB</p>

  <div class="drop-zone" id="dropZone">
    <div class="icon">📁</div>
    <div class="text">点击选择文件 或 拖拽文件到这里</div>
    <div class="hint">支持 txt, md, pdf, doc, csv, json, py, js, png, jpg 等</div>
    <input type="file" id="fileInput" multiple>
  </div>

  <div class="file-list" id="fileList"></div>
  <button class="upload-all" id="uploadBtn" style="display:none" onclick="uploadAll()">上传全部</button>
  <div id="result"></div>
  <a class="back" href="/">← 返回聊天</a>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const uploadBtn = document.getElementById('uploadBtn');
const resultEl = document.getElementById('result');
let selectedFiles = [];

// 点击选择
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  addFiles(fileInput.files);
  fileInput.value = '';
});

// 拖拽
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  addFiles(e.dataTransfer.files);
});

function addFiles(files) {
  for (const f of files) selectedFiles.push({ file: f, status: 'pending' });
  render();
}

function removeFile(i) {
  selectedFiles.splice(i, 1);
  render();
}

function render() {
  fileList.innerHTML = selectedFiles.map((f, i) => {
    const size = f.file.size < 1024 ? f.file.size + ' B'
               : f.file.size < 1048576 ? (f.file.size/1024).toFixed(1) + ' KB'
               : (f.file.size/1048576).toFixed(1) + ' MB';
    const statusHtml = f.status === 'ok' ? '<span class="status ok">✓ 上传成功</span>'
                     : f.status === 'error' ? `<span class="status err">✗ ${f.error}</span>`
                     : '';
    return `<div class="file-item">
      <span class="name">${escapeHtml(f.file.name)}</span>
      <span class="size">${size}</span>
      ${statusHtml}
      <button class="remove" onclick="removeFile(${i})">×</button>
    </div>`;
  }).join('');
  uploadBtn.style.display = selectedFiles.some(f => f.status === 'pending') ? '' : 'none';
}

async function uploadAll() {
  uploadBtn.disabled = true;
  uploadBtn.textContent = '上传中...';
  resultEl.innerHTML = '';

  for (const f of selectedFiles) {
    if (f.status !== 'pending') continue;
    try {
      const formData = new FormData();
      formData.append('file', f.file);
      const resp = await fetch('/upload', { method: 'POST', body: formData });
      const data = await resp.json();
      if (data.status === 'ok') {
        f.status = 'ok';
      } else {
        f.status = 'error'; f.error = data.message;
      }
    } catch (e) {
      f.status = 'error'; f.error = e.message;
    }
  }

  const okCount = selectedFiles.filter(f => f.status === 'ok').length;
  const errCount = selectedFiles.filter(f => f.status === 'error').length;
  if (errCount === 0) {
    resultEl.innerHTML = `<div class="result ok">✓ 全部上传成功（${okCount} 个文件）</div>`;
  } else {
    resultEl.innerHTML = `<div class="result err">上传完成：${okCount} 成功，${errCount} 失败</div>`;
  }

  uploadBtn.textContent = '上传全部';
  uploadBtn.disabled = false;
  render();
}

function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
</script>
</body>
</html>""";

