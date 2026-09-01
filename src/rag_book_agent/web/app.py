import json
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_book_agent.cli import project_directory
from rag_book_agent.audit_log import OperationLog
from rag_book_agent.config import load_settings
from rag_book_agent.memory import ConversationMemory
from rag_book_agent.service import RagService

PROJECT_DIR = project_directory()
STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
MAX_FILE_SIZE = 50 * 1024 * 1024

app = FastAPI(title="RAG Book Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def operation_audit(request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        status = 500
        raise
    finally:
        elapsed = int((time.perf_counter() - started) * 1000)
        # Do not append while FileResponse is streaming this same log file.
        if request.url.path != "/api/logs/export":
            OperationLog(PROJECT_DIR).write(
                "HTTP %s" % request.method,
                "%s -> %s (%d ms)" % (request.url.path, status, elapsed),
            )


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    force_web: bool = False
    agent_mode: bool = False


class DocumentUpdate(BaseModel):
    enabled: bool = None
    in_library: bool = None


def new_service() -> RagService:
    return RagService(load_settings(PROJECT_DIR))


@app.get("/")
def homepage() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    service = new_service()
    try:
        stats = service.stats()
        settings = service.settings
        api_configured = bool(settings.api_base and settings.api_model)
        key_available = bool(settings.api_key)
        return {
            "stats": stats,
            "generation": {
                "configured": api_configured,
                "key_available": key_available,
                "model": settings.api_model if api_configured else "",
                "mode": "deepseek" if api_configured and key_available else "evidence",
            },
        }
    finally:
        service.close()


@app.get("/api/logs/export")
def export_logs() -> FileResponse:
    path = OperationLog(PROJECT_DIR).path
    if not path.exists():
        OperationLog(PROJECT_DIR).write("日志导出", "暂无历史日志，创建日志文件")
    OperationLog(PROJECT_DIR).write("日志导出", "下载操作日志")
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename="rag-book-operations.txt",
    )


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)) -> dict:
    upload_dir = PROJECT_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uploaded in files:
        source_name = uploaded.filename or ""
        safe_name = Path(source_name).name
        extension = Path(safe_name).suffix.lower()
        if not safe_name or extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, detail="Only PDF, Markdown, and TXT are supported."
            )

        target = upload_dir / safe_name
        bytes_written = 0
        with target.open("wb") as handle:
            while True:
                block = await uploaded.read(1024 * 1024)
                if not block:
                    break
                bytes_written += len(block)
                if bytes_written > MAX_FILE_SIZE:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="A file is larger than 50 MB.")
                handle.write(block)
        saved_paths.append(target)

    service = new_service()
    try:
        imported = {"found": 0, "imported": 0, "skipped": 0, "chunks": 0}
        for path in saved_paths:
            try:
                result = service.ingest(path)
            except Exception as error:
                # Do not leak a plain-text 500 response to the browser.
                raise HTTPException(
                    status_code=422,
                    detail="无法解析文件 %s：%s。加密 PDF 请安装 cryptography 并提供可读取文件。"
                    % (path.name, error),
                )
            for key in imported:
                imported[key] += result[key]
        return {"uploaded": [path.name for path in saved_paths], "import": imported}
    finally:
        service.close()


@app.get("/api/documents")
def documents() -> dict:
    service = new_service()
    try:
        items = []
        for row in service.storage.list_documents():
            path = Path(row["path"])
            size = path.stat().st_size if path.exists() else 0
            items.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "path": row["path"],
                    "file_type": row["file_type"],
                    "size": size,
                    "page_count": row["page_count"],
                    "chunk_count": row["chunk_count"],
                    "indexed_at": row["indexed_at"],
                    "enabled": bool(row["enabled"]),
                    "in_library": bool(row["in_library"]),
                }
            )
        return {"documents": items}
    finally:
        service.close()


@app.get("/api/documents/{document_id}/chunks")
def document_chunks(document_id: int) -> dict:
    service = new_service()
    try:
        rows = service.storage.list_chunks([document_id], active_only=False)
        if not rows:
            raise HTTPException(status_code=404, detail="Document not found.")
        all_rows = service.storage.connection.execute(
            "SELECT id, parent_id, chunk_type, heading, page_start, page_end, position, text "
            "FROM chunks WHERE document_id = ? ORDER BY position",
            (document_id,),
        ).fetchall()
        return {
            "chunks": [
                {
                    "id": row["id"], "parent_id": row["parent_id"],
                    "chunk_type": row["chunk_type"], "heading": row["heading"],
                    "page_start": row["page_start"], "page_end": row["page_end"],
                    "position": row["position"], "preview": row["text"][:240],
                }
                for row in all_rows
            ]
        }
    finally:
        service.close()


@app.get("/api/sessions")
def sessions() -> dict:
    return {"sessions": ConversationMemory.list_sessions(PROJECT_DIR / "data")}


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str) -> dict:
    memory = ConversationMemory(PROJECT_DIR / "data", session_id)
    return {"id": session_id, "turns": memory.turns()}


@app.patch("/api/documents/{document_id}")
def update_document(document_id: int, update: DocumentUpdate) -> dict:
    service = new_service()
    try:
        if not service.storage.update_document(document_id, update.enabled, update.in_library):
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"ok": True}
    finally:
        service.close()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int) -> dict:
    service = new_service()
    try:
        if not service.storage.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"ok": True}
    finally:
        service.close()


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    service = new_service()
    try:
        answer = (
            service.ask_agent(question, request.session_id, request.force_web)
            if request.agent_mode
            else service.ask(question, request.session_id, request.force_web)
        )
        return {
            "answer": answer.text,
            "mode": answer.mode,
            "retrieval": getattr(service, "last_retrieval", {}),
            "sources": [
                {
                    "id": source.chunk.id,
                    "title": source.document_title,
                    "page": source.chunk.page_start,
                    "heading": source.chunk.heading,
                    "text": source.chunk.text,
                    "score": round(source.rerank_score, 4),
                }
                for source in answer.sources
            ],
            "web_sources": getattr(service, "last_web_results", []),
        }
    finally:
        service.close()


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def events():
        service = new_service()
        try:
            if request.agent_mode:
                yield json.dumps({"type": "progress", "stage": "planner", "text": "正在拆解问题并规划研究路径..."}, ensure_ascii=False) + "\n"
                yield json.dumps({"type": "progress", "stage": "researcher", "text": "正在并行准备本地文档与网页证据..."}, ensure_ascii=False) + "\n"
            answer = (
                service.ask_agent(question, request.session_id, request.force_web)
                if request.agent_mode
                else service.ask(question, request.session_id, request.force_web)
            )
            yield (
                json.dumps(
                    {
                        "type": "meta",
                        "mode": answer.mode,
                        "retrieval": getattr(service, "last_retrieval", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            text = answer.text
            for start in range(0, len(text), 28):
                yield (
                    json.dumps(
                        {"type": "delta", "text": text[start : start + 28]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                time.sleep(0.012)
            yield (
                json.dumps(
                    {
                        "type": "done",
                        "sources": [
                            {
                                "id": source.chunk.id,
                                "title": source.document_title,
                                "page": source.chunk.page_start,
                                "heading": source.chunk.heading,
                                "text": source.chunk.text,
                                "score": round(source.rerank_score, 4),
                            }
                            for source in answer.sources
                        ],
                        "web_sources": getattr(service, "last_web_results", []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        finally:
            service.close()

    return StreamingResponse(events(), media_type="application/x-ndjson")


def run() -> None:
    import uvicorn

    uvicorn.run("rag_book_agent.web.app:app", host="127.0.0.1", port=8008, reload=False)
