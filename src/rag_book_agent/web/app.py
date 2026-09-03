import json
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_book_agent.cli import project_directory
from rag_book_agent.audit_log import OperationLog
from rag_book_agent.config import load_settings
from rag_book_agent.memory import ConversationMemory
from rag_book_agent.service import RagService

PROJECT_DIR = project_directory()
STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".docx", ".xlsx", ".xls", ".pptx", ".csv"}
MAX_FILE_SIZE = 50 * 1024 * 1024
IMPORT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-import")
IMPORT_JOBS = {}
IMPORT_LOCK = threading.Lock()

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
    agent_mode: bool = False  # compatibility: true now selects LangGraph
    agent_engine: str = "classic"


class DocumentUpdate(BaseModel):
    enabled: bool = None
    in_library: bool = None


class EvaluationRequest(BaseModel):
    method: str = "retrieval"
    top_k: int = 10
    max_questions: int = 0


def new_service() -> RagService:
    return RagService(load_settings(PROJECT_DIR))


def answer_for_request(service: RagService, request: AskRequest):
    if request.agent_engine == "langgraph" or request.agent_mode:
        return service.ask_langgraph_agent(request.question.strip(), request.session_id, request.force_web)
    return service.ask(request.question.strip(), request.session_id, request.force_web)


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


@app.get("/api/langgraph/diagram")
def langgraph_diagram() -> Response:
    service = new_service()
    try:
        from rag_book_agent.agent.langgraph_workflow import LangGraphAgent

        return Response(content=LangGraphAgent(service).draw_png(), media_type="image/png")
    except Exception as error:
        raise HTTPException(status_code=503, detail="LangGraph 图渲染失败：%s" % error)
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


@app.get("/api/traces")
def traces(limit: int = 80) -> dict:
    service = new_service()
    try:
        return {"traces": service.storage.list_trace_details(max(1, min(limit, 200)))}
    finally:
        service.close()


@app.post("/api/evaluations/run")
def run_evaluation(request: EvaluationRequest) -> dict:
    if request.method == "pdf_quality":
        service = new_service()
        try:
            rows = service.storage.list_documents()
            pdfs = [row for row in rows if row["file_type"] == "pdf"]
            details = []
            for row in pdfs:
                chunks = service.storage.list_chunks([row["id"]], active_only=False)
                text_chars = sum(len(item["text"]) for item in chunks)
                details.append({"document": row["title"], "pages": row["page_count"], "chunks": len(chunks), "text_chars": text_chars, "quality_score": 1.0 if text_chars > 100 else 0.0})
            average = sum(item["quality_score"] for item in details) / len(details) if details else 0.0
            return {"method": "pdf_quality", "model": "轻量规则检查器", "documents": len(pdfs), "score": average, "details": details}
        finally:
            service.close()
    if request.method == "ragas":
        report_path = PROJECT_DIR / "data" / "reports" / "ragas-latest.json"
        if not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail="尚无 RAGAS 报告。请按 docs/DEPLOYMENT.md 在独立 RAGAS 环境运行评测脚本。",
            )
        try:
            return {"method": "ragas", "cached": True, "report": json.loads(report_path.read_text(encoding="utf-8"))}
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="RAGAS 报告格式无效。")

    from rag_book_agent.evaluation import Evaluator

    service = new_service()
    try:
        top_k = max(1, min(request.top_k, 20))
        max_questions = request.max_questions if request.max_questions > 0 else None
        report = Evaluator(service.storage, service.retriever).run(top_k=top_k, max_questions=max_questions)
        Evaluator.save(report, PROJECT_DIR / "data" / "reports" / "retrieval-latest.json")
        return {"method": "retrieval", "cached": False, "report": report}
    finally:
        service.close()


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
                status_code=400, detail="支持 PDF、Markdown、TXT、Word、Excel、PPT 和 CSV。"
            )
        if extension in {".docx", ".xlsx", ".xls", ".pptx"}:
            signature = await uploaded.read(4)
            await uploaded.seek(0)
            if extension != ".xls" and signature != b"PK\\x03\\x04":
                raise HTTPException(status_code=400, detail="Office 文件格式无效。")

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

    jobs = []
    immediate = {"found": 0, "imported": 0, "skipped": 0, "chunks": 0}
    for path in saved_paths:
        if path.suffix.lower() in {".md", ".markdown", ".txt", ".csv"}:
            service = new_service()
            try:
                result = service.ingest(path)
                for key in immediate:
                    immediate[key] += result[key]
            finally:
                service.close()
        else:
            job_id = uuid.uuid4().hex[:12]
            with IMPORT_LOCK:
                IMPORT_JOBS[job_id] = {"id": job_id, "name": path.name, "status": "queued", "progress": 0, "message": "等待处理"}
            IMPORT_EXECUTOR.submit(_run_import_job, job_id, path)
            jobs.append(job_id)
    return {"uploaded": [path.name for path in saved_paths], "import": immediate, "jobs": jobs}


def _run_import_job(job_id: str, path: Path) -> None:
    _update_import_job(job_id, status="processing", progress=8, message="正在解析文件")
    service = new_service()
    try:
        _update_import_job(job_id, progress=35, message="正在提取文本、图片和表格")
        result = service.ingest(path)
        if result["chunks"] == 0 and result["imported"] > 0:
            message = "处理完成，但未生成片段；请查看导入日志中的 IMPORT_WARN"
            _update_import_job(job_id, status="warning", progress=100, message=message, result=result)
            OperationLog(PROJECT_DIR).write("IMPORT_JOB_WARN", "file=%s | %s" % (path.name, message))
        else:
            _update_import_job(job_id, status="done", progress=100, message="处理完成", result=result)
            OperationLog(PROJECT_DIR).write("IMPORT_JOB_DONE", "file=%s | %s" % (path.name, result))
    except Exception as error:
        message = "处理失败：%s" % str(error)[:240]
        _update_import_job(job_id, status="error", progress=100, message=message)
        OperationLog(PROJECT_DIR).write("IMPORT_JOB_ERROR", "file=%s | %s" % (path.name, message))
    finally:
        service.close()


def _update_import_job(job_id: str, **values) -> None:
    with IMPORT_LOCK:
        if job_id in IMPORT_JOBS:
            IMPORT_JOBS[job_id].update(values)


@app.get("/api/import-jobs")
def import_jobs() -> dict:
    with IMPORT_LOCK:
        return {"jobs": list(IMPORT_JOBS.values())[-50:]}


@app.get("/api/import-jobs/{job_id}")
def import_job(job_id: str) -> dict:
    with IMPORT_LOCK:
        job = IMPORT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="导入任务不存在")
    return job


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
        answer = answer_for_request(service, request)
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
            if request.agent_mode or request.agent_engine == "langgraph":
                yield json.dumps({"type": "progress", "stage": "planner", "text": "正在拆解问题并规划研究路径..."}, ensure_ascii=False) + "\n"
                yield json.dumps({"type": "progress", "stage": "researcher", "text": "正在准备本地文档与网页证据..."}, ensure_ascii=False) + "\n"
            answer = answer_for_request(service, request)
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


@app.post("/api/langgraph/run/stream")
def run_langgraph_stream(request: AskRequest) -> StreamingResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def events():
        service = new_service()
        try:
            from rag_book_agent.agent.langgraph_workflow import LangGraphAgent

            agent = LangGraphAgent(service)
            for node, update in agent.stream(question, request.session_id, request.force_web):
                yield json.dumps({"type": "node", "node": node, "update": summarize_graph_update(update)}, ensure_ascii=False) + "\n"
            answer = update["answer"]
            yield json.dumps({"type": "done", "answer": answer.text, "mode": answer.mode, "retrieval": service.last_retrieval, "sources": serialize_sources(answer.sources)}, ensure_ascii=False) + "\n"
        except Exception as error:
            yield json.dumps({"type": "error", "message": str(error)[:300]}, ensure_ascii=False) + "\n"
        finally:
            service.close()

    return StreamingResponse(events(), media_type="application/x-ndjson")


def summarize_graph_update(update: dict) -> dict:
    if "evidence_preview" in update:
        return {"sources": update["evidence_preview"], "budget": update.get("budget", {})}
    if "next_action" in update:
        return {"action": update["next_action"], "reason": update.get("routing_reason", "")}
    if "observation" in update:
        return {"observation": update["observation"], "step": update.get("step_count", 0)}
    if "plan" in update:
        return {"sub_questions": len(update["plan"]), "planner": update.get("planner_mode", "rule")}
    if "lane" in update:
        return {"lane": update["lane"], "reason": update.get("lane_reason", "")}
    if "routing_reason" in update:
        return {"route": update["routing_reason"]}
    if "local_results" in update:
        return {"results": len(update["local_results"])}
    if "web_results" in update:
        return {"results": len(update["web_results"]), "status": update.get("web_search_status", "")}
    if "answer" in update:
        return {"mode": update["answer"].mode}
    return {"events": len(update.get("trace", []))}


def serialize_sources(items):
    return [
        {"id": source.chunk.id, "title": source.document_title, "page": source.chunk.page_start,
         "heading": source.chunk.heading, "text": source.chunk.text, "score": round(source.rerank_score, 4)}
        for source in items
    ]


def run() -> None:
    import uvicorn

    uvicorn.run("rag_book_agent.web.app:app", host="127.0.0.1", port=8008, reload=False)
