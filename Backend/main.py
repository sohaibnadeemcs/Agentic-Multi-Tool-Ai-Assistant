"""
FastAPI backend for the Agentic Multi-Tool AI Assistant.

Endpoints:
  POST /chat            -> streaming chat response (text/event-stream)
  POST /upload_pdf       -> upload + index a PDF for rag_qa
  GET  /history/{sid}    -> full session transcript
  GET  /health           -> liveness/readiness check
"""
import json
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from backend.config import settings
from backend.logger import get_logger
from backend.models.schemas import ChatRequest, UploadPdfResponse
from backend.memory.sqlite_store import init_db, save_message, get_session_history
from backend.memory.chroma_memory import memory
from backend.supervisor import classify_intent, ROUTES
from backend.agents.rag_qa import index_pdf

logger = get_logger(__name__)

app = FastAPI(title="Agentic Multi-Tool AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    warnings = settings.validate()
    for w in warnings:
        logger.warning(w)
    logger.info("Backend started.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload_pdf", response_model=UploadPdfResponse)
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks_indexed = index_pdf(session_id, str(tmp_path), file.filename)

        return UploadPdfResponse(
            session_id=session_id,
            filename=file.filename,
            chunks_indexed=chunks_indexed,
            message=f"Indexed {chunks_indexed} chunks from {file.filename}. "
            "You can now ask questions about this document.",
        )
    except ValueError as e:
        logger.warning("PDF indexing rejected: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("PDF upload/indexing failed")
        raise HTTPException(status_code=500, detail="Failed to process the PDF.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/history/{session_id}")
def history(session_id: str):
    try:
        items = get_session_history(session_id)
        return {"session_id": session_id, "history": [item.model_dump() for item in items]}
    except Exception:
        logger.exception("Failed to fetch history for session %s", session_id)
        raise HTTPException(status_code=500, detail="Failed to fetch session history.")


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Streams a Server-Sent-Events-style response. Each event is a JSON object:
        {"type": "intent", "intent": "research"}
        {"type": "token", "content": "..."}
        {"type": "done"}
        {"type": "error", "detail": "..."}
    """
    session_id = req.session_id or str(uuid.uuid4())
    message = req.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty.")

    save_message(session_id, "user", message)

    async def event_stream():
        full_response_chunks = []
        try:
            intent = req.intent_override or classify_intent(message)
            yield f"data: {json.dumps({'type': 'intent', 'intent': intent})}\n\n"

            memory_context = memory.get_relevant_context(session_id, message)
            generator_factory = ROUTES.get(intent, ROUTES["general_chat"])

            for chunk in generator_factory(session_id, message, memory_context):
                full_response_chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            final_response = "".join(full_response_chunks)
            save_message(session_id, "assistant", final_response, intent=intent)
            memory.add_exchange(session_id, message, final_response, intent)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.exception("Error while streaming chat response")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
