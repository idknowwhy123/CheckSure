"""FastAPI entry: /health, /check, /ocr, serve index.html."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import config
from backend.llm import ping_ollama
from backend.ocr import extract_text, init_ocr_reader, is_ocr_ready, ocr_error
from backend.pipeline import run_check
from backend.websearch import debug_search_for_message

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Loading EasyOCR models…")
    await asyncio.to_thread(init_ocr_reader)
    yield


app = FastAPI(title="CheckSure v3", version="3.0.0", lifespan=lifespan)


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1)


class DebugSearchRequest(BaseModel):
    text: str = Field(..., min_length=1)
    extract_only: bool = False


class OcrResponse(BaseModel):
    text: str
    order_path: str
    box_count: int
    dropped_boxes: int


@app.get("/health")
def health() -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ok",
        "ollama": ping_ollama(),
        "tavily_configured": bool(config.TAVILY_API_KEY),
        "ocr_ready": is_ocr_ready(),
        "reply_suggestions_enabled": config.REPLY_SUGGESTIONS_ENABLED,
    }
    err = ocr_error()
    if err:
        payload["ocr_error"] = err
    return payload


@app.get("/")
def index() -> FileResponse:
    path = Path(config.INDEX_HTML_PATH)
    if not path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(path)


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> OcrResponse:
    if not is_ocr_ready():
        raise HTTPException(
            status_code=503,
            detail="ระบบ OCR ยังไม่พร้อม กรุณาวางข้อความเอง",
        )

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in config.OCR_ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="รองรับเฉพาะไฟล์ JPEG หรือ PNG",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="ไฟล์ว่างเปล่า")
    if len(data) > config.OCR_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail="ไฟล์ใหญ่เกินไป (สูงสุด 5 MB)",
        )

    try:
        result = await asyncio.to_thread(extract_text, data)
        logger.info(
            "ocr_order=%s box_count=%d dropped=%d",
            result.order_path,
            result.box_count,
            result.dropped_boxes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="อ่านข้อความจากรูปไม่สำเร็จ กรุณาลองใหม่",
        ) from exc

    return OcrResponse(
        text=result.text,
        order_path=result.order_path,
        box_count=result.box_count,
        dropped_boxes=result.dropped_boxes,
    )


@app.post("/check")
def check(body: CheckRequest) -> dict:
    message = body.text.strip()
    if not message:
        raise HTTPException(status_code=400, detail="กรุณาวางข้อความที่ต้องการตรวจสอบ")

    try:
        return run_check(message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="เกิดข้อผิดพลาดภายในระบบ กรุณาลองใหม่อีกครั้ง",
        ) from exc


@app.post("/debug/search")
def debug_search(body: DebugSearchRequest) -> dict:
    """Inspect claim extraction and Tavily responses (dev tooling)."""
    if not config.DEBUG_SEARCH_ENABLED:
        raise HTTPException(status_code=404, detail="Debug search is disabled")

    message = body.text.strip()
    if not message:
        raise HTTPException(status_code=400, detail="กรุณาวางข้อความที่ต้องการทดสอบ")

    try:
        return debug_search_for_message(message, extract_only=body.extract_only)
    except Exception as exc:
        logger.exception("debug search failed")
        raise HTTPException(
            status_code=500,
            detail=f"debug search failed: {exc}",
        ) from exc
