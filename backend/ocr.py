"""EasyOCR wrapper — box detection, reading-order, digit cleanup."""

from __future__ import annotations

import io
import logging
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from backend import config

if TYPE_CHECKING:
    import easyocr

logger = logging.getLogger(__name__)

_reader: easyocr.Reader | None = None
_reader_lock = threading.Lock()
_init_error: str | None = None

_DIGIT_SPAN = re.compile(r"(?<!\w)([0-9OIlSsZz\-]{6,})(?!\w)")
OCR_MIN_CONFIDENCE = 0.3


@dataclass
class OcrBox:
    id: int
    text: str
    confidence: float
    bbox: list[list[float]]
    cx: float
    cy: float


@dataclass
class ExtractTextResult:
    text: str
    order_path: str


def _use_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def init_ocr_reader() -> None:
    """Load EasyOCR models once at startup (blocking)."""
    global _reader, _init_error

    with _reader_lock:
        if _reader is not None or _init_error is not None:
            return

        try:
            import easyocr

            gpu = _use_gpu()
            logger.info("Loading EasyOCR models (langs=%s, gpu=%s)...", config.OCR_LANGS, gpu)
            _reader = easyocr.Reader(config.OCR_LANGS, gpu=gpu, verbose=False)
            logger.info("EasyOCR ready")
        except Exception as exc:
            _init_error = str(exc)
            logger.exception("EasyOCR init failed: %s", exc)


def is_ocr_ready() -> bool:
    return _reader is not None


def ocr_error() -> str | None:
    return _init_error


def _fix_digit_confusions(span: str) -> str:
    """Fix common OCR mistakes inside phone/account-like numeric runs."""
    chars = list(span)
    for i, ch in enumerate(chars):
        if ch in "Oo":
            chars[i] = "0"
        elif ch in "Il|":
            chars[i] = "1"
        elif ch == "S":
            chars[i] = "5"
        elif ch == "Z":
            chars[i] = "2"
    return "".join(chars)


def postprocess_digits(text: str) -> str:
    """Apply digit heuristics only on numeric spans; leave Thai text untouched."""

    def replacer(match: re.Match[str]) -> str:
        return _fix_digit_confusions(match.group(1))

    return _DIGIT_SPAN.sub(replacer, text)


def _bbox_center(
    bbox: list[list[float]], img_w: int, img_h: int
) -> tuple[float, float]:
    xs = [float(p[0]) for p in bbox]
    ys = [float(p[1]) for p in bbox]
    cx = ((min(xs) + max(xs)) / 2) / img_w
    cy = ((min(ys) + max(ys)) / 2) / img_h
    return cx, cy


def _decode_image(image_bytes: bytes) -> tuple[np.ndarray, int, int]:
    if _reader is None:
        raise RuntimeError("ระบบ OCR ยังไม่พร้อม")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
    except Exception as exc:
        raise ValueError("อ่านไฟล์รูปไม่ได้ กรุณาใช้ JPEG หรือ PNG") from exc

    img_w, img_h = image.size
    return np.array(image), img_w, img_h


def detect_boxes(image_bytes: bytes) -> list[OcrBox]:
    """Run EasyOCR detail=1 and return filtered boxes with normalized centers."""
    array, img_w, img_h = _decode_image(image_bytes)
    raw = _reader.readtext(array, detail=1, paragraph=False)

    boxes: list[OcrBox] = []
    next_id = 0
    for item in raw:
        if len(item) < 3:
            continue
        bbox, text, confidence = item[0], str(item[1]).strip(), float(item[2])
        if not text or confidence < OCR_MIN_CONFIDENCE:
            continue
        cx, cy = _bbox_center(bbox, img_w, img_h)
        boxes.append(
            OcrBox(
                id=next_id,
                text=text,
                confidence=confidence,
                bbox=[[float(p[0]), float(p[1])] for p in bbox],
                cx=cx,
                cy=cy,
            )
        )
        next_id += 1

    return boxes


def join_boxes(boxes: list[OcrBox]) -> str:
    lines = [b.text.strip() for b in boxes if b.text.strip()]
    return "\n".join(lines)


def extract_text(image_bytes: bytes) -> ExtractTextResult:
    """Detect boxes, order, and return final text with ordering path metadata."""
    from backend.ocr_order import order_and_join

    boxes = detect_boxes(image_bytes)

    if len(boxes) <= 1:
        text = postprocess_digits(join_boxes(boxes))
        return ExtractTextResult(text=text, order_path="single_box")

    text, order_path = order_and_join(boxes)
    return ExtractTextResult(text=text, order_path=order_path)
