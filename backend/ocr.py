"""EasyOCR wrapper — box detection, reading-order, digit cleanup."""

from __future__ import annotations

import io
import logging
import re
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from backend import config
from backend.ocr_preprocess import (
    make_thumbnail_bytes,
    maybe_invert_for_ocr,
    to_base64,
)

if TYPE_CHECKING:
    import easyocr

logger = logging.getLogger(__name__)

_reader: easyocr.Reader | None = None
_reader_lock = threading.Lock()
_pipeline_lock = threading.Lock()
_init_error: str | None = None

_DIGIT_SPAN = re.compile(r"(?<!\w)([0-9OIlSsZz\-]{6,})(?!\w)")
OCR_MIN_CONFIDENCE = 0.3


@contextmanager
def _easyocr_quiet():
    """EasyOCR uses pin_memory=True even on CPU — harmless, but noisy in logs."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*pin_memory.*no accelerator.*",
            category=UserWarning,
        )
        yield


@dataclass
class OcrBox:
    id: int
    text: str
    confidence: float
    bbox: list[list[float]]
    cx: float
    cy: float


@dataclass
class DetectResult:
    boxes: list[OcrBox]
    dropped_boxes: int


@dataclass
class ExtractTextResult:
    text: str
    order_path: str
    box_count: int
    dropped_boxes: int


def _use_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _ensure_reader_loaded() -> None:
    """Load EasyOCR reader if not present (reload after VRAM release)."""
    global _reader, _init_error

    if _reader is not None:
        return
    if _init_error is not None:
        raise RuntimeError("ระบบ OCR ยังไม่พร้อม")

    with _reader_lock:
        if _reader is not None:
            return
        if _init_error is not None:
            raise RuntimeError("ระบบ OCR ยังไม่พร้อม")

        try:
            import easyocr

            gpu = _use_gpu()
            logger.info(
                "Loading EasyOCR models (langs=%s, gpu=%s)...",
                config.OCR_LANGS,
                gpu,
            )
            with _easyocr_quiet():
                _reader = easyocr.Reader(config.OCR_LANGS, gpu=gpu, verbose=False)
            logger.info("EasyOCR ready")
        except Exception as exc:
            _init_error = str(exc)
            logger.exception("EasyOCR init failed: %s", exc)
            raise RuntimeError("ระบบ OCR ยังไม่พร้อม") from exc


def init_ocr_reader() -> None:
    """Load EasyOCR models once at startup (blocking)."""
    try:
        _ensure_reader_loaded()
    except RuntimeError:
        pass


def is_ocr_ready() -> bool:
    """True unless EasyOCR failed to init; reader may be unloaded between requests."""
    return _init_error is None


def ocr_error() -> str | None:
    return _init_error


def release_ocr_vram() -> None:
    """Drop EasyOCR models from GPU before multimodal Gemma."""
    global _reader

    with _reader_lock:
        if _reader is None:
            return
        logger.info("Releasing EasyOCR VRAM")
        del _reader
        _reader = None

    if _use_gpu():
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("torch.cuda.empty_cache failed: %s", exc)


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


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode upload bytes to RGB numpy array."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
    except Exception as exc:
        raise ValueError("อ่านไฟล์รูปไม่ได้ กรุณาใช้ JPEG หรือ PNG") from exc
    return np.array(image)


def detect_boxes(rgb_array: np.ndarray) -> DetectResult:
    """Run EasyOCR detail=1 and return filtered boxes with normalized centers."""
    _ensure_reader_loaded()

    img_h, img_w = rgb_array.shape[:2]
    with _easyocr_quiet():
        raw = _reader.readtext(rgb_array, detail=1, paragraph=False)

    boxes: list[OcrBox] = []
    dropped = 0
    next_id = 0
    for item in raw:
        if len(item) < 3:
            continue
        bbox, text, confidence = item[0], str(item[1]).strip(), float(item[2])
        if not text:
            continue
        if confidence < OCR_MIN_CONFIDENCE:
            dropped += 1
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

    if dropped:
        logger.info("OCR dropped %d low-confidence boxes", dropped)

    return DetectResult(boxes=boxes, dropped_boxes=dropped)


def join_boxes(boxes: list[OcrBox]) -> str:
    lines = [b.text.strip() for b in boxes if b.text.strip()]
    return "\n".join(lines)


def _extract_text_unlocked(image_bytes: bytes) -> ExtractTextResult:
    from backend.ocr_order import order_and_join

    original = decode_image_bytes(image_bytes)
    ocr_array, inverted = maybe_invert_for_ocr(original)
    if inverted:
        logger.info("Applied dark-background invert before OCR")

    thumbnail_b64 = to_base64(make_thumbnail_bytes(original))
    detect = detect_boxes(ocr_array)
    boxes = detect.boxes
    dropped = detect.dropped_boxes

    release_ocr_vram()

    if len(boxes) <= 1:
        text = postprocess_digits(join_boxes(boxes))
        return ExtractTextResult(
            text=text,
            order_path="single_box",
            box_count=len(boxes),
            dropped_boxes=dropped,
        )

    text, order_path = order_and_join(boxes, thumbnail_b64)
    return ExtractTextResult(
        text=text,
        order_path=order_path,
        box_count=len(boxes),
        dropped_boxes=dropped,
    )


def extract_text(image_bytes: bytes) -> ExtractTextResult:
    """Full pipeline: preprocess → OCR → VRAM release → order → text."""
    with _pipeline_lock:
        return _extract_text_unlocked(image_bytes)
