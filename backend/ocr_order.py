"""Reading-order for EasyOCR boxes — geometric fallback + local LLM (same gemma3)."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from statistics import median

from backend import config
from backend.ocr import OcrBox, join_boxes, postprocess_digits
from backend.ollama_chat import chat_json

logger = logging.getLogger(__name__)

ORDER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "order": {
            "type": "array",
            "items": {"type": "integer"},
        }
    },
    "required": ["order"],
}

SYSTEM_PROMPT = """คุณช่วยจัดลำดับการอ่านข้อความจากภาพแชต (เช่น LINE)
ผู้ใช้ส่งรายการกล่องข้อความ OCR แต่ละกล่องมี id, text, cx, cy (พิกัดศูนย์กลางแบบ 0–1)

กติกา:
- ตอบเป็น JSON เท่านั้น: {"order": [id, id, ...]}
- order ต้องมี id ครบทุกตัวที่ได้รับ ไม่ซ้ำ ไม่ขาด
- เรียงตามลำดับการอ่านธรรมชาติ: บนลงล่าง คอลัมน์ซ้ายก่อนขวา ข้อความหลักก่อนเวลา/สถานะเมื่อไม่แน่ใจ
- ห้ามแก้ไขหรือรวมข้อความใน text — จัดลำดับ id เท่านั้น"""


def _box_height(box: OcrBox) -> float:
    ys = [p[1] for p in box.bbox]
    return max(ys) - min(ys)


def _cx_px(box: OcrBox) -> float:
    xs = [p[0] for p in box.bbox]
    return (min(xs) + max(xs)) / 2


def _cy_px(box: OcrBox) -> float:
    ys = [p[1] for p in box.bbox]
    return (min(ys) + max(ys)) / 2


def geometric_order(boxes: list[OcrBox]) -> list[OcrBox]:
    """Top-to-bottom, left-to-right with line grouping."""
    if len(boxes) <= 1:
        return list(boxes)

    heights = [_box_height(b) for b in boxes]
    med_h = median(heights) if heights else 1.0
    line_threshold = max(med_h * 0.6, 1.0)

    sorted_by_y = sorted(boxes, key=lambda b: (_cy_px(b), _cx_px(b)))
    lines: list[list[OcrBox]] = []
    current_line: list[OcrBox] = []

    for box in sorted_by_y:
        if not current_line:
            current_line = [box]
            continue
        ref_cy = median(_cy_px(b) for b in current_line)
        if abs(_cy_px(box) - ref_cy) <= line_threshold:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]

    if current_line:
        lines.append(current_line)

    ordered: list[OcrBox] = []
    for line in lines:
        ordered.extend(sorted(line, key=_cx_px))
    return ordered


def _validate_order(order: object, n: int) -> list[int] | None:
    if not isinstance(order, list) or len(order) != n:
        return None
    try:
        ids = [int(x) for x in order]
    except (TypeError, ValueError):
        return None
    if sorted(ids) != list(range(n)):
        return None
    return ids


def _call_order_llm(boxes: list[OcrBox]) -> list[int]:
    manifest = [
        {
            "id": b.id,
            "text": b.text,
            "cx": round(b.cx, 3),
            "cy": round(b.cy, 3),
        }
        for b in boxes
    ]
    user_prompt = (
        "จัดลำดับ id ของกล่องข้อความต่อไปนี้:\n\n"
        f"{json.dumps(manifest, ensure_ascii=False)}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    result = chat_json(messages, temperature=0.1, schema=ORDER_JSON_SCHEMA)
    validated = _validate_order(result.get("order"), len(boxes))
    if validated is None:
        raise ValueError("invalid order permutation from LLM")
    return validated


def order_boxes_llm(boxes: list[OcrBox]) -> list[OcrBox]:
    """LLM reading order with timeout; raises on failure."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call_order_llm, boxes)
        try:
            order = future.result(timeout=config.OCR_ORDER_TIMEOUT_SEC)
        except FuturesTimeoutError as exc:
            raise TimeoutError("OCR order LLM timeout") from exc

    return [boxes[i] for i in order]


def order_and_join(boxes: list[OcrBox]) -> tuple[str, str]:
    """
    Order boxes and join to final text.
    Returns (text, order_path) where order_path is single_box | llm | geometric.
    """
    if len(boxes) <= 1:
        text = join_boxes(boxes)
        return postprocess_digits(text), "single_box"

    use_llm = (
        config.OCR_ORDER_ENABLED
        and len(boxes) <= config.OCR_MAX_BOXES
    )

    if use_llm:
        try:
            ordered = order_boxes_llm(boxes)
            path = "llm"
        except Exception as exc:
            logger.warning("OCR LLM ordering failed (%s), using geometric", exc)
            ordered = geometric_order(boxes)
            path = "geometric"
    else:
        ordered = geometric_order(boxes)
        path = "geometric"

    text = join_boxes(ordered)
    return postprocess_digits(text), path
