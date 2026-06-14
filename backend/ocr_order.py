"""Reading-order for EasyOCR boxes — layout heuristics + multimodal fallback."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from statistics import median

from backend import config
from backend.ocr import OcrBox, join_boxes, postprocess_digits
from backend.ollama_chat import chat_json_with_image

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

SYSTEM_PROMPT = """คุณช่วยจัดลำดับการอ่านข้อความจากภาพอินโฟกราฟิก (หลายคอลัมน์ หลายแผง)
ผู้ใช้ส่งรายการกล่องข้อความ OCR แต่ละกล่องมี id, text, cx, cy (พิกัดศูนย์กลางแบบ 0–1)
และส่งภาพย่อของเลย์เอาต์มาด้วย — ใช้ภาพเพื่อเข้าใจโครงสร้างเท่านั้น

กติกา:
- ตอบเป็น JSON เท่านั้น: {"order": [id, id, ...]}
- id เป็นจำนวนเต็มเริ่มจาก 0 (กล่องแรกคือ id 0 กล่องสุดท้ายคือ id N-1)
- order ต้องมี id ครบทุกตัวที่ได้รับ ไม่ซ้ำ ไม่ขาด ความยาว order ต้องเท่ากับจำนวนกล่องพอดี
- อินโฟกราฟิกหลายคอลัมน์: อ่านลงล่างทีละคอลัมน์ คอลัมน์ซ้ายก่อนขวา
- อินโฟกราฟิกหลายแผง (1 2 3 4): อ่านทีละแผงตามลำดับหมายเลขหรือตำแหน่งบน→ล่าง ซ้าย→ขวา
- ห้ามแก้ไข รวม หรือถอดความข้อความใน text — จัดลำดับ id เท่านั้น
- ห้ามอ่านข้อความใหม่จากภาพ — ใช้ภาพเพื่อจัดลำดับเท่านั้น"""

_PANEL_ANCHOR_RE = re.compile(r"^(\d+)\.")
_COLUMN_GAP_FACTOR = 1.5


def _box_height(box: OcrBox) -> float:
    ys = [p[1] for p in box.bbox]
    return max(ys) - min(ys)


def _cx_px(box: OcrBox) -> float:
    xs = [p[0] for p in box.bbox]
    return (min(xs) + max(xs)) / 2


def _cy_px(box: OcrBox) -> float:
    ys = [p[1] for p in box.bbox]
    return (min(ys) + max(ys)) / 2


def _box_width(box: OcrBox) -> float:
    xs = [p[0] for p in box.bbox]
    return max(xs) - min(xs)


def _centroid_dist_sq(a: OcrBox, b: OcrBox) -> float:
    dx = _cx_px(a) - _cx_px(b)
    dy = _cy_px(a) - _cy_px(b)
    return dx * dx + dy * dy


def _find_panel_anchors(boxes: list[OcrBox]) -> dict[int, OcrBox]:
    anchors: dict[int, OcrBox] = {}
    for box in boxes:
        match = _PANEL_ANCHOR_RE.match(box.text.strip())
        if not match:
            continue
        num = int(match.group(1))
        if num not in anchors:
            anchors[num] = box
    return anchors


def _anchors_are_column_headers(anchors: dict[int, OcrBox]) -> bool:
    """True when numbered headers share one row (multi-column, not 2×2 panel grid)."""
    if len(anchors) < 2:
        return False
    cys = [_cy_px(anchors[num]) for num in anchors]
    cy_spread = max(cys) - min(cys)
    heights = [_box_height(anchors[num]) for num in anchors]
    med_h = median(heights) if heights else 1.0
    return cy_spread <= med_h * 1.5


def numbered_panel_order(boxes: list[OcrBox]) -> list[OcrBox] | None:
    """Order by numbered panel headers (1. 2. 3. …) when ≥2 anchors exist."""
    anchors = _find_panel_anchors(boxes)
    if len(anchors) < 2:
        return None
    if _anchors_are_column_headers(anchors):
        return None

    anchor_nums = sorted(anchors.keys())
    anchor_ids = {anchors[num].id for num in anchor_nums}
    panels: dict[int, list[OcrBox]] = {num: [anchors[num]] for num in anchor_nums}

    min_anchor_cy = min(_cy_px(anchors[num]) for num in anchor_nums)
    max_anchor_cy = max(_cy_px(anchors[num]) for num in anchor_nums)
    heights = [_box_height(b) for b in boxes]
    line_threshold = max(median(heights) if heights else 1.0, 1.0) * 0.8

    header: list[OcrBox] = []
    footer: list[OcrBox] = []
    for box in boxes:
        if box.id in anchor_ids:
            continue
        cy = _cy_px(box)
        if cy < min_anchor_cy - line_threshold:
            header.append(box)
        elif cy > max_anchor_cy + line_threshold * 2:
            footer.append(box)
        else:
            nearest = min(anchor_nums, key=lambda n: _centroid_dist_sq(box, anchors[n]))
            panels[nearest].append(box)

    ordered: list[OcrBox] = []
    ordered.extend(sorted(header, key=_cy_px))
    for num in anchor_nums:
        ordered.extend(sorted(panels[num], key=_cy_px))
    ordered.extend(sorted(footer, key=_cy_px))

    if len(ordered) != len(boxes):
        logger.warning("numbered_panel_order lost boxes; declining")
        return None
    return ordered


def _split_columns_once(
    group: list[OcrBox],
    gap_threshold: float,
) -> list[list[OcrBox]]:
    if len(group) <= 1:
        return [group]

    sorted_group = sorted(group, key=_cx_px)
    best_gap = 0.0
    best_idx = -1
    for idx in range(len(sorted_group) - 1):
        gap = _cx_px(sorted_group[idx + 1]) - _cx_px(sorted_group[idx])
        if gap > best_gap:
            best_gap = gap
            best_idx = idx

    if best_gap < gap_threshold or best_idx < 0:
        return [group]

    left = sorted_group[: best_idx + 1]
    right = sorted_group[best_idx + 1 :]
    return _split_columns_once(left, gap_threshold) + _split_columns_once(
        right, gap_threshold
    )


def column_cluster_order(boxes: list[OcrBox]) -> list[OcrBox] | None:
    """Order by x-gap column clustering; declines when layout is single-column."""
    if len(boxes) <= 1:
        return None

    widths = [_box_width(b) for b in boxes]
    gap_threshold = max(median(widths) if widths else 1.0, 1.0) * _COLUMN_GAP_FACTOR
    columns = _split_columns_once(boxes, gap_threshold)
    if len(columns) <= 1:
        return None

    ordered: list[OcrBox] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda b: (_cy_px(b), _cx_px(b))))
    return ordered


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


def _repair_order(
    order: object,
    n: int,
    boxes: list[OcrBox],
) -> list[int] | None:
    """
    Salvage near-valid permutations: 1-based ids, duplicates, missing ids.
    Missing ids are appended in geometric reading order.
    """
    if not isinstance(order, list) or not order:
        return None

    try:
        raw_ids = [int(x) for x in order]
    except (TypeError, ValueError):
        return None

    # Model sometimes returns 1..N instead of 0..N-1.
    if (
        len(raw_ids) == n
        and len(set(raw_ids)) == n
        and all(1 <= i <= n for i in raw_ids)
    ):
        return [i - 1 for i in raw_ids]

    seen: set[int] = set()
    deduped: list[int] = []
    for i in raw_ids:
        if 0 <= i < n and i not in seen:
            seen.add(i)
            deduped.append(i)

    missing = [i for i in range(n) if i not in seen]
    if missing:
        missing_boxes = [boxes[i] for i in missing]
        for box in geometric_order(missing_boxes):
            deduped.append(box.id)

    if len(deduped) != n:
        return None
    if sorted(deduped) != list(range(n)):
        return None
    return deduped


def _normalize_order(
    order: object,
    boxes: list[OcrBox],
) -> list[int] | None:
    n = len(boxes)
    strict = _validate_order(order, n)
    if strict is not None:
        return strict
    repaired = _repair_order(order, n, boxes)
    if repaired is not None:
        logger.info("Repaired multimodal order permutation (n=%d)", n)
    return repaired


def _call_order_multimodal(boxes: list[OcrBox], thumbnail_b64: str) -> list[int]:
    n = len(boxes)
    manifest = [
        {
            "id": b.id,
            "text": b.text[:120],
            "cx": round(b.cx, 3),
            "cy": round(b.cy, 3),
        }
        for b in boxes
    ]
    user_prompt = (
        f"มีกล่องข้อความ {n} กล่อง (id 0 ถึง {n - 1}) "
        "จัดลำดับ id ตามลำดับการอ่านของอินโฟกราฟิก:\n\n"
        f"{json.dumps(manifest, ensure_ascii=False)}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    result = chat_json_with_image(
        messages,
        thumbnail_b64,
        temperature=0.0,
        schema=ORDER_JSON_SCHEMA,
    )
    validated = _normalize_order(result.get("order"), boxes)
    if validated is None:
        logger.warning(
            "Invalid multimodal order (n=%d, got=%s)",
            n,
            result.get("order"),
        )
        raise ValueError("invalid order permutation from multimodal LLM")
    return validated


def order_boxes_multimodal(boxes: list[OcrBox], thumbnail_b64: str) -> list[OcrBox]:
    """Multimodal reading order with timeout; raises on failure."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call_order_multimodal, boxes, thumbnail_b64)
        try:
            order = future.result(timeout=config.OCR_ORDER_TIMEOUT_SEC)
        except FuturesTimeoutError as exc:
            raise TimeoutError("OCR multimodal order timeout") from exc

    return [boxes[i] for i in order]


def order_and_join(
    boxes: list[OcrBox],
    thumbnail_b64: str | None = None,
) -> tuple[str, str]:
    """
    Order boxes and join to final text.

    Cascade: numbered panels → column clusters → multimodal Gemma → geometric.
    Returns (text, order_path).
    """
    if len(boxes) <= 1:
        text = join_boxes(boxes)
        return postprocess_digits(text), "single_box"

    ordered: list[OcrBox] | None = None
    path = "geometric"

    panel_ordered = numbered_panel_order(boxes)
    if panel_ordered is not None:
        ordered = panel_ordered
        path = "panel_numbered"
    else:
        column_ordered = column_cluster_order(boxes)
        if column_ordered is not None:
            ordered = column_ordered
            path = "column"
        else:
            use_multimodal = (
                config.OCR_ORDER_ENABLED
                and len(boxes) <= config.OCR_MAX_BOXES
                and thumbnail_b64 is not None
            )
            if use_multimodal:
                try:
                    ordered = order_boxes_multimodal(boxes, thumbnail_b64)
                    path = "multimodal"
                except Exception as exc:
                    logger.warning(
                        "OCR multimodal ordering failed (%s), using geometric", exc
                    )
                    ordered = geometric_order(boxes)
                    path = "geometric"
            else:
                ordered = geometric_order(boxes)
                path = "geometric"

    text = join_boxes(ordered)
    return postprocess_digits(text), path
