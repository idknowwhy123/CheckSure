"""Per-source stance extraction for evidence grounding."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from backend import config
from backend.ollama_chat import chat_json

logger = logging.getLogger(__name__)

STANCE_ENUM = ("refutes", "supports", "unrelated", "unclear")

STANCE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": list(STANCE_ENUM)},
        "conclusion_th": {"type": "string"},
        "claim_matched": {"type": "boolean"},
    },
    "required": ["stance", "conclusion_th", "claim_matched"],
}

STANCE_PROMPT = """คุณวิเคราะห์ว่าแหล่งข้อมูลหนึ่งพูดถึงข้อความที่ผู้ใช้สงสัยอย่างไร

กฎ:
- อ่านเนื้อหาแหล่ง (content_full) เป็นหลัก ไม่ใช่แค่หัวข้อ
- ใช้ "refutes" เมื่อแหล่งชัดเจนว่าข้อความ/ข่าวนี้ไม่จริง เป็นข่าวปลอม หรือหลอกลวง
- ใช้ "supports" เมื่อแหล่งชัดเจนว่าข้อความนี้ถูกต้องหรือน่าเชื่อถือ
- ใช้ "unrelated" เมื่อแหล่งพูดเรื่องอื่น หรือ fact-check คนละข่าว/คนละข้ออ้าง
- ใช้ "unclear" เมื่ออ่านแล้วยังสรุปไม่ได้
- ตั้ง claim_matched=true เฉพาะเมื่อแหล่งนี้พูดถึงข้ออ้างเดียวกับข้อความผู้ใช้โดยตรง
- conclusion_th: หนึ่งประโยคภาษาไทย สรุปว่าแหล่งนี้พูดอะไรเกี่ยวกับข้ออ้างนี้

ตอบเป็น JSON object เดียวเท่านั้น ครบทุกฟิลด์: stance, conclusion_th, claim_matched"""


def is_authoritative_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return False
    return any(domain in host for domain in config.AUTHORITATIVE_DOMAINS)


def _fail_safe(hit: dict[str, Any]) -> None:
    hit["stance"] = "unclear"
    hit["conclusion_th"] = ""
    hit["claim_matched"] = False
    hit["authoritative"] = is_authoritative_url(str(hit.get("source_url") or ""))


def _classify_one(message: str, hit: dict[str, Any]) -> None:
    source_url = str(hit.get("source_url") or "")
    hit["authoritative"] = is_authoritative_url(source_url)

    title = str(hit.get("title") or hit.get("source") or "")
    content = str(hit.get("content_full") or hit.get("text") or "")

    user_prompt = f"""ข้อความที่ผู้ใช้สงสัย:
{message}

แหล่งข้อมูล:
หัวข้อ: {title}
URL: {source_url}
เนื้อหา:
{content}"""

    try:
        result = chat_json(
            [
                {"role": "system", "content": STANCE_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            schema=STANCE_JSON_SCHEMA,
        )
        stance = result.get("stance", "unclear")
        if stance not in STANCE_ENUM:
            stance = "unclear"
        hit["stance"] = stance
        hit["conclusion_th"] = str(result.get("conclusion_th") or "")
        hit["claim_matched"] = bool(result.get("claim_matched"))
    except Exception as exc:
        logger.warning("stance classification failed for %s: %s", source_url, exc)
        _fail_safe(hit)


def classify_stances(message: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not config.STANCE_ENABLED or not hits:
        return hits

    out = [dict(h) for h in hits]
    for hit in out[: config.STANCE_MAX_SOURCES]:
        _classify_one(message, hit)
    return out
