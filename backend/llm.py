"""Ollama JSON verdict generation, validation, and political clamp."""

from __future__ import annotations

import logging
from typing import Any

from backend import config
from backend.ollama_chat import chat_json
from backend.prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(config.VERDICT_ENUM)},
        "confidence": {"type": "string", "enum": list(config.CONFIDENCE_ENUM)},
        "category": {"type": "string", "enum": list(config.CATEGORY_ENUM)},
        "summary_th": {"type": "string"},
        "reason_th": {"type": "string"},
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": list(config.HIGHLIGHT_TYPE_ENUM),
                    },
                    "note_th": {"type": "string"},
                    "signal_th": {"type": "string"},
                },
                "required": ["text", "type", "note_th", "signal_th"],
            },
        },
        "red_flags_th": {"type": "array", "items": {"type": "string"}},
        "advice_th": {"type": "string"},
        "source_ids": {
            "type": "array",
            "items": {"type": "string", "enum": list(config.SOURCE_ID_ENUM)},
        },
        "reply_polite_th": {"type": "string"},
        "reply_firm_th": {"type": "string"},
    },
    "required": [
        "verdict",
        "confidence",
        "category",
        "summary_th",
        "reason_th",
        "highlights",
        "red_flags_th",
        "advice_th",
        "source_ids",
        "reply_polite_th",
        "reply_firm_th",
    ],
}


def ping_ollama() -> bool:
    try:
        from backend.ollama_chat import client

        client().list()
        return True
    except Exception:
        return False


def _call_llm(message: str, evidence_records: list[dict[str, Any]] | None) -> dict[str, Any]:
    user_prompt = build_user_prompt(message, evidence_records)
    return chat_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        schema=VERDICT_JSON_SCHEMA,
    )


def _filter_highlights(message: str, highlights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for hl in highlights:
        text = hl.get("text", "")
        if text and text in message:
            valid.append(hl)
    return valid


def _filter_source_ids(source_ids: list[str]) -> list[str]:
    allowed = set(config.SOURCE_ID_ENUM)
    seen: set[str] = set()
    filtered: list[str] = []
    for sid in source_ids:
        if sid in allowed and sid not in seen:
            filtered.append(sid)
            seen.add(sid)
    for required in config.DEFAULT_SOURCE_IDS:
        if required not in seen:
            filtered.insert(0 if required == "antifake" else len(filtered), required)
            seen.add(required)
    return filtered


def validate_verdict(message: str, obj: dict[str, Any]) -> dict[str, Any]:
    for key in VERDICT_JSON_SCHEMA["required"]:
        if key not in obj:
            raise ValueError(f"missing required field: {key}")

    if obj["verdict"] not in config.VERDICT_ENUM:
        raise ValueError("invalid verdict")
    if obj["confidence"] not in config.CONFIDENCE_ENUM:
        raise ValueError("invalid confidence")
    if obj["category"] not in config.CATEGORY_ENUM:
        raise ValueError("invalid category")

    highlights = obj.get("highlights") or []
    if not isinstance(highlights, list):
        raise ValueError("highlights must be a list")

    red_flags = obj.get("red_flags_th") or []
    if not isinstance(red_flags, list):
        raise ValueError("red_flags_th must be a list")

    source_ids = obj.get("source_ids") or []
    if not isinstance(source_ids, list):
        raise ValueError("source_ids must be a list")

    cleaned = {
        "verdict": obj["verdict"],
        "confidence": obj["confidence"],
        "category": obj["category"],
        "summary_th": str(obj["summary_th"]),
        "reason_th": str(obj["reason_th"]),
        "highlights": _filter_highlights(message, highlights),
        "red_flags_th": [str(x) for x in red_flags],
        "advice_th": str(obj["advice_th"]),
        "source_ids": _filter_source_ids([str(x) for x in source_ids]),
        "reply_polite_th": str(obj["reply_polite_th"]).strip(),
        "reply_firm_th": str(obj["reply_firm_th"]).strip(),
    }
    return cleaned


def apply_political_clamp(message: str, verdict_obj: dict[str, Any]) -> dict[str, Any]:
    lowered = message.lower()
    if not any(kw in message or kw in lowered for kw in config.POLITICAL_KEYWORDS):
        return verdict_obj

    clamped = dict(verdict_obj)
    if clamped["verdict"] in ("fake", "credible"):
        clamped["verdict"] = "unverified"
    if clamped["confidence"] == "high":
        clamped["confidence"] = "medium"

    clamped["advice_th"] = (
        "เนื้อหาทางการเมืองควรตรวจสอบจากแหล่งข่าวทางการและสื่อที่น่าเชื่อถือ "
        "อย่าแชร์ต่อหากยังไม่ยืนยันความจริง"
    )
    return clamped


def generate_verdict(
    message: str, evidence_records: list[dict[str, Any]] | None
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = _call_llm(message, evidence_records)
            validated = validate_verdict(message, raw)
            return apply_political_clamp(message, validated)
        except Exception as exc:
            last_error = exc
            logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)

    raise RuntimeError(
        "ไม่สามารถวิเคราะห์ข้อความได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
    ) from last_error
