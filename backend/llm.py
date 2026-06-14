"""Ollama JSON verdict generation, validation, and political clamp."""

from __future__ import annotations

import logging
from typing import Any

from backend import config
from backend.ollama_chat import chat_json
from backend.prompt import build_user_prompt, get_system_prompt

logger = logging.getLogger(__name__)

_VERDICT_BASE_PROPERTIES: dict[str, Any] = {
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
}

_VERDICT_BASE_REQUIRED = [
    "verdict",
    "confidence",
    "category",
    "summary_th",
    "reason_th",
    "highlights",
    "red_flags_th",
    "advice_th",
]


def get_verdict_json_schema() -> dict[str, Any]:
    properties = dict(_VERDICT_BASE_PROPERTIES)
    required = list(_VERDICT_BASE_REQUIRED)
    if config.REPLY_SUGGESTIONS_ENABLED:
        properties["reply_polite_th"] = {"type": "string"}
        properties["reply_firm_th"] = {"type": "string"}
        required.extend(["reply_polite_th", "reply_firm_th"])
    return {"type": "object", "properties": properties, "required": required}


VERDICT_JSON_SCHEMA = get_verdict_json_schema()


def ping_ollama() -> bool:
    try:
        from backend.ollama_chat import client

        client().list()
        return True
    except Exception:
        return False


def _call_llm(
    message: str,
    evidence_records: list[dict[str, Any]] | None,
    *,
    temperature: float = 0.3,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(message, evidence_records)
    return chat_json(
        [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        schema=get_verdict_json_schema(),
    )


def _filter_highlights(message: str, highlights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for hl in highlights:
        text = hl.get("text", "")
        if text and text in message:
            valid.append(hl)
    return valid


def derive_source_ids(category: str, message: str) -> list[str]:
    """Deterministic source_ids from category + message keywords."""
    ids: list[str] = list(config.DEFAULT_SOURCE_IDS)
    seen = set(ids)
    lowered = message.lower()
    extras: list[str] = []

    if category == "health":
        extras.append("fda")
        if any(kw in message or kw in lowered for kw in ("แพทย์", "รักษา", "ยา", "โรค")):
            extras.append("doctor")
    elif category == "scam":
        if any(
            kw in message or kw in lowered
            for kw in ("หุ้น", "ลงทุน", "กองทุน", "ก.ล.ต.")
        ):
            extras.append("sec")
        if any(
            kw in message or kw in lowered
            for kw in ("otp", "ลิงก์", "โอน", "บัญชี", "ธนาคาร")
        ):
            extras.extend(["aoc", "bot"])
    elif category == "official":
        extras.extend(["gov", "aoc"])
    else:
        extras.append("gov")

    for sid in extras:
        if sid not in config.SOURCE_ID_ENUM or sid in seen:
            continue
        ids.append(sid)
        seen.add(sid)
        if len(ids) >= 4:
            break
    return ids[:4]


def validate_verdict(message: str, obj: dict[str, Any]) -> dict[str, Any]:
    schema = get_verdict_json_schema()
    for key in schema["required"]:
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

    category = obj["category"]
    cleaned: dict[str, Any] = {
        "verdict": obj["verdict"],
        "confidence": obj["confidence"],
        "category": category,
        "summary_th": str(obj["summary_th"]),
        "reason_th": str(obj["reason_th"]),
        "highlights": _filter_highlights(message, highlights),
        "red_flags_th": [str(x) for x in red_flags],
        "advice_th": str(obj["advice_th"]),
        "source_ids": derive_source_ids(category, message),
    }
    if config.REPLY_SUGGESTIONS_ENABLED:
        cleaned["reply_polite_th"] = str(obj["reply_polite_th"]).strip()
        cleaned["reply_firm_th"] = str(obj["reply_firm_th"]).strip()
    return cleaned


def apply_evidence_clamp(
    message: str,
    verdict_obj: dict[str, Any],
    hits: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not config.STANCE_ENABLED or not hits:
        return verdict_obj

    matched = [
        h for h in hits if h.get("claim_matched") and h.get("authoritative")
    ]
    if not matched:
        return verdict_obj

    refuting = [h for h in matched if h.get("stance") == "refutes"]
    supporting = [h for h in matched if h.get("stance") == "supports"]

    clamped = dict(verdict_obj)

    if refuting and not supporting:
        if clamped["verdict"] != "fake":
            clamped["verdict"] = "fake"
        clamped["confidence"] = "high" if len(refuting) >= 2 else "medium"
        top = refuting[0]
        if top.get("conclusion_th"):
            clamped["reason_th"] = str(top["conclusion_th"])
        return clamped

    if refuting and supporting:
        if clamped["confidence"] == "high":
            clamped["confidence"] = "medium"
        return clamped

    return verdict_obj


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
            raw = _call_llm(
                message,
                evidence_records,
                temperature=0.2 if attempt > 0 else 0.3,
            )
            validated = validate_verdict(message, raw)
            validated = apply_evidence_clamp(message, validated, evidence_records)
            return apply_political_clamp(message, validated)
        except Exception as exc:
            last_error = exc
            logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)

    raise RuntimeError(
        "ไม่สามารถวิเคราะห์ข้อความได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
    ) from last_error
