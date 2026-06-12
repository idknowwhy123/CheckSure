"""Web search grounding via Tavily — restricted to Thai fact-check domains."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

from tavily import TavilyClient

from backend import config
from backend.ollama_chat import chat_json

logger = logging.getLogger(__name__)

_tavily: TavilyClient | None = None

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["queries"],
}

EXTRACT_PROMPT = """อ่านข้อความที่ผู้ใช้ได้รับแล้วสกัดคำค้นหาสั้น ๆ ภาษาไทย 3-5 ข้อ
เพื่อค้นหาว่าข้ออ้างนี้เคยถูกตรวจสอบข่าวปลอมหรือไม่

กฎ:
- แต่ละ query สั้น ไม่เกิน 80 ตัวอักษร
- เน้นข้ออ้างสำคัญ ไม่ใส่คำชวนแชร์หรือคำหยาบ
- ตอบเป็น JSON เท่านั้น: {"queries": ["...", "..."]}"""


def _tavily_client() -> TavilyClient | None:
    global _tavily
    if not config.TAVILY_API_KEY:
        return None
    if _tavily is None:
        _tavily = TavilyClient(api_key=config.TAVILY_API_KEY)
    return _tavily


def _heuristic_queries(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    snippet = cleaned[:120]
    return [snippet]


def extract_claims(text: str) -> list[str]:
    """Turn message into 3-5 short Thai search queries. Privacy: only these leave the machine."""
    message = text.strip()
    if not message:
        return []

    try:
        parsed = chat_json(
            [
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.1,
            schema=CLAIM_SCHEMA,
        )
        queries = parsed.get("queries") or []
        if isinstance(queries, list):
            cleaned = [str(q).strip() for q in queries if str(q).strip()][:3]
            if cleaned:
                logger.info("[websearch] extracted queries: %s", cleaned)
                return cleaned
    except Exception as exc:
        logger.warning("[websearch] claim extraction failed: %s", exc)

    fallback = _heuristic_queries(message)
    logger.info("[websearch] heuristic queries: %s", fallback)
    return fallback


def _domain_label(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return "fact-check"
    for domain, label in config.DOMAIN_LABELS.items():
        if domain in host:
            return label
    return host or "fact-check"


def _normalize_hit(hit: dict[str, Any]) -> dict[str, Any] | None:
    url = str(hit.get("url") or "").strip()
    snippet = str(hit.get("content") or hit.get("raw_content") or "").strip()
    if not url or not snippet:
        return None
    return {
        "text": snippet[:500],
        "source": _domain_label(url),
        "source_url": url,
        "reason_th": snippet[:200],
    }


def search(queries: list[str]) -> list[dict[str, Any]]:
    """Search Tavily on allow-listed domains. Returns [] on any failure."""
    if not queries:
        return []

    client = _tavily_client()
    if client is None:
        logger.warning("[websearch] Tavily not configured — skipping search")
        return []

    seen_urls: set[str] = set()
    results: list[dict[str, Any]] = []

    for query in queries[:3]:
        try:
            # Privacy: only the extracted claim query is sent — not the full forwarded message.
            response = client.search(
                query=query,
                search_depth=config.SEARCH_DEPTH,
                max_results=config.SEARCH_MAX_RESULTS,
                include_domains=config.FACTCHECK_DOMAINS,
            )
            for hit in response.get("results") or []:
                if not isinstance(hit, dict):
                    continue
                normalized = _normalize_hit(hit)
                if not normalized:
                    continue
                url = normalized["source_url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(normalized)
                logger.info("[websearch] hit: %s", url)
                if len(results) >= config.SEARCH_MAX_RESULTS:
                    return results
        except Exception as exc:
            logger.warning("[websearch] search failed for %r: %s", query, exc)

    return results


def search_for_message(text: str) -> list[dict[str, Any]]:
    """extract_claims → search convenience wrapper."""
    queries = extract_claims(text)
    return search(queries)
