"""Web search grounding via Tavily — structured queries, tiered domains, score filtering."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal
from urllib.parse import urlparse

from tavily import TavilyClient

from backend import config
from backend.ollama_chat import chat_json

logger = logging.getLogger(__name__)

_tavily: TavilyClient | None = None

DomainMode = Literal["filtered", "unfiltered"]

STRUCTURED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "core_claim": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "broad_queries": {"type": "array", "items": {"type": "string"}},
        "narrow_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["core_claim", "entities", "broad_queries", "narrow_queries"],
}

STRUCTURED_PROMPT = """คุณคือผู้เชี่ยวชาญด้านการสกัดข้อมูลเพื่อตรวจสอบข้อเท็จจริง จงอ่านข้อความที่ผู้ใช้ได้รับ แล้วสกัดข้อมูลตามข้อกำหนดด้านล่างเพื่อนำไปค้นหาในฐานข้อมูลข่าวปลอม

กฎการสกัดข้อมูล:
1. core_claim: สรุปใจความสำคัญที่เป็นข้ออ้างหลัก 1 ประโยคสั้น น้ำเสียงเป็นกลาง (ไม่ใช่ข้อความแชร์ต่อ)
2. entities: รายชื่อหน่วยงาน บุคคล แอพพลิเคชัน หรือตัวเลขสำคัญ (ใส่เป็น Array, 0-5 รายการ)
3. broad_queries: ประโยคค้นหาภาษาไทย 2-3 ข้อ โดยผสม [ชื่อหน่วยงาน/แบรนด์] + [สิ่งที่อ้าง] + [คำค้นหาข้อเท็จจริง เช่น "ข่าวปลอม", "จริงไหม", "ชัวร์ก่อนแชร์", "เฟคนิวส์"] เพื่อให้ตรงกับหัวข้อบทความตรวจสอบข่าวปลอม
4. narrow_queries: คีย์เวิร์ดหรือประโยคสั้น 1-2 ข้อ ที่เจาะจงมุมอันตรายของข้ออ้างนั้น เช่น ชื่อ LINE ปลอม, การขอ OTP, การให้กดลิงก์, การโอนเงิน

ข้อห้าม:
- ตัดคำชวนแชร์, คำเร่งด่วน (เช่น ด่วนที่สุด, แชร์เลย), และอีโมจิออกทั้งหมด
- ห้ามคัดลอกข้อความยาว ๆ มาใส่ใน query ให้เน้นประโยคสั้นกระชับเท่านั้น

ตัวอย่างการตอบกลับ (ต้องตอบเป็น JSON รูปแบบนี้เท่านั้น):
{
    "core_claim": "การไฟฟ้าส่วนภูมิภาคเปิดช่องทางไลน์ฝ่ายงานทะเบียนเพื่อรับแจ้งปัญหาและทำธุรกรรม",
    "entities": ["การไฟฟ้าส่วนภูมิภาค", "PEA", "LINE", "ฝ่ายงานทะเบียน"],
    "broad_queries": [
        "PEA เปิดไลน์กลุ่มทะเบียนใหม่ ข่าวปลอม",
        "การไฟฟ้าส่วนภูมิภาค เปิดไลน์ฝ่ายงานทะเบียน จริงไหม"
    ],
    "narrow_queries": [
        "ไลน์ปลอม PEA ฝ่ายงานทะเบียน คืนเงินค่าไฟ",
        "แอดไลน์ PEA รับเงินคืนค่าส่วนต่างหม้อแปลง"
    ]
}

ข้อความที่ต้องสกัด:
"{user_text}"
"""

NOISE_WORDS = re.compile(
    r"(ด่วน!?|ช่วยแชร์|ส่งต่อ|แชร์ให้|โอกาสสุดท้าย|สมัครวันนี้|วันนี้เท่านั้น)",
    re.IGNORECASE,
)

_OFFICIAL_AGENCY_PREFIX = re.compile(r"^(กรม|สำนักงาน|การไฟฟ้า|ไปรษณีย์)")


def _clip_query(text: str) -> str:
    return text.strip()[: config.SEARCH_QUERY_MAX_LEN]


def _tavily_client() -> TavilyClient | None:
    global _tavily
    if not config.TAVILY_API_KEY:
        return None
    if _tavily is None:
        _tavily = TavilyClient(api_key=config.TAVILY_API_KEY)
    return _tavily


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _url_key(url: str) -> str:
    try:
        p = urlparse(url.strip())
        host = p.netloc.lower().removeprefix("www.")
        path = p.path.rstrip("/") or "/"
        return f"{host}{path}"
    except Exception:
        return url.strip().lower()


def _is_official_advisory(message: str) -> bool:
    """Skip debunk suffix for likely legitimate government advisories."""
    text = message.strip()
    if ".go.th" in text:
        return True
    return bool(_OFFICIAL_AGENCY_PREFIX.match(text))


def _heuristic_structured(text: str) -> dict[str, Any]:
    cleaned = NOISE_WORDS.sub("", _normalize(text)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return {
            "core_claim": "",
            "entities": [],
            "broad_queries": [],
            "narrow_queries": [],
        }
    core = _clip_query(cleaned)
    narrow = _clip_query(cleaned) if len(cleaned) > 40 else ""
    return {
        "core_claim": core,
        "entities": [],
        "broad_queries": [core] if core else [],
        "narrow_queries": [narrow] if narrow and narrow != core else [],
    }


def extract_structured(text: str) -> dict[str, Any]:
    """Turn message into structured search plan. Privacy: only queries leave the machine."""
    message = text.strip()
    if not message:
        return {
            "core_claim": "",
            "entities": [],
            "broad_queries": [],
            "narrow_queries": [],
        }

    try:
        parsed = chat_json(
            [
                {"role": "system", "content": STRUCTURED_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.1,
            schema=STRUCTURED_SCHEMA,
        )
        raw_broad = [
            str(q).strip() for q in (parsed.get("broad_queries") or []) if str(q).strip()
        ]
        raw_narrow = [
            str(q).strip() for q in (parsed.get("narrow_queries") or []) if str(q).strip()
        ]
        broad = [_clip_query(q) for q in raw_broad if _clip_query(q)][:3]
        narrow = [_clip_query(q) for q in raw_narrow if _clip_query(q)][:2]
        entities = [
            str(e).strip() for e in (parsed.get("entities") or []) if str(e).strip()
        ][:5]
        core_claim = str(parsed.get("core_claim") or "").strip()
        if not broad and core_claim:
            broad = [_clip_query(core_claim)]
        if not narrow and entities:
            narrow = [_clip_query(" ".join(entities[:3]))]
        result = {
            "core_claim": core_claim,
            "entities": entities,
            "broad_queries": broad,
            "narrow_queries": narrow,
        }
        logger.info(
            "[websearch] structured core_claim=%r broad=%s narrow=%s",
            core_claim,
            broad,
            narrow,
        )
        return result
    except Exception as exc:
        logger.warning("[websearch] structured extraction failed: %s", exc)

    fallback = _heuristic_structured(message)
    logger.info("[websearch] heuristic structured: %s", fallback)
    return fallback


def _with_debunk_suffix(query: str, suffix: str) -> str:
    combined = f"{query} {suffix}".strip()
    return _clip_query(combined)


def _build_broad_queries(
    broad: list[str],
    *,
    message: str,
    apply_suffix: bool,
) -> list[str]:
    """Build broad pass queries; first stays neutral, later ones get debunk suffix."""
    if not broad:
        return []

    if not apply_suffix or _is_official_advisory(message):
        return list(broad)

    suffixes = config.SEARCH_DEBUNK_SUFFIXES
    if not suffixes:
        return list(broad)

    out: list[str] = []
    seen: set[str] = set()
    for i, q in enumerate(broad):
        if i == 0:
            candidate = q
        else:
            suffix = suffixes[(i - 1) % len(suffixes)]
            candidate = _with_debunk_suffix(q, suffix)
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _domain_label(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return "fact-check"
    for domain, label in config.DOMAIN_LABELS.items():
        if domain in host:
            return label
    return host or "fact-check"


def _normalize_hit(
    hit: dict[str, Any],
    *,
    query: str,
    domain_tier: DomainMode,
) -> dict[str, Any] | None:
    url = str(hit.get("url") or "").strip()
    raw = str(hit.get("raw_content") or hit.get("content") or "").strip()
    snippet = str(hit.get("content") or raw).strip()
    if not url or not snippet:
        return None
    raw_score = hit.get("score")
    try:
        score = float(raw_score) if raw_score is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0
    title = str(hit.get("title") or "").strip() or _domain_label(url)
    return {
        "text": snippet[:800],
        "content_full": raw[: config.STANCE_CONTENT_MAX_CHARS],
        "title": title,
        "source": _domain_label(url),
        "source_url": url,
        "reason_th": snippet[:200],
        "score": score,
        "query": query,
        "domain_tier": domain_tier,
    }


def _entity_matches_snippet(entities: list[str], snippet: str) -> bool:
    if not entities:
        return True
    snippet_lower = snippet.lower()
    for entity in entities:
        token = entity.strip()
        if len(token) >= 2 and token.lower() in snippet_lower:
            return True
    return False


def _hits_sufficient(hits: list[dict[str, Any]]) -> bool:
    return any(h.get("score", 0) >= config.SEARCH_MIN_SCORE for h in hits)


def _rank_and_filter_hits(
    hits: list[dict[str, Any]],
    *,
    entities: list[str],
) -> list[dict[str, Any]]:
    """Sort by score, apply threshold, optional entity gate, dedupe, cap."""
    filtered: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for hit in hits:
        score = hit.get("score", 0)
        if score < config.SEARCH_MIN_SCORE:
            continue
        snippet = hit.get("text") or ""
        if entities and not _entity_matches_snippet(entities, snippet):
            continue
        url = hit.get("source_url") or ""
        key = _url_key(url)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        filtered.append(hit)

    # Prefer filtered-domain hits when scores are close
    filtered.sort(
        key=lambda h: (
            h.get("domain_tier") != "filtered",
            -float(h.get("score", 0)),
        )
    )
    return filtered[: config.SEARCH_MAX_RESULTS]


def _search_queries(
    queries: list[str],
    *,
    domain_mode: DomainMode,
    pass_name: str,
    capture_raw: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    if not queries:
        return [], traces

    client = _tavily_client()
    if client is None:
        logger.warning("[websearch] Tavily not configured — skipping search")
        if capture_raw:
            traces.append(
                {
                    "pass": pass_name,
                    "domain_mode": domain_mode,
                    "error": "Tavily not configured",
                    "queries": [],
                }
            )
        return [], traces

    seen_keys: set[str] = set()
    hits: list[dict[str, Any]] = []

    kwargs: dict[str, Any] = {
        "search_depth": config.SEARCH_DEPTH,
        "max_results": config.SEARCH_HITS_PER_QUERY,
        "include_raw_content": True,
    }
    if domain_mode == "filtered":
        kwargs["include_domains"] = config.FACTCHECK_DOMAINS

    for query in queries:
        if not query:
            continue
        trace: dict[str, Any] = {
            "pass": pass_name,
            "query": query,
            "domain_mode": domain_mode,
            "search_depth": config.SEARCH_DEPTH,
            "max_results": config.SEARCH_HITS_PER_QUERY,
            "include_domains": list(config.FACTCHECK_DOMAINS)
            if domain_mode == "filtered"
            else None,
            "error": None,
            "raw_results": [],
            "hits": [],
        }
        try:
            response = client.search(query=query, **kwargs)
        except Exception as exc:
            logger.warning(
                "[websearch] search failed pass=%s query=%r: %s",
                pass_name,
                query,
                exc,
            )
            trace["error"] = str(exc)
            if capture_raw:
                traces.append(trace)
            continue

        if capture_raw:
            trace["raw_results"] = response.get("results") or []

        query_hits = 0
        for raw in response.get("results") or []:
            if not isinstance(raw, dict):
                continue
            normalized = _normalize_hit(
                raw, query=query, domain_tier=domain_mode
            )
            if not normalized:
                continue
            key = _url_key(normalized["source_url"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hits.append(normalized)
            trace["hits"].append(normalized)
            query_hits += 1
            logger.info(
                "[websearch] hit pass=%s tier=%s score=%.2f url=%s",
                pass_name,
                domain_mode,
                normalized.get("score", 0),
                normalized["source_url"],
            )
            if query_hits >= config.SEARCH_HITS_PER_QUERY:
                break

        if capture_raw:
            traces.append(trace)

    return hits, traces


def _run_search_pipeline(
    message: str,
    *,
    capture_raw: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run full search cascade; return production evidence + debug trace."""
    structured = extract_structured(message)
    entities = structured.get("entities") or []

    apply_suffix = config.SEARCH_SUFFIX_ON_BROAD_ONLY
    pass1_queries = _build_broad_queries(
        structured.get("broad_queries") or [],
        message=message,
        apply_suffix=apply_suffix,
    )

    trace: dict[str, Any] = {
        "message": message,
        "official_advisory": _is_official_advisory(message),
        "structured": structured,
        "suffix_applied_to_broad": apply_suffix and not _is_official_advisory(message),
        "passes": [],
        "all_hits_before_rank": [],
        "final_hits": [],
        "production_evidence": [],
    }

    all_hits: list[dict[str, Any]] = []
    pass2_used = False
    pass2_queries: list[str] = []

    pass1_hits, pass1_traces = _search_queries(
        pass1_queries,
        domain_mode="filtered",
        pass_name="pass1",
        capture_raw=capture_raw,
    )
    all_hits.extend(pass1_hits)
    pass1_sufficient = _hits_sufficient(all_hits)
    trace["passes"].append(
        {
            "name": "pass1",
            "domain_mode": "filtered",
            "queries": pass1_queries,
            "hit_count": len(pass1_hits),
            "sufficient": pass1_sufficient,
            "traces": pass1_traces,
        }
    )

    if not pass1_sufficient:
        pass2_queries = list(structured.get("narrow_queries") or [])
        if not pass2_queries and entities:
            pass2_queries = [_clip_query(" ".join(entities[:3]))]
        pass2_used = bool(pass2_queries)
        pass2_hits, pass2_traces = _search_queries(
            pass2_queries,
            domain_mode="filtered",
            pass_name="pass2",
            capture_raw=capture_raw,
        )
        all_hits.extend(pass2_hits)
        trace["passes"].append(
            {
                "name": "pass2",
                "domain_mode": "filtered",
                "queries": pass2_queries,
                "hit_count": len(pass2_hits),
                "sufficient": _hits_sufficient(all_hits),
                "traces": pass2_traces,
            }
        )

    if not _hits_sufficient(all_hits) and config.SEARCH_OPEN_WEB_FALLBACK:
        fallback_queries = pass1_queries or (pass2_queries if pass2_used else [])
        if not fallback_queries:
            core = structured.get("core_claim") or ""
            if core:
                fallback_queries = [_clip_query(core)]
        pass3_hits, pass3_traces = _search_queries(
            fallback_queries[:2],
            domain_mode="unfiltered",
            pass_name="pass3",
            capture_raw=capture_raw,
        )
        all_hits.extend(pass3_hits)
        trace["passes"].append(
            {
                "name": "pass3",
                "domain_mode": "unfiltered",
                "queries": fallback_queries[:2],
                "hit_count": len(pass3_hits),
                "sufficient": _hits_sufficient(all_hits),
                "traces": pass3_traces,
            }
        )

    trace["all_hits_before_rank"] = list(all_hits)
    final = _rank_and_filter_hits(all_hits, entities=entities)
    trace["final_hits"] = final
    trace["top_score"] = final[0].get("score", 0) if final else 0.0
    trace["pass2_used"] = pass2_used

    production = [
        {
            "text": h["text"],
            "content_full": h.get("content_full", h["text"]),
            "title": h.get("title", h["source"]),
            "source": h["source"],
            "source_url": h["source_url"],
            "reason_th": h["reason_th"],
        }
        for h in final
    ]
    trace["production_evidence"] = production
    return production, trace


def debug_search_for_message(text: str, *, extract_only: bool = False) -> dict[str, Any]:
    """Full pipeline trace: structured extraction + per-query Tavily responses."""
    message = text.strip()
    if not message:
        return {
            "message": "",
            "extract_only": extract_only,
            "official_advisory": False,
            "structured": extract_structured(""),
            "passes": [],
            "all_hits_before_rank": [],
            "final_hits": [],
            "production_evidence": [],
        }

    if extract_only:
        structured = extract_structured(message)
        broad = structured.get("broad_queries") or []
        return {
            "message": message,
            "extract_only": True,
            "official_advisory": _is_official_advisory(message),
            "structured": structured,
            "pass1_queries_preview": _build_broad_queries(
                broad,
                message=message,
                apply_suffix=config.SEARCH_SUFFIX_ON_BROAD_ONLY,
            ),
            "passes": [],
            "all_hits_before_rank": [],
            "final_hits": [],
            "production_evidence": [],
        }

    _, trace = _run_search_pipeline(message, capture_raw=True)
    trace["extract_only"] = False
    return trace


def search_for_message(text: str) -> list[dict[str, Any]]:
    """Structured extract → broad pass → narrow pass → open-web fallback → rank."""
    message = text.strip()
    if not message:
        return []

    production, trace = _run_search_pipeline(message, capture_raw=False)
    logger.info(
        "[websearch] final hits=%d top_score=%.2f pass2_used=%s",
        len(production),
        trace.get("top_score", 0),
        trace.get("pass2_used", False),
    )
    return production


# Backward-compatible alias for callers/tests that used flat extraction
def extract_claims(text: str) -> list[str]:
    structured = extract_structured(text)
    broad = structured.get("broad_queries") or []
    if broad:
        return _build_broad_queries(broad, message=text, apply_suffix=False)
    core = structured.get("core_claim") or ""
    return [_clip_query(core)] if core else []


def search(queries: list[str]) -> list[dict[str, Any]]:
    """Legacy single-pass filtered search (used by tests)."""
    hits, _ = _search_queries(queries, domain_mode="filtered", pass_name="legacy")
    return _rank_and_filter_hits(hits, entities=[])
