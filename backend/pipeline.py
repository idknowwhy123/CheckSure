"""Shared /check pipeline for the API."""

from __future__ import annotations

from typing import Any

from backend import config
from backend.llm import generate_verdict
from backend.websearch import search_for_message


def build_sources(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for hit in hits[: config.SOURCES_UI]:
        text = hit.get("text") or ""
        snippet = text[:80] + ("…" if len(text) > 80 else "")
        sources.append(
            {
                "snippet_th": snippet,
                "source": hit.get("source") or "fact-check",
                "source_url": hit.get("source_url"),
            }
        )
    return sources


def run_check(text: str, use_web: bool = True) -> dict[str, Any]:
    message = text.strip()
    if not message:
        raise ValueError("text is required")

    evidence: list[dict[str, Any]] | None = None
    hits: list[dict[str, Any]] = []

    if use_web:
        hits = search_for_message(message)
        if hits:
            evidence = hits

    verdict = generate_verdict(message, evidence)
    verdict["sources"] = build_sources(hits) if hits else []
    verdict["grounding"] = "web" if hits else "tone"
    return verdict
