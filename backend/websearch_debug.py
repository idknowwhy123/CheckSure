"""CLI: inspect claim extraction and Tavily search results.

Usage:
  python -m backend.websearch_debug "ข้อความที่ต้องการทดสอบ"
  python -m backend.websearch_debug --extract-only "ข้อความ..."
  python -m backend.websearch_debug --json "ข้อความ..."
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.websearch import debug_search_for_message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Debug websearch: structured extraction + Tavily traces"
    )
    parser.add_argument("text", help="Message to test")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only run LLM claim extraction (no Tavily calls)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON trace",
    )
    args = parser.parse_args(argv)

    result = debug_search_for_message(args.text, extract_only=args.extract_only)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("=== Structured extraction ===")
    structured = result.get("structured") or {}
    print(f"core_claim: {structured.get('core_claim', '')}")
    print(f"entities:   {structured.get('entities', [])}")
    print(f"broad:      {structured.get('broad_queries', [])}")
    print(f"narrow:     {structured.get('narrow_queries', [])}")
    print(f"official_advisory: {result.get('official_advisory')}")

    if result.get("extract_only"):
        preview = result.get("pass1_queries_preview") or []
        if preview:
            print(f"pass1_queries_preview: {preview}")
        return 0

    print("\n=== Search passes ===")
    for p in result.get("passes") or []:
        print(
            f"\n[{p.get('name')}] domain={p.get('domain_mode')} "
            f"queries={p.get('queries')} hits={p.get('hit_count')} "
            f"sufficient={p.get('sufficient')}"
        )
        for trace in p.get("traces") or []:
            print(f"  query: {trace.get('query')}")
            if trace.get("error"):
                print(f"  error: {trace['error']}")
            for raw in trace.get("raw_results") or []:
                url = raw.get("url", "")
                score = raw.get("score", "")
                snippet = str(raw.get("content") or "")[:120]
                print(f"    - score={score} {url}")
                print(f"      {snippet}...")

    print("\n=== Final (production evidence) ===")
    for hit in result.get("production_evidence") or []:
        print(f"- {hit.get('source')}: {hit.get('source_url')}")
        print(f"  {str(hit.get('text', ''))[:160]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
