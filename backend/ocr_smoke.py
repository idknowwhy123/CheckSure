"""CLI smoke test for infographic OCR pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Run OCR pipeline on a local image.")
    parser.add_argument("image", type=Path, help="Path to JPEG or PNG")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=200,
        help="How many characters of extracted text to print",
    )
    args = parser.parse_args()

    path = args.image
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    from backend.ocr import extract_text, init_ocr_reader

    print("Loading EasyOCR…")
    init_ocr_reader()

    data = path.read_bytes()
    print(f"Running pipeline on {path.name} ({len(data)} bytes)…")
    result = extract_text(data)

    print(f"order_path:    {result.order_path}")
    print(f"box_count:     {result.box_count}")
    print(f"dropped_boxes: {result.dropped_boxes}")
    print("--- text preview ---")
    preview = result.text[: args.preview_chars]
    if len(result.text) > args.preview_chars:
        preview += "…"
    print(preview or "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
