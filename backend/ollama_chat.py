"""Ollama chat helpers — schema-first JSON with plain fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import ollama

from backend import config

logger = logging.getLogger(__name__)

_client: ollama.Client | None = None


def client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from model output (raw or fenced)."""
    text = text.strip()
    if not text:
        raise ValueError("empty LLM response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        parsed = json.loads(fence.group(1))
        if isinstance(parsed, dict):
            return parsed

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("LLM response is not a JSON object")


# Per-request Go templates passed to /api/generate. Using generate (not chat)
# avoids Ollama rendering the model's embedded Jinja chat template — the source
# of the `selectattr: unknown test 'tool_calls'` crash on Ollama 0.30.x with
# some Gemma3-based models. We supply the prompt format explicitly instead.
_GEMMA_TEMPLATE = (
    "{{ if .System }}<start_of_turn>user\n{{ .System }}\n\n{{ .Prompt }}<end_of_turn>\n"
    "{{ else }}<start_of_turn>user\n{{ .Prompt }}<end_of_turn>\n{{ end }}"
    "<start_of_turn>model\n"
)
_QWEN_TEMPLATE = (
    "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
    "<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n"
)


def _gen_template() -> str:
    m = config.LLM_MODEL.lower()
    if "qwen" in m:
        return _QWEN_TEMPLATE
    return _GEMMA_TEMPLATE  # gemma3 default


def _split_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    user = "\n\n".join(
        m["content"] for m in messages if m.get("role") in ("user", "assistant")
    )
    return system, user


def _chat_raw(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    fmt: str | dict[str, Any] | None = None,
) -> str:
    system, user = _split_messages(messages)
    kwargs: dict[str, Any] = {
        "model": config.LLM_MODEL,
        "prompt": user,
        "system": system,
        "template": _gen_template(),
        "stream": False,
        "options": {"temperature": temperature},
    }
    if fmt is not None:
        kwargs["format"] = fmt
    response = client().generate(**kwargs)
    content = response.get("response", "")
    if not content:
        raise ValueError("empty LLM response")
    return content


def chat_text(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
) -> str:
    return _chat_raw(messages, temperature=temperature)


def chat_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Try JSON schema (Qwen etc.), then format=json, then plain parse."""
    if schema is not None:
        try:
            return extract_json_object(
                _chat_raw(messages, temperature=temperature, fmt=schema)
            )
        except Exception as exc:
            logger.warning("schema chat failed (%s), trying format=json", exc)

        try:
            return extract_json_object(
                _chat_raw(messages, temperature=temperature, fmt="json")
            )
        except Exception as exc:
            logger.warning("format=json failed (%s), trying plain", exc)

    return extract_json_object(_chat_raw(messages, temperature=temperature))
