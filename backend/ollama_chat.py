"""Ollama chat helpers — schema-first JSON with plain fallback."""

from __future__ import annotations

import json
import logging
import re
from json import JSONDecoder
from typing import Any

import ollama

from backend import config

logger = logging.getLogger(__name__)

_client: ollama.Client | None = None

_THINK_BLOCK = re.compile(r"``", re.DOTALL | re.IGNORECASE)


def client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client


def _strip_noise(text: str) -> str:
    text = _THINK_BLOCK.sub("", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from model output (raw, fenced, or with trailing prose)."""
    text = _strip_noise(text)
    if not text:
        raise ValueError("empty LLM response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        decoder = JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(text, start)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        end = text.rfind("}")
        if end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

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
        "options": {
            "temperature": temperature,
            "num_ctx": config.LLM_NUM_CTX,
            "num_predict": config.LLM_NUM_PREDICT,
        },
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


def _parse_json_response(
    raw: str,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        return extract_json_object(raw)
    except Exception as exc:
        preview = raw.strip().replace("\n", " ")[:500]
        logger.error("%s parse failed (%s); raw preview: %s", label, exc, preview)
        raise ValueError(f"{label} response is not a JSON object") from exc


def _schema_supported() -> bool:
    """Use JSON schema when the caller supplies one (Ollama structured output)."""
    return True


def _json_with_fallback(
    fetch: Any,
    *,
    label: str,
    temperature: float,
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Try JSON schema (Qwen), then format=json, then plain parse."""
    effective_schema = schema if _schema_supported() else None

    if effective_schema is not None:
        try:
            return _parse_json_response(
                fetch(temperature=temperature, fmt=effective_schema), label=label
            )
        except Exception as exc:
            logger.warning("%s schema failed (%s), trying format=json", label, exc)

    try:
        return _parse_json_response(
            fetch(temperature=temperature, fmt="json"), label=label
        )
    except Exception as exc:
        logger.warning("%s format=json failed (%s), trying plain", label, exc)

    return _parse_json_response(fetch(temperature=temperature, fmt=None), label=label)


def chat_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Try JSON schema (Qwen etc.), then format=json, then plain parse."""

    def fetch(*, temperature: float, fmt: str | dict[str, Any] | None) -> str:
        return _chat_raw(messages, temperature=temperature, fmt=fmt)

    return _json_with_fallback(
        fetch, label="chat", temperature=temperature, schema=schema
    )


def _chat_raw_with_image(
    messages: list[dict[str, str]],
    image_b64: str,
    *,
    temperature: float,
    fmt: str | dict[str, Any] | None = None,
) -> str:
    """Multimodal chat — vision models require /api/chat, not custom generate templates."""
    system, user = _split_messages(messages)
    chat_messages: list[dict[str, Any]] = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    chat_messages.append(
        {"role": "user", "content": user, "images": [image_b64]}
    )
    kwargs: dict[str, Any] = {
        "model": config.LLM_MODEL,
        "messages": chat_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": config.LLM_NUM_CTX,
            "num_predict": config.LLM_NUM_PREDICT,
        },
    }
    if fmt is not None:
        kwargs["format"] = fmt
    response = client().chat(**kwargs)
    content = response.get("message", {}).get("content", "")
    if not content:
        raise ValueError("empty multimodal LLM response")
    return content


def chat_json_with_image(
    messages: list[dict[str, str]],
    image_b64: str,
    *,
    temperature: float = 0.0,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON from a vision-capable model; image is layout context only."""

    def fetch(*, temperature: float, fmt: str | dict[str, Any] | None) -> str:
        return _chat_raw_with_image(
            messages, image_b64, temperature=temperature, fmt=fmt
        )

    return _json_with_fallback(
        fetch, label="multimodal", temperature=temperature, schema=schema
    )
