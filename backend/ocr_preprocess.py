"""CPU image preprocessing for infographic OCR — invert, thumbnail."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from backend import config


def sample_border_luminance(rgb_array: np.ndarray) -> float:
    """Mean luminance of border and corner pixels (grayscale)."""
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    if h < 2 or w < 2:
        return float(np.mean(gray))

    strip = max(1, min(h, w) // 20)
    regions = [
        gray[:strip, :],
        gray[-strip:, :],
        gray[:, :strip],
        gray[:, -strip:],
    ]
    return float(np.mean(np.concatenate([r.ravel() for r in regions])))


def maybe_invert_for_ocr(
    rgb_array: np.ndarray,
    threshold: int | None = None,
) -> tuple[np.ndarray, bool]:
    """Invert dark-background images so CRAFT sees dark-on-light text."""
    limit = threshold if threshold is not None else config.OCR_INVERT_LUMINANCE_THRESHOLD
    if sample_border_luminance(rgb_array) >= limit:
        return rgb_array, False
    inverted = cv2.bitwise_not(rgb_array)
    return inverted, True


def make_thumbnail_bytes(
    rgb_array: np.ndarray,
    max_side: int | None = None,
) -> bytes:
    """Resize for multimodal layout context; returns JPEG bytes."""
    cap = max_side if max_side is not None else config.OCR_THUMBNAIL_MAX_SIDE
    h, w = rgb_array.shape[:2]
    longest = max(h, w)
    if longest > cap:
        scale = cap / longest
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(rgb_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = rgb_array

    image = Image.fromarray(resized)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
