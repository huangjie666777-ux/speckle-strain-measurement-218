"""Image decoding, validation and sub-pixel sampling.

Coordinate convention: x to the right, y downward, origin at the
top-left pixel centre of the image. Images are 8-bit grayscale PNG.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

MAX_SIDE = 512


class ImageValidationError(ValueError):
    """Raised when an uploaded image does not meet the requirements."""


def decode_gray_png(data: bytes, name: str) -> np.ndarray:
    """Decode bytes as an 8-bit grayscale PNG and return a float64 array."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != "PNG":
                raise ImageValidationError(f"{name}: not a PNG image")
            if img.mode != "L":
                raise ImageValidationError(
                    f"{name}: expected 8-bit grayscale PNG, got mode {img.mode!r}"
                )
            arr = np.asarray(img, dtype=np.float64)
    except ImageValidationError:
        raise
    except Exception as exc:  # corrupt / undecodable payload
        raise ImageValidationError(f"{name}: cannot decode PNG ({exc})") from exc
    h, w = arr.shape
    if h > MAX_SIDE or w > MAX_SIDE:
        raise ImageValidationError(
            f"{name}: side length {w}x{h} exceeds {MAX_SIDE}px limit"
        )
    return arr


def check_same_shape(ref: np.ndarray, defm: np.ndarray) -> None:
    if ref.shape != defm.shape:
        raise ImageValidationError(
            f"reference {ref.shape[::-1]} and deformed {defm.shape[::-1]} "
            "images must have the same size"
        )


def sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of img at fractional coordinates (xs, ys).

    Coordinates outside the image are clamped to the border; callers that
    need strict bounds must check them beforehand.
    """
    h, w = img.shape
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xs - x0
    wy = ys - y0
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    ia = img[y0c, x0c]
    ib = img[y0c, x1c]
    ic = img[y1c, x0c]
    id_ = img[y1c, x1c]
    return (
        ia * (1 - wx) * (1 - wy)
        + ib * wx * (1 - wy)
        + ic * (1 - wx) * wy
        + id_ * wx * wy
    )


def sample_with_gradient(
    img: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear value plus analytic gradients (d/dx, d/dy) at (xs, ys)."""
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xs - x0
    wy = ys - y0
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    ia = img[y0c, x0c]
    ib = img[y0c, x1c]
    ic = img[y1c, x0c]
    id_ = img[y1c, x1c]
    val = (
        ia * (1 - wx) * (1 - wy)
        + ib * wx * (1 - wy)
        + ic * (1 - wx) * wy
        + id_ * wx * wy
    )
    gx = (ib - ia) * (1 - wy) + (id_ - ic) * wy
    gy = (ic - ia) * (1 - wx) + (id_ - ib) * wx
    return val, gx, gy
