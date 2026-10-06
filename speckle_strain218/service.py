"""Measurement orchestration: grid build, per-point DIC, strain, ZIP."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .correlation import measure_point
from .strain import compute_strain_grid

MAX_GRID_POINTS = 400
MAX_SEARCH_RADIUS = 16
MAX_ITER = 100


class ParameterError(ValueError):
    """Raised when request parameters are invalid."""


@dataclass
class MeasureParams:
    roi_x: int
    roi_y: int
    roi_w: int
    roi_h: int
    mm_per_pixel: float
    subset: int
    step_x: int
    step_y: int
    search_radius: int
    max_iter: int


def validate_params(params: MeasureParams, width: int, height: int) -> None:
    p = params
    if p.mm_per_pixel <= 0:
        raise ParameterError("mm_per_pixel must be positive")
    if p.subset < 3 or p.subset % 2 == 0:
        raise ParameterError("subset must be an odd integer >= 3")
    if p.step_x < 1 or p.step_y < 1:
        raise ParameterError("grid step must be a positive integer")
    if not (0 <= p.search_radius <= MAX_SEARCH_RADIUS):
        raise ParameterError(
            f"search_radius must be within [0, {MAX_SEARCH_RADIUS}]"
        )
    if not (1 <= p.max_iter <= MAX_ITER):
        raise ParameterError(f"max_iter must be within [1, {MAX_ITER}]")
    if p.roi_w < p.subset or p.roi_h < p.subset:
        raise ParameterError("ROI must be at least the subset size")
    if (
        p.roi_x < 0
        or p.roi_y < 0
        or p.roi_x + p.roi_w > width
        or p.roi_y + p.roi_h > height
    ):
        raise ParameterError("ROI lies outside the image")
    nx = (p.roi_w - p.subset) // p.step_x + 1
    ny = (p.roi_h - p.subset) // p.step_y + 1
    if nx * ny > MAX_GRID_POINTS:
        raise ParameterError(
            f"grid has {nx * ny} points, limit is {MAX_GRID_POINTS}"
        )


def build_grid(p: MeasureParams) -> tuple[np.ndarray, np.ndarray]:
    """Reference grid point centres (x right, y down), all kept."""
    half = p.subset // 2
    xs = np.arange(
        p.roi_x + half, p.roi_x + p.roi_w - half, p.step_x, dtype=np.float64
    )
    ys = np.arange(
        p.roi_y + half, p.roi_y + p.roi_h - half, p.step_y, dtype=np.float64
    )
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy


def run_measurement(
    ref: np.ndarray, defm: np.ndarray, p: MeasureParams
) -> bytes:
    """Run the full measurement and return the result ZIP as bytes."""
    validate_params(p, ref.shape[1], ref.shape[0])
    gx, gy = build_grid(p)
    ny, nx = gx.shape

    valid = np.zeros((ny, nx), dtype=bool)
    u_px = np.full((ny, nx), np.nan)
    v_px = np.full((ny, nx), np.nan)
    zncc = np.zeros((ny, nx))
    iters = np.zeros((ny, nx), dtype=int)
    converged = np.zeros((ny, nx), dtype=bool)
    failures: list[list[str]] = [[""] * nx for _ in range(ny)]

    for j in range(ny):
        for i in range(nx):
            r = measure_point(
                ref, defm, gx[j, i], gy[j, i],
                p.subset, p.search_radius, p.max_iter,
            )
            valid[j, i] = r.valid
            zncc[j, i] = r.zncc
            iters[j, i] = r.iterations
            converged[j, i] = r.converged
            failures[j][i] = r.failure
            if r.valid:
                u_px[j, i] = r.u
                v_px[j, i] = r.v

    mpp = p.mm_per_pixel
    u_mm = u_px * mpp
    v_mm = v_px * mpp
    strain = compute_strain_grid(
        gx * mpp,
        gy * mpp,
        np.where(valid, u_mm, np.nan),
        np.where(valid, v_mm, np.nan),
        valid,
    )

    return _pack_zip(
        p, gx, gy, valid, u_px, v_px, u_mm, v_mm, zncc, iters,
        converged, failures, strain,
    )


def _fmt(value: float, ok: bool) -> str:
    return f"{value:.6g}" if ok and np.isfinite(value) else ""


def _pack_zip(
    p: MeasureParams,
    gx: np.ndarray,
    gy: np.ndarray,
    valid: np.ndarray,
    u_px: np.ndarray,
    v_px: np.ndarray,
    u_mm: np.ndarray,
    v_mm: np.ndarray,
    zncc: np.ndarray,
    iters: np.ndarray,
    converged: np.ndarray,
    failures: list[list[str]],
    strain,
) -> bytes:
    ny, nx = gx.shape
    mpp = p.mm_per_pixel

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "point_id", "row", "col",
            "x_px", "y_px", "x_mm", "y_mm",
            "valid", "u_px", "v_px", "u_mm", "v_mm",
            "zncc", "iterations", "converged",
            "strain_valid", "exx", "eyy", "exy",
            "failure_reason",
        ]
    )
    pid = 0
    failure_summary: dict[str, int] = {}
    for j in range(ny):
        for i in range(nx):
            s = strain[j][i]
            reason = failures[j][i]
            if reason:
                failure_summary[reason] = failure_summary.get(reason, 0) + 1
            writer.writerow(
                [
                    pid, j, i,
                    f"{gx[j, i]:.1f}", f"{gy[j, i]:.1f}",
                    f"{gx[j, i] * mpp:.6g}", f"{gy[j, i] * mpp:.6g}",
                    int(valid[j, i]),
                    _fmt(u_px[j, i], valid[j, i]),
                    _fmt(v_px[j, i], valid[j, i]),
                    _fmt(u_mm[j, i], valid[j, i]),
                    _fmt(v_mm[j, i], valid[j, i]),
                    f"{zncc[j, i]:.6f}",
                    int(iters[j, i]),
                    int(converged[j, i]),
                    int(s.valid),
                    _fmt(s.exx, s.valid),
                    _fmt(s.eyy, s.valid),
                    _fmt(s.exy, s.valid),
                    reason,
                ]
            )
            pid += 1

    mask = valid.astype(np.uint8) * 255
    mask_buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(mask_buf, format="PNG")

    meta = {
        "parameters": {
            "roi": {"x": p.roi_x, "y": p.roi_y, "w": p.roi_w, "h": p.roi_h},
            "mm_per_pixel": p.mm_per_pixel,
            "subset": p.subset,
            "step_x": p.step_x,
            "step_y": p.step_y,
            "search_radius": p.search_radius,
            "max_iter": p.max_iter,
        },
        "grid": {"rows": ny, "cols": nx, "points": ny * nx},
        "coordinates": "x right, y down, origin at top-left pixel centre",
        "units": {
            "displacement": "mm",
            "strain": "dimensionless",
            "exy": "tensor shear E12",
        },
        "valid_points": int(valid.sum()),
        "strain_valid_points": int(
            sum(s.valid for row in strain for s in row)
        ),
        "failure_summary": failure_summary,
    }

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("points.csv", buf.getvalue())
        zf.writestr("valid_mask.png", mask_buf.getvalue())
        zf.writestr("params.json", json.dumps(meta, indent=2))
    return out.getvalue()
