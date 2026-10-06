"""Per-point subset correlation: integer search + sub-pixel affine ZNCC.

Each grid point is tracked independently (no whole-image registration).
The deformed subset is modelled with a 6-parameter local affine warp and
optimised with a forward-additive Gauss-Newton scheme on the zero-mean
normalised sum-of-squared-differences (ZNSSD), which is invariant to
brightness offset and positive gain changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .imaging import sample_with_gradient

FLAT_STD_THRESHOLD = 3.0  # gray levels; below this a subset is "flat"
CONVERGENCE_TOL = 1e-3  # warp-increment norm; ~1e-3 px sub-pixel accuracy
MIN_ZNCC = 0.8


@dataclass
class PointResult:
    valid: bool = False
    u: float = 0.0  # displacement in pixels, x to the right
    v: float = 0.0  # displacement in pixels, y downward
    zncc: float = 0.0
    iterations: int = 0
    converged: bool = False
    failure: str = ""


def _subset_offsets(subset: int) -> tuple[np.ndarray, np.ndarray]:
    half = subset // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    dx, dy = np.meshgrid(ax, ax)
    return dx.ravel(), dy.ravel()


def _zncc(f: np.ndarray, g: np.ndarray) -> float:
    fz = f - f.mean()
    gz = g - g.mean()
    denom = np.sqrt(np.sum(fz * fz) * np.sum(gz * gz))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(fz * gz) / denom)


def _integer_search(
    ref_win: np.ndarray,
    defm: np.ndarray,
    cx: float,
    cy: float,
    radius: int,
) -> tuple[int, int] | None:
    """Exhaustive integer translation search maximising ZNCC."""
    h, w = defm.shape
    best: tuple[float, int, int] | None = None
    half_h = ref_win.shape[0] // 2
    half_w = ref_win.shape[1] // 2
    for iy in range(-radius, radius + 1):
        for ix in range(-radius, radius + 1):
            xc = int(round(cx)) + ix
            yc = int(round(cy)) + iy
            x0, x1 = xc - half_w, xc + half_w + 1
            y0, y1 = yc - half_h, yc + half_h + 1
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            win = defm[y0:y1, x0:x1]
            c = _zncc(ref_win.ravel(), win.ravel())
            if best is None or c > best[0]:
                best = (c, ix, iy)
    if best is None:
        return None
    return best[1], best[2]


def measure_point(
    ref: np.ndarray,
    defm: np.ndarray,
    cx: float,
    cy: float,
    subset: int,
    search_radius: int,
    max_iter: int,
) -> PointResult:
    """Track one reference grid point in the deformed image."""
    res = PointResult()
    h, w = ref.shape
    half = subset // 2

    if (
        cx - half < 0
        or cy - half < 0
        or cx + half > w - 1
        or cy + half > h - 1
    ):
        res.failure = "subset_out_of_bounds"
        return res

    dx, dy = _subset_offsets(subset)
    xs_ref = cx + dx
    ys_ref = cy + dy
    f = ref[ys_ref.astype(np.int64), xs_ref.astype(np.int64)]
    if float(f.std()) < FLAT_STD_THRESHOLD:
        res.failure = "flat_texture"
        return res
    fz = f - f.mean()
    fz_norm = np.sqrt(np.sum(fz * fz))

    seed = _integer_search(
        f.reshape(subset, subset), defm, cx, cy, search_radius
    )
    if seed is None:
        res.failure = "search_out_of_bounds"
        return res

    # Warp parameters p = (u, v, a, b, c, d):
    #   x' = x + u + a*dx + b*dy
    #   y' = y + v + c*dx + d*dy
    p = np.array([float(seed[0]), float(seed[1]), 0.0, 0.0, 0.0, 0.0])
    margin = 1.0  # keep one pixel border for interpolation
    converged = False
    zncc = 0.0
    for it in range(1, max_iter + 1):
        wx = xs_ref + p[0] + p[2] * dx + p[3] * dy
        wy = ys_ref + p[1] + p[4] * dx + p[5] * dy
        if (
            wx.min() < margin
            or wy.min() < margin
            or wx.max() > w - 1 - margin
            or wy.max() > h - 1 - margin
        ):
            res.failure = "warp_out_of_bounds"
            res.iterations = it
            return res
        g, gx, gy = sample_with_gradient(defm, wx, wy)
        gz = g - g.mean()
        gz_norm = np.sqrt(np.sum(gz * gz))
        if gz_norm <= 1e-12:
            res.failure = "degenerate_intensity"
            res.iterations = it
            return res
        scale = fz_norm / gz_norm
        # ZNSSD residual and its steepest-descent images.
        r = fz - scale * gz
        jx = scale * gx
        jy = scale * gy
        J = np.stack(
            [jx, jy, jx * dx, jx * dy, jy * dx, jy * dy], axis=1
        )
        H = J.T @ J
        b = J.T @ r
        try:
            delta = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            res.failure = "degenerate_hessian"
            res.iterations = it
            return res
        if not np.all(np.isfinite(delta)):
            res.failure = "degenerate_hessian"
            res.iterations = it
            return res
        p += delta
        zncc = float(np.sum(fz * gz) / (fz_norm * gz_norm))
        res.iterations = it
        if np.linalg.norm(delta) < CONVERGENCE_TOL:
            converged = True
            break

    res.u = float(p[0])
    res.v = float(p[1])
    res.zncc = zncc
    res.converged = converged
    if not converged:
        res.failure = "not_converged"
        return res
    if zncc < MIN_ZNCC:
        res.failure = "low_correlation"
        return res
    res.valid = True
    return res
