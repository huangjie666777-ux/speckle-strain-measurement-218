"""Strain recovery from displacement fields.

For every grid point the valid displacements inside its 3x3 grid
neighbourhood are fitted with a 2D linear model (least squares). At
least three non-collinear valid points are required, otherwise the
strain at that point is invalid and left empty (never interpolated).
The displacement gradient gives F = I + grad(u), and the Green-Lagrange
strain is E = (F^T F - I) / 2 with tensor shear Exy = E12.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_FIT_POINTS = 3


@dataclass
class StrainResult:
    valid: bool = False
    exx: float = 0.0
    eyy: float = 0.0
    exy: float = 0.0  # tensor shear component E12
    failure: str = ""


def compute_strain_grid(
    xs_mm: np.ndarray,
    ys_mm: np.ndarray,
    u_mm: np.ndarray,
    v_mm: np.ndarray,
    valid: np.ndarray,
) -> list[list[StrainResult]]:
    """Compute strain for every point of an (ny, nx) grid."""
    ny, nx = valid.shape
    out: list[list[StrainResult]] = []
    for j in range(ny):
        row: list[StrainResult] = []
        for i in range(nx):
            row.append(_fit_point(j, i, xs_mm, ys_mm, u_mm, v_mm, valid))
        out.append(row)
    return out


def _fit_point(
    j: int,
    i: int,
    xs_mm: np.ndarray,
    ys_mm: np.ndarray,
    u_mm: np.ndarray,
    v_mm: np.ndarray,
    valid: np.ndarray,
) -> StrainResult:
    res = StrainResult()
    ny, nx = valid.shape
    j0, j1 = max(0, j - 1), min(ny, j + 2)
    i0, i1 = max(0, i - 1), min(nx, i + 2)
    m = valid[j0:j1, i0:i1]
    xs = xs_mm[j0:j1, i0:i1][m]
    ys = ys_mm[j0:j1, i0:i1][m]
    us = u_mm[j0:j1, i0:i1][m]
    vs = v_mm[j0:j1, i0:i1][m]
    if xs.size < MIN_FIT_POINTS:
        res.failure = "insufficient_valid_neighbours"
        return res
    x0 = xs_mm[j, i]
    y0 = ys_mm[j, i]
    A = np.column_stack([np.ones_like(xs), xs - x0, ys - y0])
    sv = np.linalg.svd(A, compute_uv=False)
    if sv.size < 3 or sv[-1] <= 1e-12 * max(sv[0], 1.0):
        res.failure = "collinear_neighbours"
        return res
    cu, *_ = np.linalg.lstsq(A, us, rcond=None)
    cv, *_ = np.linalg.lstsq(A, vs, rcond=None)
    dudx, dudy = cu[1], cu[2]
    dvdx, dvdy = cv[1], cv[2]
    F = np.array([[1.0 + dudx, dudy], [dvdx, 1.0 + dvdy]], dtype=np.float64)
    E = 0.5 * (F.T @ F - np.eye(2))
    res.valid = True
    res.exx = float(E[0, 0])
    res.eyy = float(E[1, 1])
    res.exy = float(E[0, 1])
    return res
