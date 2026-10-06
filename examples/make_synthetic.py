"""Generate a reproducible synthetic speckle pair with known deformation.

The deformed image applies a rigid translation plus homogeneous strain:
    x_def = x + U0 + A @ (x - c)
with U0 = (2.0, 1.0) px, du/dx = 0.01, dv/dy = -0.005, du/dy = 0.002,
dv/dx = 0.0, sampled by inverse mapping with bilinear interpolation.
A brightness offset (+8) and gain (x1.05) are applied to the deformed
image to demonstrate ZNCC robustness.

Usage: .venv/bin/python examples/make_synthetic.py [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speckle_strain218.imaging import sample_bilinear  # noqa: E402

SIZE = 200
SEED = 218

U0 = np.array([2.0, 1.0])  # rigid translation (px)
A = np.array([[0.01, 0.002], [0.0, -0.005]])  # displacement gradient


def make_reference(size: int = SIZE, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    n_speckles = 4000
    cx = rng.uniform(0, size, n_speckles)
    cy = rng.uniform(0, size, n_speckles)
    amp = rng.uniform(60, 200, n_speckles)
    sig = rng.uniform(1.0, 2.0, n_speckles)
    for k in range(n_speckles):
        r2 = (xx - cx[k]) ** 2 + (yy - cy[k]) ** 2
        img += amp[k] * np.exp(-r2 / (2 * sig[k] ** 2))
    img -= img.min()
    img = img / img.max() * 255.0
    return img


def deform(ref: np.ndarray) -> np.ndarray:
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    # Forward map: x_def = x + U0 + A (x - c); invert for sampling.
    M = np.eye(2) + A
    Minv = np.linalg.inv(M)
    coords = np.stack([xx.ravel() - c[0], yy.ravel() - c[1]])
    src = Minv @ (coords - U0[:, None])
    xs = (src[0] + c[0]).reshape(h, w)
    ys = (src[1] + c[1]).reshape(h, w)
    out = sample_bilinear(ref, xs, ys)
    out = out * 1.05 + 8.0  # gain + brightness offset
    return np.clip(out, 0, 255)


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/out")
    outdir.mkdir(parents=True, exist_ok=True)
    ref = make_reference()
    defm = deform(ref)
    Image.fromarray(np.round(ref).astype(np.uint8), mode="L").save(
        outdir / "reference.png"
    )
    Image.fromarray(np.round(defm).astype(np.uint8), mode="L").save(
        outdir / "deformed.png"
    )
    print(f"wrote {outdir/'reference.png'} and {outdir/'deformed.png'}")
    print(f"imposed: translation {U0.tolist()} px, gradient {A.tolist()}")


if __name__ == "__main__":
    main()
