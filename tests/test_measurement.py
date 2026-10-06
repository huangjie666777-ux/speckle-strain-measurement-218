import csv
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from speckle_strain218.app import app
from speckle_strain218.correlation import measure_point
from speckle_strain218.imaging import ImageValidationError, decode_gray_png
from speckle_strain218.service import (
    MeasureParams,
    ParameterError,
    run_measurement,
    validate_params,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from make_synthetic import A, U0, deform, make_reference  # noqa: E402

PARAMS = MeasureParams(
    roi_x=20, roi_y=20, roi_w=160, roi_h=160,
    mm_per_pixel=0.01, subset=21, step_x=20, step_y=20,
    search_radius=8, max_iter=50,
)


def _png_bytes(arr: np.ndarray, mode: str = "L") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _post(client, ref, defm, **overrides):
    fields = {
        "roi_x": 20, "roi_y": 20, "roi_w": 160, "roi_h": 160,
        "mm_per_pixel": 0.01, "subset": 21, "step_x": 20, "step_y": 20,
        "search_radius": 8, "max_iter": 50,
    }
    fields.update(overrides)
    return client.post(
        "/measure",
        files={
            "reference": ("ref.png", ref, "image/png"),
            "deformed": ("def.png", defm, "image/png"),
        },
        data={k: str(v) for k, v in fields.items()},
    )


@pytest.fixture(scope="module")
def pair():
    ref = make_reference()
    return ref, deform(ref)


def test_decode_rejects_color():
    rgb = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    with pytest.raises(ImageValidationError):
        decode_gray_png(buf.getvalue(), "reference")


def test_decode_rejects_oversize():
    arr = np.zeros((513, 10), dtype=np.uint8)
    with pytest.raises(ImageValidationError):
        decode_gray_png(_png_bytes(arr), "reference")


def test_param_limits():
    base = dict(PARAMS.__dict__)
    with pytest.raises(ParameterError):
        validate_params(MeasureParams(**{**base, "subset": 20}), 200, 200)
    with pytest.raises(ParameterError):
        validate_params(MeasureParams(**{**base, "search_radius": 17}), 200, 200)
    with pytest.raises(ParameterError):
        validate_params(MeasureParams(**{**base, "max_iter": 101}), 200, 200)
    with pytest.raises(ParameterError):
        validate_params(MeasureParams(**{**base, "mm_per_pixel": 0.0}), 200, 200)
    with pytest.raises(ParameterError):
        validate_params(
            MeasureParams(**{**base, "step_x": 1, "step_y": 1}), 200, 200
        )  # too many grid points
    with pytest.raises(ParameterError):
        validate_params(MeasureParams(**{**base, "roi_w": 500}), 200, 200)


def test_point_tracks_known_displacement(pair):
    ref, defm = pair
    r = measure_point(ref, defm, 100.0, 100.0, 21, 8, 50)
    assert r.valid and r.converged
    assert r.zncc > 0.95
    assert abs(r.u - U0[0]) < 0.05
    assert abs(r.v - U0[1]) < 0.05


def test_flat_texture_fails():
    ref = np.full((60, 60), 128.0)
    defm = np.full((60, 60), 140.0)
    r = measure_point(ref, defm, 30.0, 30.0, 21, 4, 20)
    assert not r.valid
    assert r.failure == "flat_texture"


def test_out_of_bounds_fails():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (60, 60))
    r = measure_point(img, img, 2.0, 30.0, 21, 4, 20)
    assert not r.valid
    assert r.failure == "subset_out_of_bounds"


def test_full_measurement_recovers_strain(pair):
    ref, defm = pair
    archive = run_measurement(ref, defm, PARAMS)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        assert set(zf.namelist()) == {
            "points.csv", "valid_mask.png", "params.json",
        }
        rows = list(csv.DictReader(io.StringIO(zf.read("points.csv").decode())))
        meta = json.loads(zf.read("params.json"))
    assert len(rows) == meta["grid"]["points"] == 49
    valid = [r for r in rows if r["valid"] == "1"]
    assert len(valid) >= 40
    c = (pair[0].shape[1] - 1) / 2.0
    exp_u = np.array([
        U0[0] + A[0, 0] * (float(r["x_px"]) - c) + A[0, 1] * (float(r["y_px"]) - c)
        for r in valid
    ])
    exp_v = np.array([
        U0[1] + A[1, 0] * (float(r["x_px"]) - c) + A[1, 1] * (float(r["y_px"]) - c)
        for r in valid
    ])
    u = np.array([float(r["u_px"]) for r in valid])
    v = np.array([float(r["v_px"]) for r in valid])
    assert np.abs(u - exp_u).max() < 0.05
    assert np.abs(v - exp_v).max() < 0.05
    svalid = [r for r in rows if r["strain_valid"] == "1"]
    assert svalid, "expected valid strain points"
    exx = np.array([float(r["exx"]) for r in svalid])
    eyy = np.array([float(r["eyy"]) for r in svalid])
    exy = np.array([float(r["exy"]) for r in svalid])
    # Green-Lagrange strain of the imposed gradient A.
    F = np.eye(2) + A
    E = 0.5 * (F.T @ F - np.eye(2))
    assert abs(exx.mean() - E[0, 0]) < 2e-3
    assert abs(eyy.mean() - E[1, 1]) < 2e-3
    assert abs(exy.mean() - E[0, 1]) < 2e-3
    # displacement reported in mm
    u_mm = np.array([float(r["u_mm"]) for r in valid])
    assert np.abs(u_mm - exp_u * PARAMS.mm_per_pixel).max() < 1e-3


def test_api_end_to_end_and_rejection(pair):
    client = TestClient(app)
    ref_b = _png_bytes(np.round(pair[0]))
    def_b = _png_bytes(np.round(pair[1]))
    resp = _post(client, ref_b, def_b)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        assert "points.csv" in zf.namelist()

    assert _post(client, ref_b, def_b, subset=20).status_code == 422
    assert _post(client, ref_b, def_b, search_radius=17).status_code == 422
    assert _post(client, b"not a png", def_b).status_code == 422
    other = _png_bytes(np.zeros((50, 50), dtype=np.uint8))
    assert _post(client, ref_b, other).status_code == 422
