"""FastAPI application: POST /measure -> ZIP with DIC results.

Requests are fully self-contained (no shared state between requests).
"""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile

from .imaging import (
    ImageValidationError,
    check_same_shape,
    decode_gray_png,
)
from .service import (
    MeasureParams,
    ParameterError,
    run_measurement,
)

app = FastAPI(title="speckle_strain218", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/measure")
async def measure(
    reference: UploadFile = File(...),
    deformed: UploadFile = File(...),
    roi_x: int = Form(...),
    roi_y: int = Form(...),
    roi_w: int = Form(...),
    roi_h: int = Form(...),
    mm_per_pixel: float = Form(...),
    subset: int = Form(...),
    step_x: int = Form(...),
    step_y: int = Form(...),
    search_radius: int = Form(...),
    max_iter: int = Form(...),
) -> Response:
    ref_bytes = await reference.read()
    def_bytes = await deformed.read()
    try:
        ref = decode_gray_png(ref_bytes, "reference")
        defm = decode_gray_png(def_bytes, "deformed")
        check_same_shape(ref, defm)
        params = MeasureParams(
            roi_x=roi_x,
            roi_y=roi_y,
            roi_w=roi_w,
            roi_h=roi_h,
            mm_per_pixel=mm_per_pixel,
            subset=subset,
            step_x=step_x,
            step_y=step_y,
            search_radius=search_radius,
            max_iter=max_iter,
        )
        archive = run_measurement(ref, defm, params)
    except (ImageValidationError, ParameterError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=archive,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="dic_result.zip"'
        },
    )
