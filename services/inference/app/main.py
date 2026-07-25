from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .pipeline import ProcessingConfig, analyze

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(
    title="IRIS Inference API",
    description=(
        "Deterministic infrared restoration, perceptual colorization, "
        "quality evaluation, and salient-region analysis."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready", "service": "iris-inference", "version": "0.1.0"}


@app.get("/v1/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "colorMaps": ["thermal", "inferno", "viridis", "grayscale"],
        "formats": ["PNG", "JPEG", "TIFF", "WebP"],
        "maximumUploadMb": 25,
        "processing": [
            "robust normalization",
            "impulse denoising",
            "contrast restoration",
            "detail recovery",
            "perceptual colorization",
            "uncertainty estimation",
            "thermal anomaly proposals",
        ],
    }


@app.post("/v1/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    color_map: str = Form("thermal"),
    denoise_strength: int = Form(38),
    detail_strength: int = Form(62),
    gamma: float = Form(0.92),
    hotspot_percentile: int = Form(91),
) -> JSONResponse:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 25 MB limit")

    try:
        config = ProcessingConfig(
            color_map=color_map,
            denoise_strength=denoise_strength,
            detail_strength=detail_strength,
            gamma=gamma,
            hotspot_percentile=hotspot_percentile,
        )
        result = analyze(content, config)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return JSONResponse(result.to_payload())
