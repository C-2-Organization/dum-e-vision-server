from __future__ import annotations

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from server.pipeline.grounding_dino import GroundingDINOEngine

app = FastAPI(title="DUM-E Vision Server (GroundingDINO)")

# Paths are provided via env, but we default to /models mount.
GDINO_CONFIG = os.getenv("GDINO_CONFIG", "/models/gdino/config.py")
GDINO_CKPT = os.getenv("GDINO_CKPT", "/models/gdino/model.pth")
DEVICE = os.getenv("DEVICE", "cuda")

engine: GroundingDINOEngine | None = None


@app.on_event("startup")
def _startup():
    global engine
    engine = GroundingDINOEngine(
        config_path=GDINO_CONFIG,
        checkpoint_path=GDINO_CKPT,
        device=DEVICE,
    )


@app.get("/healthz")
def healthz():
    return {"ok": engine is not None, "device": DEVICE}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
    top_k: int = Form(10),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        pil = Image.open(image.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    dets = engine.detect(
        image=pil,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    dets = dets[: max(1, int(top_k))]

    return JSONResponse(
        {
            "text_prompt": text_prompt,
            "count": len(dets),
            "detections": [
                {
                    "phrase": d.phrase,
                    "score": d.score,
                    "bbox_xyxy": list(d.bbox_xyxy),
                }
                for d in dets
            ],
        }
    )
