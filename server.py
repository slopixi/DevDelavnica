"""FastAPI backend for the CPU-only YOLO detection demo.

Loads an exported Ultralytics YOLO model once at startup and exposes a single
POST /detect endpoint that runs inference on an uploaded image. CPU-only,
no training, no persisted frames, no third-party network calls at request
time. See CLAUDE.md for the full set of invariants.
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --- Config constants -------------------------------------------------

MODEL_PATH = "models/yolo26n_openvino_model"
IMG_SIZE = 480
CONF_THRESHOLD = 0.4

# --- Model state --------------------------------------------------------

model = None  # loaded once at startup, never per-request


def _load_model():
    """Load the exported model from MODEL_PATH. Fails loudly if missing."""
    if not Path(MODEL_PATH).exists():
        sys.stderr.write(
            f"\nFATAL: MODEL_PATH '{MODEL_PATH}' does not exist.\n"
            "Run the one-time export step before starting the server:\n"
            "  yolo export model=yolo26n.pt format=openvino imgsz=480\n"
            "(see README.md for details). Refusing to start.\n\n"
        )
        raise RuntimeError(f"MODEL_PATH '{MODEL_PATH}' not found; export the model first.")

    from ultralytics import YOLO

    return YOLO(MODEL_PATH, task="detect")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = _load_model()
    yield
    model = None


app = FastAPI(lifespan=lifespan)

# Serve the static frontend (index.html, etc.) under /static.
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image")
    except Exception as exc:
        return JSONResponse(status_code=422, content={"error": f"invalid image: {exc}"})

    try:
        start = time.perf_counter()
        results = model.predict(
            img,
            device="cpu",
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return JSONResponse(status_code=422, content={"error": f"inference failed: {exc}"})

    detections = []
    result = results[0]
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append(
            {
                "class_name": names[cls_id],
                "confidence": confidence,
                "box": [x1, y1, x2, y2],
            }
        )

    return {"detections": detections, "inference_ms": round(inference_ms, 2)}
