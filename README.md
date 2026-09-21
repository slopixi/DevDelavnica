# yolo-detection-demo

CPU-only, no-training, real-time-feeling object detection demo, wrapped in a simple browser UI. Built for a boss demo on a machine with no GPU.

## What this is

A small FastAPI backend runs Ultralytics YOLO26 (CPU-only) behind one HTTP endpoint. A single static HTML/JS page captures webcam frames in the browser and draws the returned bounding boxes on a canvas overlay. No GPU, no training step, no cloud inference API.

> **Status:** the initial scaffold (`server.py`, `static/index.html`, `requirements.txt`, `.gitignore`) is implemented and has passed the smoke test described below on the dev machine (WSL2/Linux, CPU-only). The manual webcam test still needs to be done by a human with a camera attached.

## Requirements

- CPU only — no CUDA/GPU dependency anywhere in the stack. `device="cpu"` is explicit everywhere the model is loaded or exported.
- Python 3.10+ and a webcam-capable browser (for the manual test).
- No paid or cloud inference API is used; the only network calls happen once, offline, during model export.

## Setup and run

Run these in order:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. One-time model export (produces models/yolo26n_openvino_model/, gitignored)
yolo export model=yolo26n.pt format=openvino imgsz=480
# If OpenVINO export/runtime isn't available on your machine, fall back to:
#   yolo export model=yolo26n.pt format=onnx imgsz=480
# and update MODEL_PATH in server.py accordingly.

# 4. Start the backend
uvicorn server:app --reload

# 5. Open the frontend
# Open static/index.html in a browser (or serve it via FastAPI's StaticFiles mount)
# and allow webcam access.
```

The app is served at `http://localhost:8000`.

## How it works

Browser → `POST /detect` (single JPEG frame) → FastAPI → YOLO26n CPU inference → JSON detections → browser draws the overlay. There is no streaming protocol, no persistence layer, and no database — each detection is a single request/response round trip.

- `POST /detect` — accepts `multipart/form-data` with an `image` field (JPEG/PNG bytes), returns:
  ```json
  {
    "detections": [
      {"class_name": "person", "confidence": 0.87, "box": [x1, y1, x2, y2]}
    ],
    "inference_ms": 42.3
  }
  ```
- `GET /health` — liveness check, returns `{"status": "ok", "model_loaded": true}`.

Detects standard COCO classes (person, car, dog, etc.) from a live webcam feed at a "few FPS" feel, using the `yolo26n` variant for FPS headroom on CPU-only hardware.

## Performance

Inference time is measured server-side around the `model.predict(...)` call (`inference_ms` in the `/detect` response), not the HTTP round trip.

Observed on the dev machine (WSL2/Linux, 12th Gen Intel Core i7-1255U, CPU-only, OpenVINO export, `imgsz=480`), 10 requests against `ultralytics`'s sample `bus.jpg` test image:

- First request: ~1723 ms (includes one-time OpenVINO graph/model warm-up on first inference).
- Remaining 9 requests: 31.3–49.8 ms, average ≈ 36.6 ms (≈ 27 FPS steady state).

So expect a one-time warm-up delay right after `GET /health` first reports `model_loaded: true`, then a "few FPS" feel matching the design target for the rest of the session.

## Known limitations

- Snapshot-polling (capture → upload → response, on a fixed interval) rather than true continuous video streaming — there will be some perceptible latency compared to a live feed.
- Single-frame inference with no tracking or smoothing between frames, so boxes may jitter frame to frame.
- No authentication and no TLS; intended for localhost-only demo use, not external/LAN exposure.

## Testing

No formal automated test suite for this first release. Before calling any change "done":

```bash
curl -X POST http://localhost:8000/detect -F "image=@test.jpg"
```

Confirm a `200` response with a `detections` array containing at least one plausible box for a known test image (e.g. a photo with a visible person). Also manually confirm in a browser with a webcam attached that a bounding box appears and tracks roughly in real time as you move.

## Project layout

```
yolo-detection-demo/
├── server.py                       # FastAPI app, model load, /detect endpoint
├── static/
│   └── index.html                  # webcam capture + canvas overlay, vanilla JS
├── models/
│   └── yolo26n_openvino_model/     # exported model, gitignored — not committed
├── requirements.txt
├── README.md
├── .gitignore
└── CLAUDE.md
```

## Scope and boundaries

Standalone demo repo, not integrated with any production system, database, or other codebase — no shared imports, config, or credentials. See `CLAUDE.md` for the full set of invariants and forbidden actions (no GPU deps, no training, no persisted frames, no external exposure without explicit sign-off).
