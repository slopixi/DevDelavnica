# CLAUDE.md — yolo-detection-demo

## Discovery Summary
- Domain problem: need a working, presentable live object-detection demo for a boss demo, with no GPU available.
- Chosen product shape: a small FastAPI backend running Ultralytics YOLO26 (CPU-only) behind one HTTP endpoint, plus a single static HTML/JS page that captures webcam frames and draws the returned boxes.
- Architecture/stack rationale: YOLO26 is NMS-free and specifically optimized for CPU/edge inference, exported to ONNX/OpenVINO for extra CPU speed; the nano variant is the default for FPS headroom on CPU-only hardware.
- Alternatives rejected: larger YOLO26 variants (s/m/l/x) — too slow on CPU for the FPS target; continuous WebSocket video streaming — real-time-feeling but unnecessary implementation risk for a one-day demo; custom training — out of scope, pretrained COCO weights only.
- First release scope: runs locally on Primoz's own WSL2/Linux machine, reachable via a browser at `http://localhost:8000`, detecting standard COCO classes (person, car, dog, etc.) from a live webcam feed at a "few FPS" feel.

## Mission
- This repo exists to demonstrate CPU-only, no-training, real-time-feeling object detection, wrapped in a simple browser UI.
- The promise that must not be broken: it must run without a GPU, without any model training step, and without depending on any paid/cloud inference API.

## Architecture

### Repository layout
```
yolo-detection-demo/
├── server.py            # FastAPI app, model load, /detect endpoint
├── static/
│   └── index.html        # webcam capture + canvas overlay, vanilla JS
├── models/
│   └── yolo26n_openvino_model/   # exported model, gitignored
├── requirements.txt
├── README.md
├── .gitignore
└── CLAUDE.md
```

### Backend (`server.py`)
- Framework: FastAPI + Uvicorn. Model is loaded exactly once at process startup (FastAPI `lifespan`/startup hook), never per-request.
- Model: `ultralytics.YOLO`, loaded from the exported OpenVINO (preferred) or ONNX directory, not the raw `.pt` file, at inference time. Explicitly pass `device="cpu"`.
- Config constants (top of file or env vars), with suggested defaults:
  - `MODEL_PATH = "models/yolo26n_openvino_model"`
  - `IMG_SIZE = 480` (inference resolution — smaller than 640 to trade some accuracy for CPU speed)
  - `CONF_THRESHOLD = 0.4`
- Endpoint contract:
  - `POST /detect`
  - Request: `multipart/form-data`, single field `image` (JPEG/PNG bytes from the browser canvas).
  - Response (`200 OK`, `application/json`):
    ```json
    {
      "detections": [
        {"class_name": "person", "confidence": 0.87, "box": [x1, y1, x2, y2]}
      ],
      "inference_ms": 42.3
    }
    ```
  - `box` coordinates are pixel coordinates in the *submitted* image's frame, so the frontend can draw directly without rescaling math on the backend side.
  - On a decode/inference failure: `422` with `{"error": "<short reason>"}` — never a bare 500 with a stack trace exposed to the browser.
- `GET /health` — trivial liveness check returning `{"status": "ok", "model_loaded": true}`, useful for the demo to confirm the server is up before opening the page.

### Model export (one-time setup step, documented in README, not run by the server)
```bash
yolo export model=yolo26n.pt format=openvino imgsz=480
```
This produces the `models/yolo26n_openvino_model/` directory referenced by `MODEL_PATH`. If OpenVINO export/runtime isn't available on the dev machine, fall back to `format=onnx` and note that explicitly in the report — don't silently swap formats without saying so.

### Frontend (`static/index.html`)
- Vanilla JS, no build step, no framework — single file, served as a static file by FastAPI (`StaticFiles` mount) or opened directly.
- `getUserMedia({video: true})` for webcam access; a hidden `<canvas>` for frame capture, a visible `<canvas>` overlaid on the `<video>` element for drawing boxes.
- Config constants at top of the `<script>` block:
  - `CAPTURE_INTERVAL_MS = 400` (snapshot cadence)
  - `JPEG_QUALITY = 0.6` (via `canvas.toBlob(..., "image/jpeg", JPEG_QUALITY)`, to keep upload payload small)
  - `DETECT_URL = "/detect"`
- Loop: capture frame → `toBlob` → `fetch(DETECT_URL, {method: "POST", body: formData})` → on response, clear the overlay canvas and redraw boxes + `class_name (confidence%)` labels.
- If a request is still in flight when the next interval fires, skip that tick rather than queuing requests — the backend should never see overlapping requests from one client.

### Data flow
Browser → `POST /detect` (single JPEG frame) → FastAPI → YOLO26n CPU inference → JSON detections → browser draws overlay. No streaming protocol, no persistence layer, no database.

### Ownership boundary
Standalone demo repo. Not integrated with any Advansys production system, Oracle database, or the SMIB/gaming-platform codebase — no shared imports, no shared config, no shared credentials with those projects.

## Non-negotiable invariants
- No GPU/CUDA dependency anywhere in the stack — this is the whole point of the demo. `device="cpu"` must be explicit wherever the model is loaded or exported, not left to auto-detection.
- No training step — pretrained COCO weights only.
- No captured frame or image is written to persistent storage, ever, including temp files that outlive a single request.
- No third-party network calls at inference time (the model export happens once, offline, as a setup step — not at request time).

## Forbidden actions
- Do not commit model weight or exported-model files (`*.pt`, `*.onnx`, `*.xml`, `*.bin`, anything under `models/`) to git — `.gitignore` must exclude `models/` and document the export command in README instead.
- Do not reference, import, or connect to Advansys production databases, Oracle credentials, or SMIB code from this repo.
- Do not add GPU-only packages (`torch+cuda`, `tensorrt`, etc.) to `requirements.txt`.
- Do not log or persist webcam frames or detection images to disk, including in debug/error logging paths.
- Do not add authentication, TLS, or external exposure (no ngrok/tunneling, no binding to `0.0.0.0` for anything beyond localhost/LAN demo use) without asking first — this is a local demo, not a deployed service.

## Workflow
- Feature branches off `main`; no direct commits to `main` once the initial demo scaffold exists.
- Given the one-day timeline, the very first scaffold commit (repo init, `server.py`, `static/index.html`, `requirements.txt`, `README.md`) may go straight to `main` — but say so explicitly in the commit message (e.g. `"initial scaffold, direct to main for demo deadline"`), and switch to branch+PR for anything after the demo.
- Keep commits scoped: backend, frontend, and docs changes can be separate commits even within one PR.
- Suggested branch naming for anything after the initial scaffold: `feature/<short-description>` (e.g. `feature/openvino-export`).

## Testing
- No formal unit-test suite is required for this first release — explicitly out of scope for v1.
- Required smoke test before calling anything "done":
  ```bash
  curl -X POST http://localhost:8000/detect -F "image=@test.jpg"
  ```
  Confirm a `200` with a `detections` array containing at least one plausible box for a known test image (e.g. a photo with a visible person).
- Required manual test: open `static/index.html` in a browser with a webcam attached, confirm a bounding box appears on-screen and tracks roughly in real time as you move.
- FPS measurement: time the inference call itself (`inference_ms` in the response, measured server-side around the `model.predict(...)` call only, not the HTTP round trip) and report the average over at least 10 frames.
- Report any skipped or failing check honestly rather than as "passed."

## Documentation
- `README.md` must state exact setup and run commands, in order: create venv, `pip install -r requirements.txt`, run the one-time `yolo export ...` command, `uvicorn server:app --reload`, then open `static/index.html`.
- `README.md` must state the CPU-only requirement and the observed approximate FPS/`inference_ms` on the dev machine.
- `README.md` must state known limitations: snapshot-polling latency vs. true continuous streaming; single-frame inference with no tracking/smoothing between frames; no auth, localhost-only by default.

## Reporting
- Final report for any change must state: what changed, how it was tested (smoke test output + manual test result), observed `inference_ms`/FPS, and any known issue outstanding before the demo.
- If live behavior differs from what this file assumes (e.g., detection endpoint shape, model file name, OpenVINO export failing and falling back to ONNX), report the discrepancy — don't silently code around it.
