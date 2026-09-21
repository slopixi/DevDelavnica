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
- Backend: FastAPI + Uvicorn, one Python module exposing `POST /detect`, accepts an image, returns JSON with boxes/classes/confidences.
- Model: `yolo26n.pt` (Ultralytics, pretrained on COCO), exported to ONNX or OpenVINO for the CPU inference path; loaded once at startup, not per request.
- Frontend: one static `index.html` using `getUserMedia` to capture the webcam, a JS loop that snapshots a frame every ~300–500ms to a canvas, POSTs it, and draws the returned boxes on an overlay canvas.
- Data flow: browser → HTTP POST (single frame) → FastAPI → YOLO26n inference on CPU → JSON → browser draws overlay. No streaming protocol, no persistence layer.
- Ownership boundary: standalone demo repo, not integrated with any Advansys production system, Oracle database, or the SMIB/gaming-platform codebase.

## Non-negotiable invariants
- No GPU/CUDA dependency anywhere in the stack — this is the whole point of the demo.
- No training step — pretrained COCO weights only.
- No captured frame or image is written to persistent storage.
- No third-party network calls at inference time (model weights may be downloaded once at setup, not per request).

## Forbidden actions
- Do not commit model weight files (`*.pt`, `*.onnx`, `*.xml`, `*.bin`) to git — document the one-time download command instead.
- Do not reference, import, or connect to Advansys production databases, Oracle credentials, or SMIB code from this repo.
- Do not add GPU-only packages (`torch+cuda`, `tensorrt`, etc.) as dependencies.
- Do not log or persist webcam frames or detection images to disk.

## Workflow
- Feature branches off `main`; no direct commits to `main` once the initial demo scaffold exists.
- Given the one-day timeline, the very first scaffold commit may go straight to `main` — but say so explicitly in the commit message, and switch to branch+PR for anything after the demo.
- Keep commits scoped: backend, frontend, and docs changes can be separate commits even within one PR.

## Testing
- No formal unit-test suite is required for this first release — explicitly out of scope for v1.
- Required smoke test before calling anything "done": start the server, `curl` a sample image at `/detect`, confirm valid JSON with at least one plausible detection.
- Required manual test: open the HTML page, stand in front of the webcam, confirm a bounding box appears and tracks roughly in real time.
- Report any skipped or failing check honestly rather than as "passed."

## Documentation
- `README.md` must state exact setup and run commands: install, one-time model download, start backend, open frontend.
- `README.md` must state the CPU-only requirement and the observed approximate FPS on the dev machine.
- `README.md` must state known limitations (snapshot-polling latency vs. true continuous streaming; single-frame inference, no tracking between frames).

## Reporting
- Final report for any change must state: what changed, how it was tested (smoke test + manual test result), observed FPS, and any known issue outstanding before the demo.
- If live behavior differs from what this file assumes (e.g., detection endpoint shape, model file name), report the discrepancy — don't silently code around it.
