# SurveillanceAI

A real-time AI surveillance prototype that analyzes live webcam feeds and uploaded video files to detect suspicious activity — weapons, aggression, and wanted individuals — and generates timestamped alerts with snapshots.

Built for hackathon **AI-02: Real-Time AI Surveillance System for Suspicious Activity Detection**.

---

## Demo

| Detection | Alert Log | Snapshot Modal |
|---|---|---|
| Live bounding boxes with threat labels | Severity-coded alert cards with timestamps | Click any alert to view the saved snapshot |

**Live feed → threat detected → alert fired → snapshot saved → face matched** — all in under a second.

---

## Features

| | Feature |
|---|---|
| ✅ | Webcam live feed via browser |
| ✅ | Video file upload and processing |
| ✅ | YOLOv8n object detection (weapons, persons) |
| ✅ | Weapon-finetuned YOLO support (drop-in) |
| ✅ | Pose-based aggression detection (YOLOv8-pose) |
| ✅ | Real-time alert generation with timestamps |
| ✅ | Auto-saved alert snapshots |
| ✅ | Severity classification — HIGH / MEDIUM / LOW |
| ✅ | 3-second alert throttle (no spam) |
| ✅ | Criminal face DB matching (ArcFace ONNX) |
| ✅ | Live stats dashboard |
| ✅ | Snapshot modal viewer |
| ✅ | Deployable to Render (HTTPS, webcam-compatible) |

---

## Architecture

```
Browser (Webcam / Video)
        │  JPEG frames over WebSocket (5 FPS)
        ▼
FastAPI + WebSocket  ──►  ThreadPoolExecutor
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             YOLOv8 Inference        ArcFace ONNX
             (weapon / person)    (face embedding match)
                    │                       │
                    └───────────┬───────────┘
                                ▼
                        Alert Manager
                   (throttle → save snapshot → log)
                                │
                        WebSocket response
                                │
                                ▼
                 Frontend renders annotated frame
                 + alert card + stats update
```

The inference pipeline runs entirely in a `ThreadPoolExecutor` — never blocking the asyncio event loop — which keeps the WebSocket responsive at all times.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| WebSocket | Native FastAPI WebSocket |
| Object Detection | YOLOv8n (Ultralytics) |
| Face Detection | OpenCV DNN (SSD MobileNet) |
| Face Embedding | ArcFace R100 (ONNX Runtime) |
| Video Capture | Browser MediaDevices API + Canvas |
| Frontend | Vanilla HTML / CSS / JS |
| Deployment | Render (free tier) |

---

## Project Structure

```
surveillance-ai/
├── backend/
│   ├── main.py              # FastAPI app, WebSocket, REST endpoints
│   ├── detector.py          # YOLOv8 inference + aggression detection
│   ├── alert_manager.py     # Throttled alert saving, stats, severity
│   ├── face_recognizer.py   # ArcFace ONNX face matching pipeline
│   └── models/              # Place weapon_yolov8.pt here (optional)
├── frontend/
│   ├── index.html           # Dashboard layout
│   ├── app.js               # WebSocket client, frame loop, UI logic
│   └── style.css            # Tactical dark UI
├── data/
│   ├── criminal_db/         # One folder per person, one face photo each
│   ├── alerts/              # Auto-saved alert snapshots (runtime)
│   └── sample_videos/       # Drop test videos here
├── requirements.txt
├── pyproject.toml           # uv project config
├── Procfile                 # Render deployment
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://astral.sh/uv): `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Install & Run

```bash
# Install dependencies
uv sync

# (Optional) Pre-download YOLO model to avoid first-run delay
cd backend && uv run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Start server
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** and click **WEBCAM** or upload a video file.

---

## Criminal Face Database

Add one clear, front-facing photo per person:

```
data/criminal_db/
├── john_doe/
│   └── photo.jpg
└── jane_smith/
    └── photo.jpg
```

Folder names become the display name (underscores → spaces, title-cased). Add a corresponding entry to `_MOCK_RECORDS` in `backend/face_recognizer.py` for the crime description shown in alerts.

To enroll without restarting, use the API:

```bash
curl -X POST "http://localhost:8000/api/enroll?name=john_doe" \
     -F "file=@/path/to/photo.jpg"
```

Face models (SSD MobileNet + ArcFace ONNX) auto-download to `data/models/` on first startup.

---

## Weapon Detection

The default `yolov8n.pt` (COCO) reliably detects knives and scissors. For gun detection, drop a weapon-finetuned model at:

```
backend/models/weapon_yolov8.pt
```

Free pretrained weights: search **Roboflow Universe** for `weapon detection yolov8`. The detector auto-selects the finetuned model if the file exists.

To enable pose-based aggression detection (disabled by default to save CPU):

```python
# backend/detector.py
AGGRESSION_ENABLED = True
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/api/alerts` | Get alert log (last 50) |
| `GET` | `/api/stats` | Alert counts by severity |
| `DELETE` | `/api/alerts` | Clear alert log |
| `POST` | `/api/upload-video` | Upload a video file |
| `POST` | `/api/enroll?name=` | Enroll a face photo |
| `POST` | `/api/reload-db` | Rebuild face embedding cache |
| `WS` | `/ws/stream` | Frame stream — send JPEG b64, receive annotated frame + detections |

---

## Deployment (Render)

1. Push this repo to GitHub
2. [render.com](https://render.com) → **New Web Service** → connect repo
3. Configure:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy — Render provides HTTPS automatically, which is required for browser webcam access

> **Note:** Render's free tier has ephemeral storage — alert snapshots and uploaded videos won't persist across deploys. The alert log is in-memory and resets on restart.

---

## Performance Notes

- **Inference speed:** ~200–400ms per frame on CPU (YOLOv8n). The frontend sends at 5 FPS and uses an ack-gated loop — it never queues ahead of the backend.
- **Frame size:** Capped at 640px wide before YOLO inference. Larger frames provide no detection benefit on `yolov8n`.
- **Face matching:** Embedding cache is built once at startup. No disk I/O per match at runtime — pure in-memory cosine distance.
- **GPU:** If a CUDA GPU is available, Ultralytics and ONNX Runtime will use it automatically with no config changes needed.

---

## Tuning

| Parameter | File | Default | Effect |
|---|---|---|---|
| `INFERENCE_WIDTH` | `detector.py` | `640` | Lower = faster, less accurate |
| `AGGRESSION_ENABLED` | `detector.py` | `False` | Enables pose model (~200ms/frame extra) |
| `MIN_FRAME_INTERVAL_MS` | `app.js` | `200` | Raise to reduce CPU load |
| `ALERT_THROTTLE_SECONDS` | `alert_manager.py` | `3.0` | Min seconds between saved alerts |
| `MATCH_THRESHOLD` | `face_recognizer.py` | `0.35` | Lower = stricter face matching |
| `DETECT_CONFIDENCE` | `face_recognizer.py` | `0.70` | Min face detector confidence |

---

## License

MIT
