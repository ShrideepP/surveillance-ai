import cv2
import json
import base64
import asyncio
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from detector import SurveillanceDetector
from alert_manager import save_alert, get_alerts, get_stats, should_trigger_alert, should_trigger_face_alert
from face_recognizer import recognize_faces, is_available as face_recog_available, seed_demo_db, build_db

_inference_pool = ThreadPoolExecutor(max_workers=1)
detector: SurveillanceDetector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    print("[Main] Loading detection model...")
    loop = asyncio.get_event_loop()
    detector = await loop.run_in_executor(_inference_pool, SurveillanceDetector)
    seed_demo_db()
    build_db()          # ← was missing; populates _db before any frame arrives
    print("[Main] Ready.")
    yield
    _inference_pool.shutdown(wait=False)
    print("[Main] Shutting down.")


app = FastAPI(title="SurveillanceAI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
ALERTS_DIR   = Path("data/alerts")
VIDEOS_DIR   = Path("data/sample_videos")
ALERTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static",    StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/snapshots", StaticFiles(directory=str(ALERTS_DIR)),   name="snapshots")


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text())

@app.get("/api/alerts")
async def alerts_endpoint():
    return JSONResponse(get_alerts())

@app.get("/api/stats")
async def stats_endpoint():
    return JSONResponse({**get_stats(), "face_recog_enabled": face_recog_available()})

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    allowed = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    if Path(file.filename).suffix.lower() not in allowed:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)
    contents = await file.read()
    (VIDEOS_DIR / file.filename).write_bytes(contents)
    return JSONResponse({"filename": file.filename, "size": len(contents)})

@app.delete("/api/alerts")
async def clear_alerts():
    from alert_manager import alerts_log
    alerts_log.clear()
    return JSONResponse({"cleared": True})


def _run_inference(frame: np.ndarray, base_url: str) -> dict:
    # ── Path A: weapon / aggression detection ──────────────────────────────
    result = detector.detect(frame)

    if not result["alert"]:
        try:
            if detector.detect_aggression(frame):
                result["alert"] = True
                result["detections"].append({
                    "label": "aggression",
                    "confidence": 0.70,
                    "suspicious": True,
                    "bbox": [],
                })
        except Exception as exc:
            print(f"[Aggression] {exc}")

    if result["alert"] and should_trigger_alert():
        # Face recog runs here too when a weapon is present
        face_match = None
        if face_recog_available():
            try:
                face_match = recognize_faces(frame)
            except Exception as exc:
                print(f"[FaceRecog] {exc}")
        alert = save_alert(result["frame"], result["detections"], face_match, base_url)
        result["alert_data"] = alert

    # ── Path B: face-only alert (independent of weapon detection) ──────────
    # Runs on every frame regardless of weapon/aggression result.
    # Has its own throttle so it doesn't consume the weapon-alert slot.
    if face_recog_available():
        try:
            face_match = recognize_faces(frame)
            if face_match and should_trigger_face_alert():
                print(f"[Main] Face alert triggered: {face_match['name']}")
                # Synthesise a face-match detection entry for the alert log
                face_detection = [{
                    "label": f"face:{face_match['name']}",
                    "confidence": face_match["confidence"],
                    "suspicious": True,
                    "bbox": [],
                }]
                alert = save_alert(
                    result["frame"], face_detection, face_match, base_url
                )
                # Surface alert to frontend even if no weapon was detected
                result["alert"] = True
                result["alert_data"] = alert
                result["detections"].extend(face_detection)
        except Exception as exc:
            print(f"[FaceRecog/PathB] {exc}")

    return result


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    scheme   = "https" if websocket.url.scheme in ("wss", "https") else "http"
    base_url = f"{scheme}://{websocket.url.netloc}"

    print(f"[WS] Client connected: {websocket.client}")

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            frame_b64 = payload.get("frame")
            if not frame_b64:
                continue

            try:
                img_bytes = base64.b64decode(frame_b64)
                np_arr    = np.frombuffer(img_bytes, np.uint8)
                frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as exc:
                await websocket.send_text(json.dumps({"error": f"Decode failed: {exc}"}))
                continue

            if frame is None:
                continue

            result = await loop.run_in_executor(
                _inference_pool, _run_inference, frame, base_url
            )

            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[WS] Unexpected error: {exc}")
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass
    finally:
        print(f"[WS] Client disconnected: {websocket.client}")