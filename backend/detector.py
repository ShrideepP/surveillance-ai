import cv2
import numpy as np
import base64
from ultralytics import YOLO
from pathlib import Path

# ---------------------------------------------------------------------------
# COCO class IDs that we treat as suspicious out-of-the-box
# YOLOv8n is trained on COCO-80; it has no "gun" class, but does have:
#   43 = knife, 76 = scissors
# If you drop a weapon-finetuned .pt into backend/models/, it takes priority.
# ---------------------------------------------------------------------------
COCO_SUSPICIOUS: dict[int, str] = {
    43: "knife",
    76: "scissors",
}

# ---------------------------------------------------------------------------
# CHANGE 1: Weapon label whitelist for the fine-tuned model.
# The jubaerad dataset uses a single class "weapon".
# Lowercase variants + synonyms act as a safety net against casing mismatches.
# ---------------------------------------------------------------------------
WEAPON_LABELS: set[str] = {
    "weapon",                                        # jubaerad dataset class
    "gun", "handgun", "pistol", "rifle", "firearm",  # synonyms / other models
    "knife", "blade",                                # knife-class models
}

WEAPON_MODEL_PATH = Path("models/weapon_yolov8.pt")
BASE_MODEL_PATH   = "yolov8n.pt"

# Max width passed to YOLO — smaller = faster, 640 is the sweet spot
INFERENCE_WIDTH = 640

# Pose-based aggression adds ~150-400ms per frame on CPU.
# Keep False for demo unless you have a GPU.
AGGRESSION_ENABLED = False


class SurveillanceDetector:
    def __init__(self):
        if WEAPON_MODEL_PATH.exists():
            print(f"[Detector] Loading weapon-finetuned model: {WEAPON_MODEL_PATH}")
            self.model = YOLO(str(WEAPON_MODEL_PATH))
            self.use_weapon_model = True
        else:
            print(f"[Detector] Weapon model not found — using {BASE_MODEL_PATH} (COCO)")
            self.model = YOLO(BASE_MODEL_PATH)
            self.use_weapon_model = False

        # CHANGE 2: Split confidence thresholds — weapon model runs permissive
        # (catches more at demo time), COCO runs strict (reduces phone/TV noise).
        # Old code used a single self.conf_threshold = 0.40 for both.
        self.conf_weapon = 0.45   # permissive — single-class models can be conservative
        self.conf_coco   = 0.55   # strict — suppresses common false positives

        # CHANGE 3: NMS IoU threshold lowered from default 0.7 → 0.45.
        # Merges overlapping weapon boxes that YOLO sometimes splits across
        # adjacent frames, giving cleaner single-box outputs.
        self.iou_threshold = 0.45

        self._pose_model = None   # lazy-loaded only when AGGRESSION_ENABLED

        # Warm-up pass eliminates the ~1s first-frame latency spike.
        print("[Detector] Running warm-up inference...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[Detector] Warm-up complete.")

    # ------------------------------------------------------------------
    # Primary entry point — called from the thread pool, never event loop
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> dict:
        # Ensure frame is capped at inference width before YOLO sees it
        h, w = frame.shape[:2]
        if w > INFERENCE_WIDTH:
            scale = INFERENCE_WIDTH / w
            frame = cv2.resize(frame, (INFERENCE_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_LINEAR)

        # CHANGE 4: Pass per-model conf + iou + agnostic_nms to YOLO call.
        # agnostic_nms=True applies NMS across all classes — better for a
        # single-class weapon model where class distinction is irrelevant.
        conf = self.conf_weapon if self.use_weapon_model else self.conf_coco
        results = self.model(
            frame,
            conf=conf,
            iou=self.iou_threshold,
            agnostic_nms=True,
            verbose=False,
        )[0]

        detections = []
        alert_triggered = False
        annotated = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.names[cls_id]

            if self.use_weapon_model:
                # CHANGE 5: Double-gate — model class AND label whitelist.
                # Old code flagged every detection from the weapon model as
                # suspicious (is_suspicious = True always). The whitelist gate
                # blocks misfires on background objects the model occasionally
                # confuses with weapons (e.g., L-shaped tools, dark objects).
                is_suspicious = label.lower() in {w.lower() for w in WEAPON_LABELS}
            else:
                label = self.model.names[cls_id]
                is_suspicious = cls_id in COCO_SUSPICIOUS

            if is_suspicious:
                alert_triggered = True
                color = (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                _draw_label(annotated, f"⚠ {label.upper()} {conf_val:.2f}", x1, y1, color)
            else:
                # CHANGE 6: Non-suspicious objects are silently skipped on
                # the weapon model — no green boxes drawn.
                # On COCO fallback, safe objects are still rendered (useful
                # for debugging what the model sees).
                if not self.use_weapon_model:
                    color = (0, 210, 90)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                    _draw_label(annotated, f"{label} {conf_val:.2f}", x1, y1, color)

            detections.append(
                {
                    "label": label,
                    "confidence": round(conf_val, 2),
                    "suspicious": is_suspicious,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        # Overlay status banner
        _draw_status_banner(annotated, alert_triggered, len(detections))

        # Encode to JPEG → base64
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
        frame_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "frame": frame_b64,
            "detections": detections,
            "alert": alert_triggered,
        }

    # ------------------------------------------------------------------
    # Optional: fight / aggression detection via pose estimation
    # Returns True if aggressive posture heuristic fires
    # ------------------------------------------------------------------
    def detect_aggression(self, frame: np.ndarray) -> bool:
        if not AGGRESSION_ENABLED:
            return False
        try:
            if self._pose_model is None:
                self._pose_model = YOLO("yolov8n-pose.pt")

            results = self._pose_model(frame, verbose=False)[0]
            if results.keypoints is None:
                return False

            for kps in results.keypoints.xy:
                if _is_aggressive_pose(kps.cpu().numpy()):
                    return True
        except Exception as e:
            print(f"[Aggression] Skipping: {e}")
        return False


# ---------------------------------------------------------------------------
# Helpers — unchanged
# ---------------------------------------------------------------------------

def _draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y0 = max(y - 6, th + 4)
    cv2.rectangle(img, (x, y0 - th - 4), (x + tw + 4, y0 + 2), color, -1)
    cv2.putText(
        img, text, (x + 2, y0 - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


def _draw_status_banner(img, alert: bool, det_count: int):
    h, w = img.shape[:2]
    banner_h = 28
    overlay = img.copy()
    color = (0, 0, 180) if alert else (20, 20, 20)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    status = "⚠  THREAT DETECTED" if alert else "✔  MONITORING"
    text = f"{status}   |   Objects: {det_count}"
    cv2.putText(
        img, text, (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


def _is_aggressive_pose(keypoints: np.ndarray) -> bool:
    """
    Heuristic: both wrists raised above shoulders → flagged as aggressive.
    Keypoint indices (COCO): 5=L-shoulder, 6=R-shoulder, 9=L-wrist, 10=R-wrist
    """
    if keypoints.shape[0] < 11:
        return False
    try:
        l_shoulder_y = keypoints[5][1]
        r_shoulder_y = keypoints[6][1]
        l_wrist_y = keypoints[9][1]
        r_wrist_y = keypoints[10][1]

        # In image coords, smaller y = higher on screen
        l_raised = l_wrist_y < l_shoulder_y and l_shoulder_y > 0
        r_raised = r_wrist_y < r_shoulder_y and r_shoulder_y > 0
        return l_raised and r_raised
    except Exception:
        return False