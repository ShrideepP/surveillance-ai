import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ALERTS_DIR = Path("data/alerts")
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

alerts_log: list[dict] = []
_last_alert_time: float = 0.0
ALERT_THROTTLE_SECONDS = 3.0


def should_trigger_alert() -> bool:
    global _last_alert_time
    now = time.time()
    if now - _last_alert_time >= ALERT_THROTTLE_SECONDS:
        _last_alert_time = now
        return True
    return False


def save_alert(
    frame_b64: str,
    detections: list,
    face_match: Optional[dict] = None,
    base_url: str = "",
) -> dict:
    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")

    # Save snapshot to disk
    img_filename = f"alert_{ts_str}.jpg"
    img_path = ALERTS_DIR / img_filename
    img_bytes = base64.b64decode(frame_b64)
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    suspicious_detections = [d for d in detections if d.get("suspicious")]

    severity = _compute_severity(suspicious_detections)

    alert = {
        "id": ts_str,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "detections": suspicious_detections,
        "snapshot_url": f"{base_url}/snapshots/{img_filename}",
        "face_match": face_match,
        "severity": severity,
    }
    alerts_log.append(alert)
    return alert


def _compute_severity(detections: list) -> str:
    labels = [d.get("label", "").lower() for d in detections]
    high_risk = {"gun", "pistol", "rifle", "weapon", "knife", "blade"}
    if any(label in high_risk for label in labels):
        return "HIGH"
    if len(detections) >= 2:
        return "MEDIUM"
    return "LOW"


def get_alerts(limit: int = 50) -> list:
    return list(reversed(alerts_log[-limit:]))


def get_stats() -> dict:
    total = len(alerts_log)
    by_severity = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for a in alerts_log:
        sev = a.get("severity", "LOW")
        by_severity[sev] = by_severity.get(sev, 0) + 1

    all_detections = [d for a in alerts_log for d in a.get("detections", [])]
    label_counts: dict = {}
    for d in all_detections:
        label = d.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "total_alerts": total,
        "by_severity": by_severity,
        "top_detections": sorted(label_counts.items(), key=lambda x: -x[1])[:5],
    }