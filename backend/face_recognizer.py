"""
Face recognition module — matches detected faces against a mock criminal DB.

Pipeline:
  1. OpenCV DNN face detector (res10 SSD) locates faces — fast, angle-tolerant
  2. Each crop upscaled to ≥120px, passed to face_recognition for 128-d encoding
  3. Euclidean distance vs DB encodings, threshold 0.6

Debug logging is ON by default — every frame logs its outcome so you can
follow the pipeline in the terminal and see exactly where a miss occurs.
"""

import urllib.request
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

CRIMINAL_DB_PATH = Path("data/criminal_db")
CRIMINAL_DB_PATH.mkdir(parents=True, exist_ok=True)

MATCH_THRESHOLD = 0.6   # raise to 0.65 if you see distances just above 0.60
MIN_FACE_PX     = 80
DETECT_CONF     = 0.5

_DNN_PROTO = Path("models/deploy.prototxt")
_DNN_MODEL = Path("models/res10_300x300_ssd_iter_140000.caffemodel")
_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

_fr  = None
_net = None
_db: list[dict] = []


def _load_fr():
    global _fr
    if _fr is None:
        try:
            import face_recognition as fr_module
            _fr = fr_module
            print("[FaceRecog] face_recognition loaded.")
        except ImportError:
            print("[FaceRecog] face_recognition not installed — disabled.")
            _fr = False
    return _fr


def _load_dnn_detector():
    global _net
    if _net is not None:
        return _net
    _DNN_PROTO.parent.mkdir(parents=True, exist_ok=True)
    if not _DNN_PROTO.exists():
        print("[FaceRecog] Downloading deploy.prototxt...")
        urllib.request.urlretrieve(_PROTO_URL, _DNN_PROTO)
    if not _DNN_MODEL.exists():
        print("[FaceRecog] Downloading res10 SSD weights (~10 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _DNN_MODEL)
    _net = cv2.dnn.readNetFromCaffe(str(_DNN_PROTO), str(_DNN_MODEL))
    print("[FaceRecog] DNN face detector ready.")
    return _net


def is_available() -> bool:
    return bool(_load_fr())


def db_size() -> int:
    return len(_db)


def _detect_face_crops(img_bgr: np.ndarray, context: str = "") -> list[np.ndarray]:
    """Detect faces using res10 SSD. Returns upscaled BGR crops."""
    net = _load_dnn_detector()
    h, w = img_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    dets = net.forward()

    crops = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < DETECT_CONF:
            continue
        x1 = max(0, int(dets[0, 0, i, 3] * w))
        y1 = max(0, int(dets[0, 0, i, 4] * h))
        x2 = min(w, int(dets[0, 0, i, 5] * w))
        y2 = min(h, int(dets[0, 0, i, 6] * h))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img_bgr[y1:y2, x1:x2]
        ch = crop.shape[0]
        if ch < MIN_FACE_PX:
            scale = MIN_FACE_PX / ch
            crop = cv2.resize(crop, (0, 0), fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        crops.append(crop)

    print(f"[FaceRecog][{context}] Frame {w}x{h} → {len(crops)} face crop(s) detected")
    return crops


def build_db():
    """Load encodings from criminal_db/ at startup. Skips non-face images."""
    global _db
    fr = _load_fr()
    if not fr:
        return

    _load_dnn_detector()
    _db = []

    if not CRIMINAL_DB_PATH.exists():
        print("[FaceRecog] criminal_db/ not found — DB empty")
        return

    for person_dir in sorted(CRIMINAL_DB_PATH.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name.replace("_", " ").title()
        encodings_found = 0

        for img_path in sorted(person_dir.glob("*.jpg")):
            if img_path.stem.startswith("_"):
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[FaceRecog][build_db] Cannot read {img_path} — skipping")
                continue

            crops = _detect_face_crops(img_bgr, context=f"build_db/{name}")
            if not crops:
                print(f"[FaceRecog][build_db] WARNING: No face in '{img_path.name}' "
                      f"for '{name}' — use a clearer photo with face filling >25% of frame")
                continue

            for crop in crops:
                encs = fr.face_encodings(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if encs:
                    _db.append({"name": name, "encoding": encs[0]})
                    encodings_found += 1
                    print(f"[FaceRecog][build_db] ✓ Encoded '{name}' from {img_path.name}")
                else:
                    print(f"[FaceRecog][build_db] face_encodings() returned [] for crop "
                          f"from {img_path.name} — crop may still be too small/blurry")

        if encodings_found == 0:
            print(f"[FaceRecog][build_db] ✗ '{name}' has 0 encodings — will NEVER match")
        else:
            print(f"[FaceRecog][build_db] ✓ '{name}' ready with {encodings_found} encoding(s)")

    print(f"[FaceRecog] DB built — {len(_db)} total encoding(s), "
          f"{len({e['name'] for e in _db})} person(s)")


def recognize_faces(frame: np.ndarray) -> Optional[dict]:
    fr = _load_fr()
    if not fr:
        print("[FaceRecog][recognize] Skipping — library not available")
        return None
    if not _db:
        print("[FaceRecog][recognize] Skipping — DB is empty (build_db() may not have run)")
        return None

    crops = _detect_face_crops(frame, context="recognize")
    if not crops:
        # Logged inside _detect_face_crops already
        return None

    db_encodings = [e["encoding"] for e in _db]
    db_names     = [e["name"]     for e in _db]

    best_name = None
    best_dist = float("inf")

    for idx_crop, crop in enumerate(crops):
        query_encs = fr.face_encodings(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not query_encs:
            print(f"[FaceRecog][recognize] Crop {idx_crop}: face_encodings() returned [] "
                  f"— crop size {crop.shape[1]}x{crop.shape[0]}")
            continue

        distances  = fr.face_distance(db_encodings, query_encs[0])
        best_idx   = int(np.argmin(distances))
        dist       = float(distances[best_idx])
        candidate  = db_names[best_idx]

        print(f"[FaceRecog][recognize] Crop {idx_crop}: "
              f"closest='{candidate}' dist={dist:.3f} threshold={MATCH_THRESHOLD} "
              f"→ {'MATCH' if dist <= MATCH_THRESHOLD else 'no match'}")

        if dist < best_dist:
            best_dist = dist
            best_name = candidate

    if best_name is None:
        print("[FaceRecog][recognize] No encodings extracted from any crop")
        return None

    if best_dist > MATCH_THRESHOLD:
        print(f"[FaceRecog][recognize] Best dist {best_dist:.3f} exceeds threshold — no alert")
        return None

    confidence = round(1.0 - best_dist, 2)
    print(f"[FaceRecog][recognize] ✓ MATCHED '{best_name}' "
          f"dist={best_dist:.3f} confidence={confidence}")
    return {
        "name": best_name,
        "confidence": confidence,
        "record": _lookup_record(best_name),
    }


_MOCK_RECORDS: dict[str, str] = {
    "Alex Turner": "Wanted: Drug trafficking (2020)",
}


def _lookup_record(name: str) -> str:
    return _MOCK_RECORDS.get(name, "Record on file — contact authorities")


def seed_demo_db():
    for name in ["john_doe", "jane_smith", "alex_turner"]:
        folder = CRIMINAL_DB_PATH / name
        folder.mkdir(exist_ok=True)
        (folder / "_placeholder").touch()
    print("[FaceRecog] Demo DB folders ready. Drop face photos in to enable matching.")