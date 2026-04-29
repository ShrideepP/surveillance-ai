"""
Face recognition module — matches detected faces against a mock criminal DB.

Criminal DB layout:
  data/criminal_db/
    john_doe/
      photo1.jpg
    jane_smith/
      photo1.jpg
    ...

Each subdirectory name becomes the criminal's display name (underscores → spaces).
Add any .jpg / .png face image into the folder to register a criminal.
"""

import os
import base64
import tempfile
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

CRIMINAL_DB_PATH = Path("data/criminal_db")
CRIMINAL_DB_PATH.mkdir(parents=True, exist_ok=True)

# Lazy import — DeepFace is heavy; only load when first needed
_deepface = None
_db_built = False


def _load_deepface():
    global _deepface
    if _deepface is None:
        try:
            import deepface as df_module
            from deepface import DeepFace
            _deepface = DeepFace
            print("[FaceRecog] DeepFace loaded successfully.")
        except ImportError:
            print("[FaceRecog] DeepFace not installed — facial recognition disabled.")
            _deepface = False
    return _deepface


def is_available() -> bool:
    return bool(_load_deepface())


def db_size() -> int:
    """Return number of registered criminals."""
    if not CRIMINAL_DB_PATH.exists():
        return 0
    return sum(
        1 for p in CRIMINAL_DB_PATH.iterdir()
        if p.is_dir() and any(p.glob("*.jpg")) or any(p.glob("*.png"))
    )


def recognize_faces(frame: np.ndarray) -> Optional[dict]:
    """
    Run face recognition on a frame.
    Returns the first match found, or None.

    Return shape:
        {
            "name": "John Doe",
            "confidence": 0.87,
            "record": "Armed robbery, 2021"
        }
    """
    DeepFace = _load_deepface()
    if not DeepFace:
        return None
    if db_size() == 0:
        return None

    try:
        # Write frame to a temp file (DeepFace needs a path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)

        results = DeepFace.find(
            img_path=tmp_path,
            db_path=str(CRIMINAL_DB_PATH),
            model_name="Facenet",          # lightweight + accurate
            detector_backend="opencv",
            enforce_detection=False,
            silent=True,
        )

        os.unlink(tmp_path)

        # results is a list of DataFrames (one per face found)
        for df in results:
            if df.empty:
                continue
            top = df.iloc[0]
            identity_path = Path(top["identity"])
            # directory name = criminal name
            criminal_name = identity_path.parent.name.replace("_", " ").title()
            distance = float(top.get("distance", 1.0))
            # Convert distance to a 0-1 confidence (lower distance = higher conf)
            confidence = round(max(0.0, 1.0 - distance), 2)

            return {
                "name": criminal_name,
                "confidence": confidence,
                "record": _lookup_record(criminal_name),
            }

    except Exception as e:
        print(f"[FaceRecog] Error: {e}")

    return None


# ---------------------------------------------------------------------------
# Mock criminal records — replace with a real DB in production
# ---------------------------------------------------------------------------
_MOCK_RECORDS: dict[str, str] = {
    "John Doe": "Wanted: Armed robbery (2021), assault (2023)",
    "Jane Smith": "Wanted: Theft, fraud (2022)",
    "Alex Turner": "Wanted: Drug trafficking (2020)",
}


def _lookup_record(name: str) -> str:
    return _MOCK_RECORDS.get(name, "Record on file — contact authorities")


def seed_demo_db():
    """
    Creates placeholder criminal DB entries with blank images so the
    module doesn't crash during demo. Replace images with real face photos.
    """
    demo_criminals = ["john_doe", "jane_smith", "alex_turner"]
    for name in demo_criminals:
        folder = CRIMINAL_DB_PATH / name
        folder.mkdir(exist_ok=True)
        placeholder = folder / "photo1.jpg"
        if not placeholder.exists():
            # 100x100 grey placeholder — swap with real face photo
            img = np.full((100, 100, 3), 180, dtype=np.uint8)
            cv2.putText(img, name[:4], (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
            cv2.imwrite(str(placeholder), img)
    print(f"[FaceRecog] Demo DB seeded with {len(demo_criminals)} entries in {CRIMINAL_DB_PATH}")