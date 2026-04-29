# scripts/prepare_dataset.py
import os, shutil, random
from pathlib import Path

RAW       = Path("data/raw")           # adjust if unzip created a subfolder
OUT       = Path("data/weapon_dataset")
VAL_SPLIT = 0.15                       # 15% val, rest train
SEED      = 42

random.seed(SEED)

for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels"  / split).mkdir(parents=True, exist_ok=True)

# Collect all images that have a corresponding label file
pairs = []
for img_path in RAW.rglob("*.jpg"):
    label_path = img_path.with_suffix(".txt")
    # also check parallel labels/ folder if dataset separates images and labels
    alt_label = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
    lbl = label_path if label_path.exists() else (alt_label if alt_label.exists() else None)
    if lbl:
        pairs.append((img_path, lbl))

print(f"Found {len(pairs)} labeled image pairs")

random.shuffle(pairs)
val_n = int(len(pairs) * VAL_SPLIT)
splits = {"val": pairs[:val_n], "train": pairs[val_n:]}

for split, items in splits.items():
    for img, lbl in items:
        shutil.copy(img, OUT / "images" / split / img.name)
        shutil.copy(lbl, OUT / "labels"  / split / lbl.name)

print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}")