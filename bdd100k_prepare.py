"""
bdd100k_prepare.py
==================
Converts BDD100K (marquis03/bdd100k structure) → YOLO format
with proper train / val / test splits.

Your confirmed structure:
  data/bdd100k_raw/
  ├── train/
  │   ├── images/   (70,000 .jpg)
  │   └── annotations/bdd100k_labels_images_train.json
  ├── val/
  │   ├── images/   (10,000 .jpg)
  │   └── annotations/bdd100k_labels_images_val.json
  └── test/         (20,000 .jpg flat, NO labels)

Split strategy:
  train  → 70,000 images  (official train, with labels)
  val    →  8,000 images  (80% of official val, for YOLO training monitor)
  test   →  2,000 images  (20% of official val, HELD-OUT for paper results)
  Note: official test has NO public labels, so we carve our test from labeled val

Output:
  data/bdd100k_yolo/
  ├── images/train/  (70,000 symlinks)
  ├── images/val/    ( 8,000 symlinks)
  ├── images/test/   ( 2,000 symlinks)
  ├── labels/train/  (70,000 .txt)
  ├── labels/val/    ( 8,000 .txt)
  ├── labels/test/   ( 2,000 .txt)
  └── dataset.yaml

BDD100K → YOLO class mapping (10 classes):
  0 pedestrian   1 rider        2 car           3 truck
  4 bus          5 train        6 motorcycle    7 bicycle
  8 traffic light               9 traffic sign
"""

import json
import os
import random
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

# ── Paths (hardcoded to your exact structure) ──────────────────────────────────
ROOT     = Path(__file__).parent
RAW_DIR  = ROOT / "data" / "bdd100k_raw"
OUT_DIR  = ROOT / "data" / "bdd100k_yolo"

TRAIN_IMG_DIR  = RAW_DIR / "train" / "images"
TRAIN_JSON     = RAW_DIR / "train" / "annotations" / "bdd100k_labels_images_train.json"
VAL_IMG_DIR    = RAW_DIR / "val"   / "images"
VAL_JSON       = RAW_DIR / "val"   / "annotations" / "bdd100k_labels_images_val.json"
TEST_IMG_DIR   = RAW_DIR / "test"                    # flat folder, no labels

# ── Split config ───────────────────────────────────────────────────────────────
VAL_FRACTION  = 0.80    # 80% of official val → YOLO val
TEST_FRACTION = 0.20    # 20% of official val → YOLO test (held-out)
RANDOM_SEED   = 42      # fixed for reproducibility

# ── Sensor image dimensions ────────────────────────────────────────────────────
IMG_W = 1280
IMG_H = 720

# ── Class mapping ──────────────────────────────────────────────────────────────
CLASS_MAP = {
    "pedestrian":    0,
    "rider":         1,
    "car":           2,
    "truck":         3,
    "bus":           4,
    "train":         5,
    "motorcycle":    6,
    "bicycle":       7,
    "traffic light": 8,
    "traffic sign":  9,
}
NAMES = list(CLASS_MAP.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversion
# ─────────────────────────────────────────────────────────────────────────────

def box2d_to_yolo(x1: float, y1: float,
                  x2: float, y2: float) -> tuple:
    """
    BDD100K absolute pixel box2d → YOLO normalized (cx, cy, w, h).
    Clamps to [0.001, 0.999] to handle rare out-of-bound annotations.
    """
    cx = ((x1 + x2) / 2.0) / IMG_W
    cy = ((y1 + y2) / 2.0) / IMG_H
    bw = (x2 - x1) / IMG_W
    bh = (y2 - y1) / IMG_H
    cx = max(0.001, min(0.999, cx))
    cy = max(0.001, min(0.999, cy))
    bw = max(0.001, min(0.999, bw))
    bh = max(0.001, min(0.999, bh))
    return cx, cy, bw, bh


# ─────────────────────────────────────────────────────────────────────────────
# JSON → YOLO conversion
# ─────────────────────────────────────────────────────────────────────────────

def parse_json(json_path: Path, img_dir: Path) -> tuple:
    """
    Parse one BDD100K JSON file.

    Returns:
        entries : list of (img_stem, src_img_path, [yolo_label_lines])
        stats   : dict with counts per class and totals
    """
    print(f"\n  [→] Reading: {json_path.name}  "
          f"({json_path.stat().st_size / 1e6:.1f} MB)")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    stats = {
        "total_images":       0,
        "images_with_labels": 0,
        "images_no_labels":   0,
        "total_boxes":        0,
        "skipped_unknown":    0,
        "skipped_crowd":      0,
        "skipped_degenerate": 0,
        "class_counts":       {c: 0 for c in CLASS_MAP},
    }

    entries = []

    for entry in tqdm(data, desc=f"  Parsing {json_path.stem[-5:]}"):
        img_name = entry["name"]
        src_img  = img_dir / img_name

        # Skip if image file is missing (should not happen, but safety check)
        if not src_img.exists():
            continue

        stem       = Path(img_name).stem
        raw_labels = entry.get("labels") or []
        yolo_lines = []

        for lbl in raw_labels:
            category = lbl.get("category", "")

            # Skip unknown categories (drivable area, lane, etc.)
            if category not in CLASS_MAP:
                stats["skipped_unknown"] += 1
                continue

            # box2d is None for non-bounding-box annotations
            box2d = lbl.get("box2d")
            if box2d is None:
                stats["skipped_unknown"] += 1
                continue

            # Skip crowd annotations (merged, unreliable boxes)
            if lbl.get("attributes", {}).get("crowd", False):
                stats["skipped_crowd"] += 1
                continue

            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])

            # Skip degenerate boxes smaller than 2×2 pixels
            if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
                stats["skipped_degenerate"] += 1
                continue

            cls_id = CLASS_MAP[category]
            cx, cy, bw, bh = box2d_to_yolo(x1, y1, x2, y2)
            yolo_lines.append(
                f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )
            stats["total_boxes"]                  += 1
            stats["class_counts"][category]       += 1

        entries.append((stem, src_img, yolo_lines))
        stats["total_images"] += 1

        if yolo_lines:
            stats["images_with_labels"] += 1
        else:
            stats["images_no_labels"]   += 1

    return entries, stats


# ─────────────────────────────────────────────────────────────────────────────
# Write split to disk
# ─────────────────────────────────────────────────────────────────────────────

def write_split(entries:     list,
                split_name:  str,
                use_symlinks: bool = True) -> None:
    """
    Write images and label .txt files for one split.
    Skips files that already exist (safe to re-run).
    """
    img_dir = OUT_DIR / "images" / split_name
    lbl_dir = OUT_DIR / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for stem, src_img, yolo_lines in tqdm(entries,
                                           desc=f"  Writing {split_name:<6}"):
        # ── Image: symlink (saves ~5 GB disk) or copy ──────────────────────
        dst_img = img_dir / src_img.name
        if not dst_img.exists():
            if use_symlinks:
                os.symlink(src_img.resolve(), dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        # ── Label: write YOLO .txt (empty file = background-only frame) ────
        dst_lbl = lbl_dir / (stem + ".txt")
        if not dst_lbl.exists():
            with open(dst_lbl, "w") as f:
                f.write("\n".join(yolo_lines))


# ─────────────────────────────────────────────────────────────────────────────
# dataset.yaml
# ─────────────────────────────────────────────────────────────────────────────

def write_dataset_yaml(n_train: int, n_val: int, n_test: int) -> None:
    content = {
        "path":  str(OUT_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_MAP),
        "names": NAMES,
    }
    out = OUT_DIR / "dataset.yaml"
    with open(out, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    print(f"\n[✓] Saved dataset.yaml")
    print(f"    train : {n_train:>7,} images")
    print(f"    val   : {n_val:>7,} images  (used during YOLO training)")
    print(f"    test  : {n_test:>7,} images  (held-out, for paper results)")


# ─────────────────────────────────────────────────────────────────────────────
# Stats display
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(stats: dict, split_name: str) -> None:
    b = stats["total_boxes"]
    print(f"\n  ── {split_name.upper()} ──────────────────────────────────")
    print(f"  Images parsed        : {stats['total_images']:>8,}")
    print(f"  Images with objects  : {stats['images_with_labels']:>8,}")
    print(f"  Background frames    : {stats['images_no_labels']:>8,}  "
          f"← these save energy in adaptive mode")
    print(f"  Total boxes kept     : {b:>8,}")
    print(f"  Skipped (unknown)    : {stats['skipped_unknown']:>8,}")
    print(f"  Skipped (crowd)      : {stats['skipped_crowd']:>8,}")
    print(f"  Skipped (degenerate) : {stats['skipped_degenerate']:>8,}")
    if b > 0:
        print(f"\n  Class distribution:")
        for cls_name, count in sorted(stats["class_counts"].items(),
                                      key=lambda x: -x[1]):
            if count == 0:
                continue
            pct = 100.0 * count / b
            bar = "█" * int(pct / 2)
            print(f"    {cls_name:<16} {count:>8,}  ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(use_symlinks: bool = True) -> None:
    print("=" * 65)
    print("  BDD100K → YOLO Conversion  (train / val / test)")
    print("=" * 65)

    # ── Verify raw files exist ─────────────────────────────────────────────────
    missing = []
    for p in [TRAIN_IMG_DIR, TRAIN_JSON, VAL_IMG_DIR, VAL_JSON]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("\n[ERROR] Missing raw files:")
        for m in missing:
            print(f"  ✗ {m}")
        print("\nExpected structure:")
        print("  data/bdd100k_raw/train/images/               (70K jpg)")
        print("  data/bdd100k_raw/train/annotations/*.json")
        print("  data/bdd100k_raw/val/images/                 (10K jpg)")
        print("  data/bdd100k_raw/val/annotations/*.json")
        return

    # ── Skip if already done ───────────────────────────────────────────────────
    if (OUT_DIR / "dataset.yaml").exists():
        n_tr = len(list((OUT_DIR / "images" / "train").glob("*.jpg")))
        n_v  = len(list((OUT_DIR / "images" / "val").glob("*.jpg")))
        n_te = len(list((OUT_DIR / "images" / "test").glob("*.jpg")))
        if n_tr > 0 and n_v > 0 and n_te > 0:
            print(f"\n[SKIP] Already converted:")
            print(f"       train={n_tr:,}  val={n_v:,}  test={n_te:,}")
            print(f"       Delete {OUT_DIR} to redo.")
            return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Parse train JSON ───────────────────────────────────────────────────────
    print("\n[1/5] Parsing train JSON (70K images)...")
    train_entries, train_stats = parse_json(TRAIN_JSON, TRAIN_IMG_DIR)
    print_stats(train_stats, "train")

    # ── Parse val JSON ─────────────────────────────────────────────────────────
    print("\n[2/5] Parsing val JSON (10K images)...")
    val_entries_all, val_stats = parse_json(VAL_JSON, VAL_IMG_DIR)
    print_stats(val_stats, "val (before split)")

    # ── Split val → val (80%) + test (20%) ────────────────────────────────────
    print(f"\n[3/5] Splitting val → {VAL_FRACTION*100:.0f}% val "
          f"+ {TEST_FRACTION*100:.0f}% test  (seed={RANDOM_SEED})...")

    rng = random.Random(RANDOM_SEED)
    shuffled = val_entries_all.copy()
    rng.shuffle(shuffled)

    n_val       = int(len(shuffled) * VAL_FRACTION)
    val_entries  = shuffled[:n_val]
    test_entries = shuffled[n_val:]

    print(f"    val  : {len(val_entries):,} images")
    print(f"    test : {len(test_entries):,} images  ← held-out, never seen during training")

    # ── Write splits ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Writing files (use_symlinks={use_symlinks})...")
    write_split(train_entries,  "train", use_symlinks)
    write_split(val_entries,    "val",   use_symlinks)
    write_split(test_entries,   "test",  use_symlinks)

    # ── Write dataset.yaml ─────────────────────────────────────────────────────
    print("\n[5/5] Writing dataset.yaml...")
    write_dataset_yaml(len(train_entries),
                       len(val_entries),
                       len(test_entries))

    # ── Final summary ──────────────────────────────────────────────────────────
    total = (len(train_entries) + len(val_entries) + len(test_entries))
    print(f"\n{'='*65}")
    print(f"  Done!  {total:,} images total")
    print(f"  Output: {OUT_DIR}")
    print(f"\n  Next steps:")
    print(f"    1. Verify config.py has  DATASET = 'bdd100k'")
    print(f"    2. python step1_generate_lr_datasets.py")
    print(f"    3. python step1_train_all.py")
    print(f"    4. python step2_evaluate.py   (evaluates on test split)")
    print(f"{'='*65}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert BDD100K to YOLO format with train/val/test splits"
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy images instead of symlinking "
             "(use on Windows or if symlinks cause issues)"
    )
    args = parser.parse_args()
    main(use_symlinks=not args.copy)
