"""
coco_prepare.py
===============
Downloads COCO 2017 images (person + car classes only) and converts
to YOLO format with train / val / test splits.

Strategy:
  Source:  COCO train2017 (118K images) → filter person+car → ~7,700 images
           COCO val2017   (  5K images) → filter person+car → ~2,700 images

  Output splits:
    train : ~6,200 images  (80% of filtered train2017)
    val   : ~1,500 images  (20% of filtered train2017)  ← used during YOLO training
    test  : ~2,700 images  (all filtered val2017)       ← held-out for paper results

  Why use val2017 as test:
    val2017 is the standard COCO benchmark split — using it as test lets
    reviewers directly compare your numbers to published YOLO baselines.

Classes (2):
  0 : person
  1 : car

Image sizes in COCO: variable, typically 640×480 or 480×640 landscape/portrait.
YOLO will letterbox all images to imgsz=640 during training automatically.

Requirements:
  pip install pycocotools requests tqdm pyyaml
"""

import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR  = DATA_DIR / "coco_raw"
OUT_DIR  = DATA_DIR / "coco_person_car_yolo"

# ── Classes to keep ────────────────────────────────────────────────────────────
# COCO category names → your YOLO class IDs
KEEP_CLASSES = {
    "person": 0,
    "car":    1,
}
NAMES = list(KEEP_CLASSES.keys())

# ── Split config ───────────────────────────────────────────────────────────────
TRAIN_VAL_SPLIT = 0.80    # 80% filtered train2017 → YOLO train
                           # 20% filtered train2017 → YOLO val
RANDOM_SEED     = 42

# ── COCO download URLs ─────────────────────────────────────────────────────────
COCO_URLS = {
    "val2017_images":   "http://images.cocodataset.org/zips/val2017.zip",
    "train2017_images": "http://images.cocodataset.org/zips/train2017.zip",
    "annotations":      "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> None:
    """Download with progress bar. Skips if file already exists."""
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [↓] Downloading {dest.name}  ({url})")
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024,
               miniters=1, desc=f"  {dest.name}") as t:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                t.total = total_size
            t.update(block_size)
        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print(f"  [✓] Saved: {dest}")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract zip. Skips if destination already populated."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"  [SKIP] Already extracted: {dest_dir.name}/")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [→] Extracting {zip_path.name} → {dest_dir}/")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print(f"  [✓] Extracted.")


def download_coco(skip_train: bool = False) -> None:
    """
    Download COCO val2017, train2017 images, and annotations.
    Set skip_train=True to only download val2017 (~800MB instead of ~19GB).
    For paper quality results, set skip_train=False (full download).
    """
    zips_dir = RAW_DIR / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    # ── Annotations (~241 MB, always needed) ────────────────────────────────
    ann_zip = zips_dir / "annotations_trainval2017.zip"
    download_file(COCO_URLS["annotations"], ann_zip)
    extract_zip(ann_zip, RAW_DIR / "annotations_raw")

    # ── Val2017 images (~778 MB, ~5K images) ────────────────────────────────
    val_zip = zips_dir / "val2017.zip"
    download_file(COCO_URLS["val2017_images"], val_zip)
    extract_zip(val_zip, RAW_DIR / "images")

    # ── Train2017 images (~18 GB, ~118K images) ─────────────────────────────
    if not skip_train:
        train_zip = zips_dir / "train2017.zip"
        download_file(COCO_URLS["train2017_images"], train_zip)
        extract_zip(train_zip, RAW_DIR / "images")
    else:
        print("\n  [INFO] skip_train=True: skipping train2017 download.")
        print("         Only val2017 will be used (smaller experiment).")


# ─────────────────────────────────────────────────────────────────────────────
# COCO JSON → YOLO conversion (filter to person + car only)
# ─────────────────────────────────────────────────────────────────────────────

def parse_coco_json(ann_json_path: Path,
                    img_dir:       Path) -> tuple:
    """
    Parse a COCO instances JSON file.
    Keeps only images that contain at least one person or car annotation.
    Converts bbox format: COCO [x, y, w, h] → YOLO [cx, cy, w, h] normalized.

    Returns:
        entries : list of (img_stem, img_path, [yolo_lines])
        stats   : dict
    """
    print(f"\n  [→] Parsing: {ann_json_path.name}")
    with open(ann_json_path, encoding="utf-8") as f:
        coco = json.load(f)

    # ── Build COCO category_id → your class ID mapping ─────────────────────
    cat_id_to_yolo = {}
    for cat in coco["categories"]:
        if cat["name"] in KEEP_CLASSES:
            cat_id_to_yolo[cat["id"]] = KEEP_CLASSES[cat["name"]]

    # ── Build image_id → image info mapping ────────────────────────────────
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # ── Group annotations by image_id ───────────────────────────────────────
    img_annotations = {}
    skipped = 0
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in cat_id_to_yolo:
            continue   # not person or car

        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []

        # COCO bbox: [x_topleft, y_topleft, width, height] absolute pixels
        x, y, bw, bh = ann["bbox"]
        img_info = img_id_to_info[img_id]
        iw = float(img_info["width"])
        ih = float(img_info["height"])

        # Skip degenerate boxes
        if bw < 2 or bh < 2:
            skipped += 1
            continue

        # Convert to YOLO normalized cx, cy, w, h
        cx = (x + bw / 2.0) / iw
        cy = (y + bh / 2.0) / ih
        nw = bw / iw
        nh = bh / ih

        # Clamp to valid range
        cx = max(0.001, min(0.999, cx))
        cy = max(0.001, min(0.999, cy))
        nw = max(0.001, min(0.999, nw))
        nh = max(0.001, min(0.999, nh))

        yolo_cls = cat_id_to_yolo[cat_id]
        img_annotations[img_id].append(
            f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )

    # ── Build entries list for images that have person or car ──────────────
    entries = []
    missing_imgs  = 0
    class_counts  = {c: 0 for c in KEEP_CLASSES}

    for img_id, yolo_lines in tqdm(img_annotations.items(),
                                    desc="  Building entries"):
        img_info = img_id_to_info[img_id]
        img_path = img_dir / img_info["file_name"]

        if not img_path.exists():
            missing_imgs += 1
            continue

        entries.append((img_path.stem, img_path, yolo_lines))

        # Count classes
        for line in yolo_lines:
            cls_id = int(line.split()[0])
            cls_name = NAMES[cls_id]
            class_counts[cls_name] += 1

    stats = {
        "total_images":    len(entries),
        "total_boxes":     sum(len(e[2]) for e in entries),
        "skipped_degen":   skipped,
        "missing_imgs":    missing_imgs,
        "class_counts":    class_counts,
    }
    return entries, stats


# ─────────────────────────────────────────────────────────────────────────────
# Write split
# ─────────────────────────────────────────────────────────────────────────────

def write_split(entries:      list,
                split_name:   str,
                use_symlinks: bool = True) -> None:
    img_dir = OUT_DIR / "images" / split_name
    lbl_dir = OUT_DIR / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for stem, src_img, yolo_lines in tqdm(entries,
                                           desc=f"  Writing {split_name:<6}"):
        dst_img = img_dir / src_img.name
        if not dst_img.exists():
            if use_symlinks:
                try:
                    os.symlink(src_img.resolve(), dst_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        dst_lbl = lbl_dir / (stem + ".txt")
        if not dst_lbl.exists():
            with open(dst_lbl, "w") as f:
                f.write("\n".join(yolo_lines))


def write_dataset_yaml(n_train: int, n_val: int, n_test: int) -> None:
    content = {
        "path":  str(OUT_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(KEEP_CLASSES),
        "names": NAMES,
    }
    out = OUT_DIR / "dataset.yaml"
    with open(out, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    print(f"\n[✓] dataset.yaml written")
    print(f"    train : {n_train:>6,} images  (for YOLO training)")
    print(f"    val   : {n_val:>6,} images  (monitored during training)")
    print(f"    test  : {n_test:>6,} images  (held-out for paper results)")


def print_stats(stats: dict, split: str) -> None:
    b = max(stats["total_boxes"], 1)
    print(f"\n  [{split.upper()}]")
    print(f"    Images with person/car : {stats['total_images']:>7,}")
    print(f"    Total boxes kept       : {stats['total_boxes']:>7,}")
    print(f"    Skipped (degenerate)   : {stats['skipped_degen']:>7,}")
    print(f"    Missing image files    : {stats['missing_imgs']:>7,}")
    print(f"    Class breakdown:")
    for cls_name, count in stats["class_counts"].items():
        pct = 100.0 * count / b
        bar = "█" * int(pct / 2)
        print(f"      {cls_name:<10} {count:>7,}  ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(skip_train: bool = False,
         use_symlinks: bool = True) -> None:

    print("=" * 65)
    print("  COCO 2017 → YOLO  (Person + Car only)")
    print("=" * 65)

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

    # ── Step 1: Download ───────────────────────────────────────────────────────
    print("\n[1/5] Downloading COCO 2017...")
    print("      val2017   ~  778 MB  (~5K images)")
    if not skip_train:
        print("      train2017 ~ 18.0 GB  (~118K images)")
        print("      annotations ~ 241 MB")
    download_coco(skip_train=skip_train)

    # ── Step 2: Parse train JSON → filter person+car ───────────────────────────
    ann_dir = RAW_DIR / "annotations_raw" / "annotations"
    img_dir = RAW_DIR / "images"

    train_entries, train_stats = [], {"total_images": 0, "total_boxes": 0,
                                       "skipped_degen": 0, "missing_imgs": 0,
                                       "class_counts": {"person": 0, "car": 0}}
    if not skip_train:
        print("\n[2/5] Parsing train2017 annotations...")
        train_json = ann_dir / "instances_train2017.json"
        train_entries, train_stats = parse_coco_json(train_json,
                                                      img_dir / "train2017")
        print_stats(train_stats, "train2017 (filtered)")
    else:
        print("\n[2/5] Skipping train2017 (skip_train=True)")

    # ── Step 3: Parse val JSON → filter person+car ─────────────────────────────
    print("\n[3/5] Parsing val2017 annotations...")
    val_json     = ann_dir / "instances_val2017.json"
    val_entries, val_stats = parse_coco_json(val_json,
                                              img_dir / "val2017")
    print_stats(val_stats, "val2017 (filtered)")

    # ── Step 4: Build final splits ─────────────────────────────────────────────
    print("\n[4/5] Building train / val / test splits...")

    rng = random.Random(RANDOM_SEED)

    if train_entries:
        # Standard split: train2017 → 80% train + 20% val
        #                 val2017   → 100% test (held-out benchmark)
        shuffled = train_entries.copy()
        rng.shuffle(shuffled)
        n_train_split = int(len(shuffled) * TRAIN_VAL_SPLIT)
        final_train = shuffled[:n_train_split]
        final_val   = shuffled[n_train_split:]
        final_test  = val_entries

        print(f"    train : {len(final_train):,}  (80% of filtered train2017)")
        print(f"    val   : {len(final_val):,}  (20% of filtered train2017)")
        print(f"    test  : {len(final_test):,}  (all filtered val2017 — standard COCO benchmark)")
    else:
        # val-only mode: split val2017 into train/val/test (60/20/20)
        print("  [INFO] val-only mode: splitting val2017 60/20/20")
        shuffled = val_entries.copy()
        rng.shuffle(shuffled)
        n  = len(shuffled)
        n1 = int(n * 0.60)
        n2 = int(n * 0.80)
        final_train = shuffled[:n1]
        final_val   = shuffled[n1:n2]
        final_test  = shuffled[n2:]
        print(f"    train : {len(final_train):,}")
        print(f"    val   : {len(final_val):,}")
        print(f"    test  : {len(final_test):,}  ← held-out")

    # ── Step 5: Write splits ───────────────────────────────────────────────────
    print(f"\n[5/5] Writing YOLO files  (symlinks={use_symlinks})...")
    write_split(final_train, "train", use_symlinks)
    write_split(final_val,   "val",   use_symlinks)
    write_split(final_test,  "test",  use_symlinks)

    write_dataset_yaml(len(final_train), len(final_val), len(final_test))

    # ── Summary ────────────────────────────────────────────────────────────────
    total = len(final_train) + len(final_val) + len(final_test)
    print(f"\n{'='*65}")
    print(f"  Done!  {total:,} total images  →  {OUT_DIR}")
    print(f"\n  Next steps:")
    print(f"    1. Update config.py:")
    print(f"         DATASET = 'coco_person_car'")
    print(f"    2. python step1_generate_lr_datasets.py")
    print(f"    3. python step1_train_all.py")
    print(f"    4. python step2_evaluate.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-only", action="store_true",
        help=(
            "Download only val2017 (~800MB, ~2,700 person+car images). "
            "Faster but smaller training set. "
            "Good for quick experiments before full run."
        )
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy images instead of symlinking (use on Windows)"
    )
    args = parser.parse_args()
    main(skip_train=args.val_only, use_symlinks=not args.copy)