"""
Step 1A: Generate all LR dataset variants from the HR Penn-Fudan YOLO dataset.

KEY INSIGHT (important for paper):
    YOLO labels use normalized coordinates [0,1] relative to image dimensions.
    When we line-skip uniformly (e.g., every k+1 th row), ALL pixel positions
    shift proportionally, so normalized coords remain EXACTLY the same.
    → Labels are COPIED unchanged. No relabeling needed.

    Proof: point at row r in H×W image → row r/(k+1) in H'×W' LR image.
    Normalized coord: r/H = (r/(k+1)) / (H/(k+1))  ✓ IDENTICAL.
"""

import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm
from config import SOURCE_YOLO, OUTPUT_DIR, VARIANTS, SPLITS, NC, NAMES


def apply_line_skip(img: np.ndarray,
                    row_skip: int,
                    col_skip: int,
                    upsample: bool) -> np.ndarray:
    """
    Simulate sensor line-skipping on a single image.

    Args:
        img:       HxWx3 BGR image (original HR)
        row_skip:  k → sample every (k+1)th row; 0 = no row skip
        col_skip:  k → sample every (k+1)th col; 0 = no col skip
        upsample:  if True, blocky NN upsample back to original (H, W)
                   if False, return the smaller native LR image

    Returns:
        LR image (smaller if upsample=False, same size if upsample=True)
    """
    H, W = img.shape[:2]
    step_r = row_skip + 1
    step_c = col_skip + 1

    # Sub-sample: select every step_r-th row, every step_c-th col
    lr_img = img[::step_r, ::step_c]   # shape: (H//step_r, W//step_c, 3)

    if upsample:
        # Blocky nearest-neighbor upsample back to original size
        # This is what a post-processing pipeline would do if LR model needs fixed input
        lr_img = cv2.resize(lr_img, (W, H), interpolation=cv2.INTER_NEAREST)

    return lr_img


def write_dataset_yaml(dataset_dir: Path, variant_name: str) -> None:
    """Write YOLO dataset.yaml for a variant."""
    yaml_content = {
        "path":  str(dataset_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NC,
        "names": NAMES,
    }
    with open(dataset_dir / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)


def generate_variant(variant_name: str, cfg: dict) -> dict:
    """
    Generate one LR variant dataset.

    Returns a dict with metadata about what was created (for logging).
    """
    row_skip = cfg["row_skip"]
    col_skip = cfg["col_skip"]
    upsample = cfg["upsample"]

    dst_dir = OUTPUT_DIR / variant_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    sample_shapes = []

    for split in SPLITS:
        src_img_dir = SOURCE_YOLO / "images" / split
        src_lbl_dir = SOURCE_YOLO / "labels" / split

        dst_img_dir = dst_dir / "images" / split
        dst_lbl_dir = dst_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(
            list(src_img_dir.glob("*.jpg")) +
            list(src_img_dir.glob("*.jpeg")) +
            list(src_img_dir.glob("*.png"))
        )

        for img_path in tqdm(img_paths, desc=f"  {variant_name}/{split}", leave=False):
            # ── Process image ──────────────────────────────────────────────────
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] Cannot read {img_path}, skipping.")
                continue

            lr_img = apply_line_skip(img, row_skip, col_skip, upsample)

            # Save as PNG to avoid JPEG compression artefacts corrupting LR pattern
            out_img_path = dst_img_dir / (img_path.stem + ".png")
            cv2.imwrite(str(out_img_path), lr_img)

            if len(sample_shapes) < 3:
                sample_shapes.append(lr_img.shape[:2])

            # ── Copy label UNCHANGED (normalized coords are invariant) ─────────
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy(str(lbl_path), str(dst_lbl_dir / lbl_path.name))
            else:
                # Create empty label file for background images
                (dst_lbl_dir / (img_path.stem + ".txt")).touch()

            total_images += 1

    write_dataset_yaml(dst_dir, variant_name)

    return {
        "variant":       variant_name,
        "total_images":  total_images,
        "sample_shapes": sample_shapes,
        "upsample":      upsample,
        "row_skip":      row_skip,
        "col_skip":      col_skip,
    }


def create_visual_comparison(n_images: int = 3) -> None:
    """
    Create a side-by-side visual comparison figure of all variants.
    Saves to results/step1/lr_visual_comparison.png
    Useful directly in the paper.
    """
    from config import RESULTS_DIR
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Pick a few sample images from the val set
    hr_val_dir = OUTPUT_DIR / "hr_640" / "images" / "val"
    sample_imgs = sorted(hr_val_dir.glob("*.png"))[:n_images]
    if not sample_imgs:
        print("[WARN] No HR images found for visual comparison. Run generation first.")
        return

    variant_keys = list(VARIANTS.keys())
    n_variants   = len(variant_keys)
    n_samples    = len(sample_imgs)

    fig, axes = plt.subplots(
        n_samples, n_variants,
        figsize=(3 * n_variants, 3 * n_samples),
        squeeze=False
    )

    for col_idx, vname in enumerate(variant_keys):
        for row_idx, hr_path in enumerate(sample_imgs):
            lr_path = OUTPUT_DIR / vname / "images" / "val" / hr_path.name
            if not lr_path.exists():
                axes[row_idx][col_idx].axis("off")
                continue

            img_bgr = cv2.imread(str(lr_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            axes[row_idx][col_idx].imshow(img_rgb)
            axes[row_idx][col_idx].axis("off")

            if row_idx == 0:
                cfg = VARIANTS[vname]
                from config import pixel_ratio
                pr  = pixel_ratio(cfg["row_skip"], cfg["col_skip"])
                axes[row_idx][col_idx].set_title(
                    f"{VARIANTS[vname]['label']}\n"
                    f"Pixels read: {pr*100:.1f}%",
                    fontsize=8, pad=4
                )

    plt.suptitle("LR Variants — Visual Comparison\n(All from same original image)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = RESULTS_DIR / "lr_visual_comparison.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved visual comparison: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import pixel_ratio, energy_savings_pct

    print("=" * 60)
    print("  Step 1A: Generating LR Dataset Variants")
    print("=" * 60)
    print(f"  Source : {SOURCE_YOLO}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Variants to generate: {list(VARIANTS.keys())}")
    print()

    assert SOURCE_YOLO.exists(), (
        f"Source dataset not found: {SOURCE_YOLO}\n"
        "Please update SOURCE_YOLO in config.py"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for vname, vcfg in VARIANTS.items():
        print(f"[→] Generating: {vname}  |  "
              f"row_skip={vcfg['row_skip']}  col_skip={vcfg['col_skip']}  "
              f"upsample={vcfg['upsample']}  "
              f"pixels_read={pixel_ratio(vcfg['row_skip'], vcfg['col_skip'])*100:.1f}%")
        result = generate_variant(vname, vcfg)
        summary.append(result)
        print(f"  [✓] Done. {result['total_images']} images. "
              f"Sample shapes: {result['sample_shapes']}")
        print()

    print("=" * 60)
    print("  Generation Summary")
    print("=" * 60)
    print(f"{'Variant':<25} {'Pixels Read':>12} {'Energy Saved':>14} {'Upsample':>10}")
    print("-" * 65)
    for vname, vcfg in VARIANTS.items():
        pr  = pixel_ratio(vcfg["row_skip"], vcfg["col_skip"])
        es  = energy_savings_pct(vcfg["row_skip"], vcfg["col_skip"])
        ups = "Yes" if vcfg["upsample"] else "No (native)"
        print(f"  {vname:<23} {pr*100:>10.1f}%  {es:>12.1f}%  {ups:>10}")
    print()

    # Create visual comparison figure
    print("[→] Creating visual comparison figure...")
    create_visual_comparison(n_images=3)

    print("\n[✓] All variants generated. Run step1_train_all.py next.")
