"""
Step 1B: Train YOLO11n on every LR variant.

Runs sequentially (one GPU). For multi-GPU or parallel training,
see the comment at the bottom about --device.

Outputs:
    runs/step1/{variant_name}/   ← weights, metrics, plots per variant
"""

import subprocess
import sys
import time
from pathlib import Path
from config import (VARIANTS, OUTPUT_DIR, RUNS_DIR, MODEL,
                    EPOCHS, PATIENCE, BATCH, DEVICE, WORKERS)


def train_variant(variant_name: str, cfg: dict) -> bool:
    """
    Train YOLO on a single variant. Returns True on success.
    Uses Ultralytics Python API directly for cleaner error handling.
    """
    from ultralytics import YOLO

    dataset_yaml = OUTPUT_DIR / variant_name / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"  [ERROR] dataset.yaml not found for {variant_name}. "
              f"Run step1_generate_lr_datasets.py first.")
        return False

    save_dir = RUNS_DIR / variant_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already trained (has best.pt)
    if (save_dir / "weights" / "best.pt").exists():
        print(f"  [SKIP] {variant_name} already trained. "
              f"Delete runs/step1/{variant_name}/ to retrain.")
        return True

    print(f"\n{'='*60}")
    print(f"  Training: {variant_name}  |  imgsz={cfg['imgsz']}")
    print(f"{'='*60}")

    model = YOLO(MODEL)

    model.train(
        data      = str(dataset_yaml),
        epochs    = EPOCHS,
        imgsz     = cfg["imgsz"],
        batch     = BATCH,
        patience  = PATIENCE,
        device    = DEVICE,
        workers   = WORKERS,
        project   = str(RUNS_DIR),
        name      = variant_name,
        exist_ok  = True,           # overwrite if exists

        # Augmentations — keep light since LR images are already degraded
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        flipud    = 0.0,
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.0,

        # Logging
        plots     = True,
        save      = True,
        verbose   = False,          # set True for detailed epoch logs
    )

    print(f"  [✓] Finished: {variant_name}")
    return True


def run_all_training() -> None:
    print("=" * 60)
    print("  Step 1B: Training All LR Variants")
    print(f"  Model:  {MODEL}")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH}  |  Device: {DEVICE}")
    print("=" * 60)

    results = {}
    start_total = time.time()

    for vname, vcfg in VARIANTS.items():
        t0 = time.time()
        success = train_variant(vname, vcfg)
        elapsed = (time.time() - t0) / 60.0
        results[vname] = {"success": success, "minutes": round(elapsed, 1)}
        print(f"  [{vname}] → {'OK' if success else 'FAILED'}  ({elapsed:.1f} min)")

    total_min = (time.time() - start_total) / 60.0
    print(f"\n[✓] All training complete in {total_min:.1f} minutes.")
    print("\nTraining Summary:")
    for vname, r in results.items():
        status = "✓" if r["success"] else "✗"
        print(f"  [{status}] {vname:<28}  {r['minutes']} min")

    print("\nNext: run step1_analyze_results.py")


if __name__ == "__main__":
    run_all_training()
