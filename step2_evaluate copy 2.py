"""
step2_evaluate.py
=================
Adaptive CIS Evaluation — Step 2
Runs AdaptiveCISPipeline over the test split at multiple LR confidence
thresholds to generate the energy-accuracy trade-off curve.

Compares three systems:
  [A] Always-LR  : mAP from Step 1 both_skip1_640 (25% pixels)
  [B] Always-HR  : mAP from Step 1 hr_640          (100% pixels)
  [C] Adaptive   : Step 2 pipeline at various conf thresholds

Dataset is read from config.py — switch DATASET there, not here.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import config as C
from step2_pipeline import AdaptiveCISPipeline

# ── Paths from config (dataset-agnostic) ──────────────────────────────────────
ROOT         = Path(__file__).parent
RUNS_DIR     = C.RUNS_DIR                          # runs/step1/<dataset>/
RESULTS_DIR  = C.RESULTS_DIR_STEP2                 # results/step2/<dataset>/
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LR_MODEL = RUNS_DIR / "both_skip1_640" / "weights" / "best.pt"
HR_MODEL = RUNS_DIR / "hr_640"         / "weights" / "best.pt"

TEST_IMG_DIR = C.SOURCE_YOLO / "images" / "test"
TEST_LBL_DIR = C.SOURCE_YOLO / "labels" / "test"

# ── Step 1 baselines — READ FROM YOUR metrics_table.csv ───────────────────────
# Values from file:211 (COCO person+car, YOLO11n, 80 epochs)
# UPDATE THESE if you retrain or switch dataset
STEP1_HR_MAP50    = 0.5555   # hr_640          mAP@0.5
STEP1_HR_MAP5095  = 0.3254   # hr_640          mAP@0.5:0.95
STEP1_HR_ENERGY   = 100.0    # 100% pixels

STEP1_LR_MAP50    = 0.5159   # both_skip1_640  mAP@0.5
STEP1_LR_MAP5095  = 0.3010   # both_skip1_640  mAP@0.5:0.95
STEP1_LR_ENERGY   = 25.0     # 25% pixels (row+col skip×2)

# ── Confidence threshold sweep ─────────────────────────────────────────────────
# Extended at the low end (0.10, 0.15) because COCO has many small objects
# that appear at low confidence in the LR detector
CONF_THRESHOLDS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="colorblind")
DATASET_LABEL = {
    "pennfudan":       "Penn-Fudan",
    "bdd100k":         "BDD100K",
    "coco_person_car": "COCO (Person + Car)",
}.get(C.DATASET, C.DATASET)


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth loader
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    """YOLO normalized xywh → pixel xyxy. Returns list of {cls, box}."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append({'cls': int(cls), 'box': [x1, y1, x2, y2]})
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# mAP computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 1e-6 else 0.0


def compute_ap(pred_boxes: list, gt_boxes: list,
               iou_thresh: float = 0.5) -> float:
    """
    Per-image Average Precision at a single IoU threshold.
    pred_boxes: [[x1,y1,x2,y2,conf,cls], ...]  sorted by conf desc
    gt_boxes:   [{'cls': int, 'box': [x1,y1,x2,y2]}, ...]
    """
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0

    pred_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    matched     = [False] * len(gt_boxes)
    tp, fp      = [], []

    for pred in pred_sorted:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if gt['cls'] != int(pred[5]):
                continue
            iou = compute_iou(pred[:4], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_thresh and best_idx >= 0 and not matched[best_idx]:
            tp.append(1); fp.append(0)
            matched[best_idx] = True
        else:
            tp.append(0); fp.append(1)

    tp_cum    = np.cumsum(tp).astype(float)
    fp_cum    = np.cumsum(fp).astype(float)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall    = tp_cum / (len(gt_boxes) + 1e-9)

    # 101-point interpolated AP (COCO-style)
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p   = precision[recall >= thr].max() if (recall >= thr).any() else 0.0
        ap += p / 101.0
    return ap


def evaluate_dataset(all_preds: dict, all_gts: dict) -> dict:
    """
    Compute mAP@0.5 and mAP@0.5:0.95 over the full dataset.
    all_preds: {img_stem: [[x1,y1,x2,y2,conf,cls], ...]}
    all_gts:   {img_stem: [{'cls': int, 'box': [...]}]}
    """
    iou_thresholds = np.arange(0.50, 1.00, 0.05)   # 0.50, 0.55, ..., 0.95
    aps_50, aps_5095 = [], []

    for stem in all_gts:
        preds = all_preds.get(stem, [])
        gts   = all_gts[stem]

        aps_50.append(compute_ap(preds, gts, iou_thresh=0.50))
        aps_5095.append(np.mean([
            compute_ap(preds, gts, iou_thresh=t) for t in iou_thresholds
        ]))

    return {
        'mAP50':   round(float(np.mean(aps_50)),   4),
        'mAP5095': round(float(np.mean(aps_5095)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation() -> pd.DataFrame:
    print("=" * 65)
    print(f"  Step 2: Adaptive CIS Evaluation — {DATASET_LABEL}")
    print("=" * 65)

    # ── Check model files exist ────────────────────────────────────────────────
    for p, name in [(LR_MODEL, "LR model"), (HR_MODEL, "HR model")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{name} not found: {p}\n"
                f"Run step1_train_all.py first."
            )
    print(f"  LR model : {LR_MODEL}")
    print(f"  HR model : {HR_MODEL}")

    # ── Build pipeline (models loaded once, reused across all thresholds) ──────
    pipeline = AdaptiveCISPipeline(
        lr_model_path  = str(LR_MODEL),
        hr_model_path  = str(HR_MODEL),
        row_skip       = C.ADAPTIVE["row_skip"],
        col_skip       = C.ADAPTIVE["col_skip"],
        padding_ratio  = C.ADAPTIVE["padding_ratio"],
        min_crop_px    = C.ADAPTIVE["min_crop_px"],
        hr_conf        = C.ADAPTIVE["hr_trigger_conf"],  # overridden per loop
        nms_iou        = C.ADAPTIVE["nms_iou"],
        lr_detect_conf = C.ADAPTIVE["lr_detect_conf"],
    )

    # ── Load test images ───────────────────────────────────────────────────────
    img_paths = sorted(
        list(TEST_IMG_DIR.glob("*.jpg")) +
        list(TEST_IMG_DIR.glob("*.png"))
    )
    if not img_paths:
        raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")
    print(f"  Test images : {len(img_paths):,}")
    print(f"  Classes     : {C.NC} → {C.NAMES}")

    # ── Pre-load all ground truths (avoid re-reading per threshold) ────────────
    all_gts = {}
    print("\n  Loading ground truths...")
    for p in tqdm(img_paths, desc="  GT loading"):
        img      = cv2.imread(str(p))
        H, W     = img.shape[:2]
        lbl_path = TEST_LBL_DIR / (p.stem + ".txt")
        all_gts[p.stem] = load_gt_boxes(lbl_path, W, H)

    n_with_obj = sum(1 for v in all_gts.values() if len(v) > 0)
    n_bg       = len(all_gts) - n_with_obj
    print(f"  Images with objects : {n_with_obj:,}  "
          f"({100*n_with_obj/len(img_paths):.1f}%)")
    print(f"  Background frames   : {n_bg:,}  "
          f"({100*n_bg/len(img_paths):.1f}%)  "
          f"← 100% energy saved on these frames")

    # ── Threshold sweep ────────────────────────────────────────────────────────
    results = []

    for conf in CONF_THRESHOLDS:
        print(f"\n[→] HR trigger conf = {conf:.2f}")
        all_preds        = {}
        energy_ratios    = []
        hr_area_pcts     = []
        n_lr_dets_total  = 0
        n_hr_crops_total = 0

        for img_path in tqdm(img_paths, desc=f"  τ={conf:.2f}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            dets, energy_info, debug = pipeline.process_frame(
                img, hr_trigger_conf=conf
            )

            all_preds[img_path.stem] = dets
            energy_ratios.append(energy_info['energy_ratio'])
            hr_area_pcts.append(energy_info['hr_pct'])
            n_lr_dets_total  += len(debug['lr_dets'])
            n_hr_crops_total += len(debug['hr_crops'])

        metrics          = evaluate_dataset(all_preds, all_gts)
        avg_energy_ratio = float(np.mean(energy_ratios))
        avg_hr_pct       = float(np.mean(hr_area_pcts))

        row = {
            'hr_trigger_conf':  conf,
            'mAP50':            metrics['mAP50'],
            'mAP5095':          metrics['mAP5095'],
            'avg_energy_ratio': round(avg_energy_ratio, 4),
            'avg_energy_pct':   round(avg_energy_ratio * 100, 2),
            'avg_energy_saved': round((1 - avg_energy_ratio) * 100, 2),
            'avg_hr_area_pct':  round(avg_hr_pct, 2),
            'total_lr_dets':    n_lr_dets_total,
            'total_hr_crops':   n_hr_crops_total,
            'hr_activation_rate': round(
                100.0 * n_hr_crops_total / max(len(img_paths), 1), 1
            ),
        }
        results.append(row)
        print(f"  mAP@0.5={metrics['mAP50']:.4f}  "
              f"mAP@0.5:0.95={metrics['mAP5095']:.4f}  "
              f"energy_used={avg_energy_ratio*100:.1f}%  "
              f"saved={row['avg_energy_saved']:.1f}%  "
              f"HR_crops={n_hr_crops_total:,}")

    df       = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "adaptive_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[✓] Saved: {csv_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Paper Figure 1: Energy vs Accuracy trade-off (THE main figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy_accuracy(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric, ylabel in zip(
        axes,
        ['mAP50', 'mAP5095'],
        ['mAP @ IoU=0.5', 'mAP @ IoU=0.5:0.95']
    ):
        # ── Adaptive curve ─────────────────────────────────────────────────
        ax.plot(df['avg_energy_saved'], df[metric],
                marker='o', linewidth=2.5, markersize=9,
                color='steelblue', label='Adaptive LR+HR (Ours)', zorder=5)

        for _, r in df.iterrows():
            ax.annotate(f"τ={r['hr_trigger_conf']:.2f}",
                        (r['avg_energy_saved'], r[metric]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7.5, color='steelblue')

        # ── Always-LR point ────────────────────────────────────────────────
        lr_map = STEP1_LR_MAP50 if metric == 'mAP50' else STEP1_LR_MAP5095
        ax.scatter([100 - STEP1_LR_ENERGY], [lr_map],
                   marker='s', s=130, color='darkorange', zorder=6,
                   label=f'Always-LR (×2 skip, 25% pixels)\nmAP={lr_map:.4f}')
        ax.axhline(lr_map, color='darkorange',
                   linestyle=':', linewidth=1.2, alpha=0.6)

        # ── Always-HR point ────────────────────────────────────────────────
        hr_map = STEP1_HR_MAP50 if metric == 'mAP50' else STEP1_HR_MAP5095
        ax.scatter([0.0], [hr_map],
                   marker='^', s=130, color='green', zorder=6,
                   label=f'Always-HR (100% pixels)\nmAP={hr_map:.4f}')
        ax.axhline(hr_map, color='green',
                   linestyle='--', linewidth=1.2, alpha=0.6)

        # ── ±2.5% acceptable accuracy band ────────────────────────────────
        ax.axhspan(hr_map - 0.025, hr_map + 0.025,
                   alpha=0.08, color='green', label='±2.5% of HR accuracy')

        ax.set_xlabel("Sensor Front-End Energy Saved (%)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Adaptive CIS: Energy vs {ylabel}\n"
                     f"({DATASET_LABEL}, YOLO11n)", fontsize=11)
        ax.legend(fontsize=8.5, loc='lower left')
        ax.set_xlim(-3, 103)
        y_min = max(0.0, min(df[metric].min(), lr_map, hr_map) - 0.05)
        y_max = min(1.0, max(df[metric].max(), lr_map, hr_map) + 0.05)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.grid(True, alpha=0.35)

    plt.suptitle(
        f"Step 2: Adaptive LR+HR Pipeline Results\n"
        f"Dataset: {DATASET_LABEL}  |  Model: YOLO11n  |  "
        f"LR mode: row+col skip×2 (25% pixels)",
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    out = RESULTS_DIR / "energy_accuracy_step2.png"
    plt.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Paper Figure 2: Qualitative samples
# ─────────────────────────────────────────────────────────────────────────────

def plot_qualitative(n_samples: int = 4) -> None:
    """
    n_samples rows × 3 columns:
      (a) HR + Ground Truth boxes (green)
      (b) LR image + LR detections (orange)
      (c) Adaptive output: HR crop regions (blue) + final detections (red)
    """
    pipeline = AdaptiveCISPipeline(
        lr_model_path  = str(LR_MODEL),
        hr_model_path  = str(HR_MODEL),
        row_skip       = C.ADAPTIVE["row_skip"],
        col_skip       = C.ADAPTIVE["col_skip"],
        padding_ratio  = C.ADAPTIVE["padding_ratio"],
        min_crop_px    = C.ADAPTIVE["min_crop_px"],
        hr_conf        = C.ADAPTIVE["hr_trigger_conf"],
        nms_iou        = C.ADAPTIVE["nms_iou"],
        lr_detect_conf = C.ADAPTIVE["lr_detect_conf"],
    )

    # Pick images that have at least one object (better visualisation)
    all_paths = sorted(
        list(TEST_IMG_DIR.glob("*.jpg")) +
        list(TEST_IMG_DIR.glob("*.png"))
    )
    img_paths = []
    for p in all_paths:
        lbl = TEST_LBL_DIR / (p.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            img_paths.append(p)
        if len(img_paths) >= n_samples:
            break

    if not img_paths:
        print("[WARN] No labeled test images found for qualitative plot.")
        return

    fig, axes = plt.subplots(
        len(img_paths), 3,
        figsize=(16, 5 * len(img_paths)),
        squeeze=False
    )
    col_titles = [
        '(a) Original HR + Ground Truth',
        '(b) LR Readout + LR Detections',
        '(c) Adaptive: HR Crops Activated + Final Detections',
    ]

    # Class name lookup
    cls_names = C.NAMES  # e.g. ["person", "car"]

    for row_idx, img_path in enumerate(img_paths):
        hr_img = cv2.imread(str(img_path))
        H, W   = hr_img.shape[:2]
        lbl    = TEST_LBL_DIR / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(lbl, W, H)

        # Run pipeline at default HR trigger threshold
        dets, energy_info, debug = pipeline.process_frame(
            hr_img, hr_trigger_conf=C.ADAPTIVE["hr_trigger_conf"]
        )

        # ── Panel (a): HR + GT ─────────────────────────────────────────────
        ax = axes[row_idx][0]
        disp = cv2.cvtColor(hr_img.copy(), cv2.COLOR_BGR2RGB)
        for gt in gt_boxes:
            x1, y1, x2, y2 = [int(v) for v in gt['box']]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label_text = cls_names[gt['cls']] if gt['cls'] < len(cls_names) else str(gt['cls'])
            cv2.putText(disp, label_text,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        ax.imshow(disp)
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=10, pad=6)
        ax.set_ylabel(f"Sample {row_idx+1}", fontsize=9)

        # ── Panel (b): LR + LR detections ─────────────────────────────────
        ax = axes[row_idx][1]
        lr_disp = cv2.cvtColor(debug['lr_img'].copy(), cv2.COLOR_BGR2RGB)
        for d in debug['lr_dets']:
            x1, y1, x2, y2, conf, cls_id = d
            cls_id = int(cls_id)
            lname  = cls_names[cls_id] if cls_id < len(cls_names) else str(cls_id)
            cv2.rectangle(lr_disp,
                          (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 165, 0), 2)
            cv2.putText(lr_disp,
                        f"{lname} {conf:.2f}",
                        (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 165, 0), 1)
        ax.imshow(lr_disp)
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[1], fontsize=10, pad=6)
        ax.text(4, H - 10,
                f"LR pixels: {energy_info['lr_pct']}%  "
                f"({C.ADAPTIVE['row_skip']+1}× skip)",
                color='white', fontsize=8,
                bbox=dict(facecolor='darkorange', alpha=0.7, pad=2))

        # ── Panel (c): Adaptive — HR crops + final detections ─────────────
        ax = axes[row_idx][2]
        adapt_disp = cv2.cvtColor(hr_img.copy(), cv2.COLOR_BGR2RGB)
        # Blue shading for HR-activated regions
        overlay = adapt_disp.copy()
        for (cx1, cy1, cx2, cy2) in debug['hr_crops']:
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (100, 149, 237), -1)
        adapt_disp = cv2.addWeighted(overlay, 0.25, adapt_disp, 0.75, 0)
        for (cx1, cy1, cx2, cy2) in debug['hr_crops']:
            cv2.rectangle(adapt_disp, (cx1, cy1), (cx2, cy2), (30, 100, 255), 2)
        # Final detections in red
        for d in dets:
            x1, y1, x2, y2, conf, cls_id = d
            cls_id = int(cls_id)
            lname  = cls_names[cls_id] if cls_id < len(cls_names) else str(cls_id)
            cv2.rectangle(adapt_disp,
                          (int(x1), int(y1)), (int(x2), int(y2)),
                          (220, 30, 30), 2)
            cv2.putText(adapt_disp,
                        f"{lname} {conf:.2f}",
                        (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 30, 30), 1)
        ax.imshow(adapt_disp)
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[2], fontsize=10, pad=6)
        ax.text(4, H - 10,
                f"Total: {energy_info['total_pct']}%  "
                f"(LR:{energy_info['lr_pct']}% + HR:{energy_info['hr_pct']}%)",
                color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.7, pad=2))

    plt.suptitle(
        f"Qualitative Results — Adaptive CIS Pipeline  ({DATASET_LABEL})\n"
        "Green=GT  |  Orange=LR detections  |  "
        "Blue=HR-activated regions  |  Red=Final detections",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out = RESULTS_DIR / "qualitative_samples.png"
    plt.savefig(str(out), dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def print_latex_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  LaTeX Table — Copy into paper")
    print("=" * 70)

    rows = [
        ("Always-HR (full sensor)",
         STEP1_HR_ENERGY, 100 - STEP1_HR_ENERGY,
         STEP1_HR_MAP50, STEP1_HR_MAP5095),
        ("Always-LR (×2 skip, ours)",
         STEP1_LR_ENERGY, 100 - STEP1_LR_ENERGY,
         STEP1_LR_MAP50, STEP1_LR_MAP5095),
    ]
    for _, r in df.iterrows():
        rows.append((
            f"Adaptive $\\tau$={r['hr_trigger_conf']:.2f} (ours)",
            round(r['avg_energy_pct'], 1),
            round(r['avg_energy_saved'], 1),
            r['mAP50'],
            r['mAP5095'],
        ))

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(f"\\caption{{Adaptive CIS pipeline results on {DATASET_LABEL} "
          r"(YOLO11n). $\tau$ = HR trigger confidence threshold.}}")
    print(r"\label{tab:adaptive_results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Pixels (\%)} & "
          r"\textbf{Energy Saved (\%)} & "
          r"\textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} \\")
    print(r"\midrule")
    for name, px, saved, m50, m5095 in rows:
        print(f"  {name} & {px:.1f} & {saved:.1f} & "
              f"{m50:.4f} & {m5095:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Run evaluation
    df = run_evaluation()

    # 2. Print results table
    print("\nFull Results:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(df.to_string(index=False))

    # 3. Generate figures
    print("\n[→] Generating paper figures...")
    plot_energy_accuracy(df)
    plot_qualitative(n_samples=4)
    print_latex_table(df)

    # 4. Print interpretation guide
    print(f"\n{'='*65}")
    print("  RESULT INTERPRETATION")
    print(f"{'='*65}")
    best = df.loc[df['mAP5095'].idxmax()]
    print(f"\n  Best mAP@0.5:0.95 = {best['mAP5095']:.4f} "
          f"at τ={best['hr_trigger_conf']:.2f}  "
          f"({best['avg_energy_saved']:.1f}% energy saved)")

    print(f"\n  Thresholds within ±2.5% of HR accuracy ({STEP1_HR_MAP5095:.4f}):")
    found = False
    for _, r in df.iterrows():
        if r['mAP5095'] >= STEP1_HR_MAP5095 - 0.025:
            print(f"    τ={r['hr_trigger_conf']:.2f} → "
                  f"mAP@0.5:0.95={r['mAP5095']:.4f}  "
                  f"energy_saved={r['avg_energy_saved']:.1f}%  ← paper claim")
            found = True
    if not found:
        print("    None within ±2.5% — try lower thresholds or check pipeline")

    print(f"\n  Outputs saved to: {RESULTS_DIR}")
