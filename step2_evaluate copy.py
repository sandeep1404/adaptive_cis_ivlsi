"""
Step 2: Evaluation + Paper Figures

Runs the AdaptiveCISPipeline over the Penn-Fudan val set at multiple
LR confidence thresholds to generate the energy-accuracy trade-off curve.

Compares three systems:
  [A] Always-LR   : mAP from Step 1 both_skip1_640  (25% pixels)
  [B] Always-HR   : mAP from Step 1 hr_640           (100% pixels)
  [C] Adaptive    : Step 2 pipeline at various conf thresholds

Outputs (results/step2/):
  adaptive_metrics.csv          ← per-threshold mAP + energy table
  energy_accuracy_step2.png     ← THE main paper figure
  qualitative_samples.png       ← 3-panel visual for paper
  latex_table.txt               ← copy-paste table for paper
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from pathlib import Path
from tqdm import tqdm

from step2_pipeline import AdaptiveCISPipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
RUNS_DIR     = ROOT / "runs" / "step1"
DATA_DIR     = ROOT / "data" / "pennfudan_yolo"
RESULTS_DIR  = ROOT / "results" / "step2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LR_MODEL  = RUNS_DIR / "both_skip1_640" / "weights" / "best.pt"
HR_MODEL  = RUNS_DIR / "hr_640"         / "weights" / "best.pt"

VAL_IMG_DIR = DATA_DIR / "images" / "test"
VAL_LBL_DIR = DATA_DIR / "labels" / "test"

# Step 1 baselines (from your metrics_table.csv)
STEP1_LR_MAP50    = 0.9911   # both_skip1_640
STEP1_LR_MAP5095  = 0.8418
STEP1_LR_ENERGY   = 25.0     # % pixels read

STEP1_HR_MAP50    = 0.9915   # hr_640
STEP1_HR_MAP5095  = 0.8075
STEP1_HR_ENERGY   = 100.0

# Confidence thresholds to sweep (controls HR activation rate)
CONF_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

sns.set_theme(style="whitegrid", palette="colorblind")


# ── mAP Calculator ─────────────────────────────────────────────────────────────

def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    """
    Load YOLO-format labels → list of {'cls': int, 'box': [x1,y1,x2,y2] pixels}.
    Normalized YOLO xywh → absolute xyxy pixel coords.
    """
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


def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 1e-6 else 0.0


def compute_ap(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    Compute Average Precision for a single IoU threshold.
    pred_boxes: list of [x1,y1,x2,y2, conf, cls], sorted by conf descending
    gt_boxes:   list of {'cls': int, 'box': [x1,y1,x2,y2]}
    Returns scalar AP.
    """
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0

    pred_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    matched     = [False] * len(gt_boxes)
    tp          = []
    fp          = []

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

    tp_cum = np.cumsum(tp).astype(float)
    fp_cum = np.cumsum(fp).astype(float)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall    = tp_cum / (len(gt_boxes) + 1e-9)

    # 101-point interpolated AP (COCO style)
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p = precision[recall >= thr].max() if (recall >= thr).any() else 0.0
        ap += p / 101.0
    return ap


def evaluate_dataset(all_preds: dict, all_gts: dict) -> dict:
    """
    all_preds: {img_stem: [[x1,y1,x2,y2,conf,cls], ...]}
    all_gts:   {img_stem: [{'cls':int,'box':[x1,y1,x2,y2]}, ...]}
    Returns mAP@0.5 and mAP@0.5:0.95.
    """
    iou_thresholds_50_95 = np.arange(0.5, 1.0, 0.05)
    aps_50    = []
    aps_5095  = []

    for stem in all_gts:
        preds = all_preds.get(stem, [])
        gts   = all_gts[stem]
        ap50  = compute_ap(preds, gts, iou_thresh=0.50)
        aps_50.append(ap50)

        ap_vals = [compute_ap(preds, gts, iou_thresh=t)
                   for t in iou_thresholds_50_95]
        aps_5095.append(np.mean(ap_vals))

    return {
        'mAP50':   round(np.mean(aps_50),   4),
        'mAP5095': round(np.mean(aps_5095), 4),
    }


# ── Main Evaluation Loop ───────────────────────────────────────────────────────

def run_evaluation() -> pd.DataFrame:
    print("=" * 60)
    print("  Step 2: Adaptive CIS Evaluation")
    print("=" * 60)

    pipeline = AdaptiveCISPipeline(
        lr_model_path = str(LR_MODEL),
        hr_model_path = str(HR_MODEL),
        row_skip      = 1,
        col_skip      = 1,
        padding_ratio = 0.07,
        min_crop_px   = 24,
        hr_conf       = 0.25,
    )

    # Load val images
    img_paths = sorted(
        list(VAL_IMG_DIR.glob("*.jpg")) +
        list(VAL_IMG_DIR.glob("*.png"))
    )
    assert img_paths, f"No images found in {VAL_IMG_DIR}"
    print(f"  Val images: {len(img_paths)}")

    # Pre-load ground truths once
    all_gts = {}
    for p in img_paths:
        img  = cv2.imread(str(p))
        H, W = img.shape[:2]
        lbl  = VAL_LBL_DIR / (p.stem + ".txt")
        all_gts[p.stem] = load_gt_boxes(lbl, W, H)

    results = []

    for conf in CONF_THRESHOLDS:
        print(f"\n[→] LR conf threshold = {conf:.1f}")
        all_preds       = {}
        energy_ratios   = []
        hr_area_pcts    = []
        n_lr_dets_total = 0
        n_hr_crops_total= 0

        for img_path in tqdm(img_paths, desc=f"  conf={conf}", leave=False):
            img = cv2.imread(str(img_path))

            dets, energy_info, debug = pipeline.process_frame(img, hr_trigger_conf=conf)


            all_preds[img_path.stem]  = dets
            energy_ratios.append(energy_info['energy_ratio'])
            hr_area_pcts.append(energy_info['hr_pct'])
            n_lr_dets_total  += len(debug['lr_dets'])
            n_hr_crops_total += len(debug['hr_crops'])

        metrics = evaluate_dataset(all_preds, all_gts)
        avg_energy_ratio = np.mean(energy_ratios)
        avg_hr_pct       = np.mean(hr_area_pcts)

        row = {
            'lr_conf':          conf,
            'mAP50':            metrics['mAP50'],
            'mAP5095':          metrics['mAP5095'],
            'avg_energy_ratio': round(avg_energy_ratio, 4),
            'avg_energy_pct':   round(avg_energy_ratio * 100, 2),
            'avg_energy_saved': round((1 - avg_energy_ratio) * 100, 2),
            'avg_hr_area_pct':  round(avg_hr_pct, 2),
            'total_lr_dets':    n_lr_dets_total,
            'total_hr_crops':   n_hr_crops_total,
        }
        results.append(row)
        print(f"  mAP@0.5={metrics['mAP50']:.4f}  "
              f"mAP@0.5:0.95={metrics['mAP5095']:.4f}  "
              f"energy_used={avg_energy_ratio*100:.1f}%  "
              f"energy_saved={row['avg_energy_saved']:.1f}%")

    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "adaptive_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[✓] Saved: {csv_path}")
    return df


# ── Paper Figures ──────────────────────────────────────────────────────────────

def plot_energy_accuracy(df: pd.DataFrame) -> None:
    """
    THE main paper figure: energy saved vs mAP, showing all three regimes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric, label in zip(
        axes,
        ['mAP50', 'mAP5095'],
        ['mAP @ IoU=0.5', 'mAP @ IoU=0.5:0.95']
    ):
        # Adaptive curve (vary lr_conf)
        ax.plot(df['avg_energy_saved'], df[metric],
                marker='o', linewidth=2.5, markersize=9,
                color='steelblue', label='Adaptive LR+HR (Ours)', zorder=5)

        # Annotate conf thresholds on the curve
        for _, r in df.iterrows():
            ax.annotate(f"τ={r['lr_conf']:.1f}",
                        (r['avg_energy_saved'], r[metric]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7.5, color='steelblue')

        # Always-LR baseline (single point)
        lr_map = STEP1_LR_MAP50 if metric == 'mAP50' else STEP1_LR_MAP5095
        ax.scatter([100 - STEP1_LR_ENERGY], [lr_map],
                   marker='s', s=120, color='darkorange', zorder=6,
                   label=f'Always-LR (both_skip×2)\n25% pixels')
        ax.axhline(lr_map, color='darkorange', linestyle=':', linewidth=1.2, alpha=0.5)

        # Always-HR baseline (single point)
        hr_map = STEP1_HR_MAP50 if metric == 'mAP50' else STEP1_HR_MAP5095
        ax.scatter([0.0], [hr_map],
                   marker='^', s=120, color='green', zorder=6,
                   label=f'Always-HR (full sensor)\n100% pixels')
        ax.axhline(hr_map, color='green', linestyle='--', linewidth=1.2, alpha=0.5)

        # Acceptable range band (HR ± 2.5%)
        ax.axhspan(hr_map - 0.025, hr_map + 0.025,
                   alpha=0.07, color='green',
                   label='±2.5% of HR accuracy')

        ax.set_xlabel("Sensor Front-End Energy Saved (%)", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"Adaptive CIS: Energy vs Accuracy\n({label})", fontsize=12)
        ax.legend(fontsize=8.5, loc='lower left')
        ax.set_xlim(-3, 103)
        ax.set_ylim(bottom=max(0, df[metric].min() - 0.05))
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.grid(True, alpha=0.35)

    plt.suptitle("Step 2 Results: Adaptive LR+HR Pipeline (Penn-Fudan, YOLO11n)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "energy_accuracy_step2.png"
    plt.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")


def plot_qualitative(n_samples: int = 3) -> None:
    """
    3-panel qualitative figure for the paper:
    Left:   Original HR image (ground truth boxes)
    Middle: Simulated LR image + LR detections
    Right:  Adaptive output — HR crops activated (blue) + final detections (red)
    """
    pipeline = AdaptiveCISPipeline(
        lr_model_path = str(LR_MODEL),
        hr_model_path = str(HR_MODEL),
        row_skip=1, 
        col_skip=1, 
        padding_ratio=0.07, 
        hr_conf=0.25, 
        nms_iou = 0.40,
        lr_detect_conf = 0.15,

    )

    img_paths = sorted(
        list(VAL_IMG_DIR.glob("*.jpg")) +
        list(VAL_IMG_DIR.glob("*.png"))
    )[:n_samples]

    fig, axes = plt.subplots(n_samples, 3,
                             figsize=(15, 5 * n_samples), squeeze=False)
    col_titles = ['(a) Original HR + Ground Truth',
                  '(b) LR Readout + LR Detections',
                  '(c) Adaptive: Activated HR Crops + Final Detections']

    for row_idx, img_path in enumerate(img_paths):
        hr_img = cv2.imread(str(img_path))
        H, W   = hr_img.shape[:2]
        lbl    = VAL_LBL_DIR / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(lbl, W, H)

        dets, energy_info, debug = pipeline.process_frame(hr_img, hr_trigger_conf=0.3)

        # ── Panel (a): HR + GT ──────────────────────────────────────────────
        ax = axes[row_idx][0]
        disp = cv2.cvtColor(hr_img.copy(), cv2.COLOR_BGR2RGB)
        for gt in gt_boxes:
            x1, y1, x2, y2 = [int(v) for v in gt['box']]
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,200,0), 2)
        ax.imshow(disp); ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=10, pad=6)
        ax.set_ylabel(f"Sample {row_idx+1}", fontsize=9)

        # ── Panel (b): LR + LR detections ──────────────────────────────────
        ax = axes[row_idx][1]
        lr_disp = cv2.cvtColor(debug['lr_img'].copy(), cv2.COLOR_BGR2RGB)
        for d in debug['lr_dets']:
            x1,y1,x2,y2,conf,_ = d
            cv2.rectangle(lr_disp, (int(x1),int(y1)), (int(x2),int(y2)),
                          (255,165,0), 2)
            cv2.putText(lr_disp, f"{conf:.2f}", (int(x1), max(0,int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,165,0), 1)
        ax.imshow(lr_disp); ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[1], fontsize=10, pad=6)
        ax.text(4, H-8, f"Pixels read: {energy_info['lr_pct']}%",
                color='orange', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=2))

        # ── Panel (c): Adaptive — HR crops + final detections ───────────────
        ax = axes[row_idx][2]
        adapt_disp = cv2.cvtColor(hr_img.copy(), cv2.COLOR_BGR2RGB)
        # Shade HR crop regions (blue)
        overlay = adapt_disp.copy()
        for (cx1, cy1, cx2, cy2) in debug['hr_crops']:
            cv2.rectangle(overlay, (cx1,cy1), (cx2,cy2), (100,149,237), -1)
        adapt_disp = cv2.addWeighted(overlay, 0.25, adapt_disp, 0.75, 0)
        for (cx1, cy1, cx2, cy2) in debug['hr_crops']:
            cv2.rectangle(adapt_disp, (cx1,cy1), (cx2,cy2), (30,100,255), 2)
        # Final HR detections (red)
        for d in dets:
            x1,y1,x2,y2,conf,_ = d
            cv2.rectangle(adapt_disp, (int(x1),int(y1)), (int(x2),int(y2)),
                          (220,30,30), 2)
            cv2.putText(adapt_disp, f"{conf:.2f}", (int(x1), max(0,int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,30,30), 1)
        ax.imshow(adapt_disp); ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[2], fontsize=10, pad=6)
        ax.text(4, H-8,
                f"Total pixels: {energy_info['total_pct']}%  "
                f"(LR:{energy_info['lr_pct']}% + HR:{energy_info['hr_pct']}%)",
                color='white', fontsize=7.5,
                bbox=dict(facecolor='black', alpha=0.6, pad=2))

    plt.suptitle("Qualitative Results — Adaptive CIS Pipeline\n"
                 "Blue shading = sensor switches to HR readout  |  "
                 "Orange = LR detections  |  Red = Final HR-refined detections",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = RESULTS_DIR / "qualitative_samples.png"
    plt.savefig(str(out), dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")


def print_latex_table(df: pd.DataFrame) -> None:
    """Print paper-ready LaTeX table comparing all three modes."""
    print("\n" + "=" * 70)
    print("  LaTeX Table — Paste into paper")
    print("=" * 70)

    # Build comparison table with baselines + best adaptive point
    best_row = df.loc[df['mAP5095'].idxmax()]

    rows = [
        ("Always-HR",       STEP1_HR_ENERGY,   100-STEP1_HR_ENERGY,
         STEP1_HR_MAP50,    STEP1_HR_MAP5095),
        ("Always-LR (Ours)", STEP1_LR_ENERGY,  100-STEP1_LR_ENERGY,
         STEP1_LR_MAP50,    STEP1_LR_MAP5095),
    ]
    for _, r in df.iterrows():
        rows.append((
            f"Adaptive τ={r['lr_conf']:.1f} (Ours)",
            round(r['avg_energy_pct'], 1),
            round(r['avg_energy_saved'], 1),
            r['mAP50'],
            r['mAP5095'],
        ))

    header = (r"\textbf{Method} & \textbf{Pixels Read (\%)} & "
              r"\textbf{Energy Saved (\%)} & "
              r"\textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} \\")

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Adaptive CIS pipeline results on Penn-Fudan dataset "
          r"(YOLO11n). $\tau$ = LR confidence threshold controlling "
          r"HR region activation.}")
    print(r"\label{tab:adaptive_results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(header)
    print(r"\midrule")
    for name, px, saved, m50, m5095 in rows:
        print(f"  {name} & {px:.1f} & {saved:.1f} & "
              f"{m50:.4f} & {m5095:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Run evaluation across all confidence thresholds
    df = run_evaluation()

    # 2. Print results
    print("\nResults Table:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))

    # 3. Generate paper figures
    print("\n[→] Generating plots...")
    plot_energy_accuracy(df)
    plot_qualitative(n_samples=4)
    print_latex_table(df)

    print(f"\n[✓] All Step 2 outputs in: {RESULTS_DIR}")
    print("\n  WHAT TO LOOK FOR:")
    best = df.loc[df['mAP5095'].idxmax()]
    print(f"  Best mAP@0.5:0.95 = {best['mAP5095']:.4f} "
          f"at conf={best['lr_conf']:.1f} "
          f"using {best['avg_energy_pct']:.1f}% of sensor "
          f"(saving {best['avg_energy_saved']:.1f}%)")
    hr_map5095 = STEP1_HR_MAP5095
    for _, r in df.iterrows():
        if r['mAP5095'] >= hr_map5095 - 0.025:
            print(f"  τ={r['lr_conf']:.1f}: within ±2.5% of HR accuracy "
                  f"at {r['avg_energy_saved']:.1f}% energy saving ✓")
