"""
step2_evaluate.py  —  Fixed v2
================================
KEY FIX: Always-HR and Always-LR baselines are now evaluated using
the SAME custom per-image AP evaluator as the Adaptive pipeline.

Previously:
  Step 1 baselines used YOLO's built-in val() → COCO global mAP
  Step 2 adaptive used custom per-image AP
  → Incomparable numbers → Adaptive appeared to "beat" HR baseline

Now:
  ALL THREE systems evaluated with same custom evaluator on same test set
  → Fair apple-to-apple comparison → Plot is meaningful
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

# ── Paths (from config, dataset-agnostic) ─────────────────────────────────────
ROOT         = Path(__file__).parent
RUNS_DIR     = C.RUNS_DIR
RESULTS_DIR  = C.RESULTS_DIR_STEP2
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LR_MODEL = RUNS_DIR / "both_skip1_640" / "weights" / "best.pt"
HR_MODEL = RUNS_DIR / "adaptive" / "weights" / "best.pt"

TEST_IMG_DIR = C.SOURCE_YOLO / "images" / "val"
TEST_LBL_DIR = C.SOURCE_YOLO / "labels" / "val"

# Confidence threshold sweep
CONF_THRESHOLDS = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
# NOTE: 0.10 removed — identical to 0.15 because lr_detect_conf=0.15

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="colorblind")
DATASET_LABEL = {
    "pennfudan":       "Penn-Fudan",
    "bdd100k":         "BDD100K",
    "coco_person_car": "COCO (Person + Car)",
}.get(C.DATASET, C.DATASET)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION UTILITIES (shared by baseline + adaptive)
# ══════════════════════════════════════════════════════════════════════════════

def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    """YOLO normalized xywh → pixel xyxy."""
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


def compute_iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 1e-6 else 0.0


def compute_ap(pred_boxes: list, gt_boxes: list,
               iou_thresh: float = 0.5) -> float:
    """
    Per-image Average Precision at one IoU threshold.
    
    IMPORTANT: If an image has NO ground truth objects:
      - No predictions made → AP = 1.0 (sensor correctly stayed quiet)
      - Predictions made    → AP = 0.0 (false alarms on empty image)
    This is consistent for ALL systems (baselines + adaptive).
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

    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p   = precision[recall >= thr].max() if (recall >= thr).any() else 0.0
        ap += p / 101.0
    return ap


def evaluate_dataset(all_preds: dict, all_gts: dict) -> dict:
    """
    Compute mAP@0.5 and mAP@0.5:0.95 over the full test set.
    Used identically for baselines AND adaptive pipeline.
    """
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
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


# ══════════════════════════════════════════════════════════════════════════════
#  THE FIX: Evaluate baselines with SAME custom evaluator
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_baseline_model(model_path: Path,
                             img_paths:  list,
                             all_gts:    dict,
                             use_lr_sim: bool = False,
                             row_skip:   int  = 0,
                             col_skip:   int  = 0,
                             conf:       float = 0.25,
                             label:      str  = "") -> dict:
    """
    Run a YOLO model on the test set and evaluate using the SAME
    custom per-image AP calculator used by the adaptive pipeline.

    Args:
        model_path:  Path to best.pt
        img_paths:   List of test image Paths
        all_gts:     Pre-loaded ground truth dict {stem: [boxes]}
        use_lr_sim:  If True, simulate LR readout before inference
                     (use this for the Always-LR baseline)
        row_skip:    Row skip factor (only used if use_lr_sim=True)
        col_skip:    Col skip factor (only used if use_lr_sim=True)
        conf:        Detection confidence threshold
        label:       Human-readable name for printing

    Returns:
        dict with mAP50, mAP5095, energy_pct, energy_saved_pct
    """
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    step_r = row_skip + 1
    step_c = col_skip + 1

    all_preds = {}
    for p in tqdm(img_paths, desc=f"  [{label}] baseline eval"):
        img = cv2.imread(str(p))
        if img is None:
            continue

        # Simulate LR readout if this is the LR baseline
        if use_lr_sim:
            H, W = img.shape[:2]
            lr_native = img[::step_r, ::step_c]
            img = cv2.resize(lr_native, (W, H),
                             interpolation=cv2.INTER_NEAREST)

        result = model.predict(img, conf=conf, iou=0.50, verbose=False)[0]
        dets   = []
        if result.boxes is not None and len(result.boxes) > 0:
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(float)
                dets.append([x1, y1, x2, y2,
                              float(b.conf[0].cpu()),
                              int(b.cls[0].cpu())])
        all_preds[p.stem] = dets

    metrics = evaluate_dataset(all_preds, all_gts)

    # Energy for baselines
    if use_lr_sim:
        energy_pct   = round(100.0 / (step_r * step_c), 2)
        energy_saved = round(100.0 - energy_pct, 2)
    else:
        energy_pct   = 100.0
        energy_saved = 0.0

    print(f"  [{label}]  mAP@0.5={metrics['mAP50']:.4f}  "
          f"mAP@0.5:0.95={metrics['mAP5095']:.4f}  "
          f"pixels={energy_pct:.1f}%  saved={energy_saved:.1f}%")

    return {
        'label':         label,
        'mAP50':         metrics['mAP50'],
        'mAP5095':       metrics['mAP5095'],
        'energy_pct':    energy_pct,
        'energy_saved':  energy_saved,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    print("=" * 65)
    print(f"  Step 2: Adaptive CIS Evaluation — {DATASET_LABEL}")
    print("=" * 65)

    for p, name in [(LR_MODEL, "LR model"), (HR_MODEL, "HR model")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{name} not found at {p}\n"
                f"Run step1_train_all.py first."
            )

    # ── Load test images ───────────────────────────────────────────────────────
    img_paths = sorted(
        list(TEST_IMG_DIR.glob("*.jpg")) +
        list(TEST_IMG_DIR.glob("*.png"))
    )
    if not img_paths:
        raise FileNotFoundError(f"No test images in {TEST_IMG_DIR}")
    print(f"\n  Test images : {len(img_paths):,}")
    print(f"  Classes     : {C.NC} → {C.NAMES}")

    # ── Pre-load all ground truths (done once, shared by all evaluations) ──────
    all_gts = {}
    print("\n  Loading ground truths...")
    for p in tqdm(img_paths, desc="  GT loading", leave=False):
        img  = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        lbl  = TEST_LBL_DIR / (p.stem + ".txt")
        all_gts[p.stem] = load_gt_boxes(lbl, W, H)

    n_with_obj = sum(1 for v in all_gts.values() if len(v) > 0)
    n_bg       = len(all_gts) - n_with_obj
    print(f"  With objects: {n_with_obj:,}  Background frames: {n_bg:,}")

    # ══════════════════════════════════════════════════════════════════════════
    # THE FIX: Evaluate baselines with the SAME custom evaluator
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*55)
    print("  Computing baselines with SAME custom evaluator...")
    print("  (This is the fix — all three systems now use same ruler)")
    print("─"*55)

    # hr_baseline = evaluate_baseline_model(
    #     model_path  = HR_MODEL,
    #     img_paths   = img_paths,
    #     all_gts     = all_gts,
    #     use_lr_sim  = False,          # full resolution
    #     conf        = 0.25,
    #     label       = "Always-HR (100% pixels)"
    # )

    hr_baseline = evaluate_baseline_model(
    model_path = HR_MODEL,
    img_paths  = img_paths,
    all_gts    = all_gts,
    use_lr_sim = False,
    conf       = C.ADAPTIVE["lr_detect_conf"],   # ← 0.15, not 0.25
    label      = "Always-HR (100% pixels)"
    )

    # lr_baseline = evaluate_baseline_model(
    #     model_path  = LR_MODEL,
    #     img_paths   = img_paths,
    #     all_gts     = all_gts,
    #     use_lr_sim  = True,           # simulate LR readout
    #     row_skip    = C.ADAPTIVE["row_skip"],
    #     col_skip    = C.ADAPTIVE["col_skip"],
    #     conf        = 0.25,
    #     label       = f"Always-LR (×2 skip, 25% pixels)"
    # )

    lr_baseline = evaluate_baseline_model(
    model_path = LR_MODEL,
    img_paths  = img_paths,
    all_gts    = all_gts,
    use_lr_sim = True,
    row_skip   = C.ADAPTIVE["row_skip"],
    col_skip   = C.ADAPTIVE["col_skip"],
    conf       = C.ADAPTIVE["lr_detect_conf"],   # ← 0.15, not 0.25
    label      = "Always-LR (×2 skip, 25% pixels)"
    )

    # Save baselines to CSV for reproducibility
    baselines_df = pd.DataFrame([hr_baseline, lr_baseline])
    baselines_df.to_csv(RESULTS_DIR / "baselines_custom_eval.csv", index=False)
    print(f"\n  [✓] Baselines saved: {RESULTS_DIR / 'baselines_custom_eval.csv'}")

    # ── Build adaptive pipeline (models loaded once, reused) ──────────────────
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

    # ── Threshold sweep ────────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("  Running adaptive threshold sweep...")
    print("─"*55)

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
            'hr_trigger_conf':    conf,
            'mAP50':              metrics['mAP50'],
            'mAP5095':            metrics['mAP5095'],
            'avg_energy_pct':     round(avg_energy_ratio * 100, 2),
            'avg_energy_saved':   round((1 - avg_energy_ratio) * 100, 2),
            'avg_hr_area_pct':    round(avg_hr_pct, 2),
            'total_lr_dets':      n_lr_dets_total,
            'total_hr_crops':     n_hr_crops_total,
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

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "adaptive_metrics.csv", index=False)
    print(f"\n[✓] Adaptive results saved: {RESULTS_DIR / 'adaptive_metrics.csv'}")

    return df, hr_baseline, lr_baseline


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_energy_accuracy(df: pd.DataFrame,
                         hr_baseline: dict,
                         lr_baseline: dict) -> None:
    """
    Main paper figure.
    All three systems now use the SAME custom evaluator → fair comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric, ylabel in zip(
        axes,
        ['mAP50', 'mAP5095'],
        ['mAP @ IoU=0.5', 'mAP @ IoU=0.5:0.95']
    ):
        hr_map = hr_baseline[metric]
        lr_map = lr_baseline[metric]

        # ── Adaptive curve ─────────────────────────────────────────────────
        ax.plot(df['avg_energy_saved'], df[metric],
                marker='o', linewidth=2.5, markersize=9,
                color='steelblue', label='Adaptive LR+HR (Ours)', zorder=5)
        for _, r in df.iterrows():
            ax.annotate(f"τ={r['hr_trigger_conf']:.2f}",
                        (r['avg_energy_saved'], r[metric]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7.5, color='steelblue')

        # ── Always-LR baseline ─────────────────────────────────────────────
        ax.scatter([lr_baseline['energy_saved']], [lr_map],
                   marker='s', s=130, color='darkorange', zorder=6,
                   label=f"Always-LR (×2 skip, 25% pixels)\n"
                         f"mAP={lr_map:.4f}  saved={lr_baseline['energy_saved']:.0f}%")
        ax.axhline(lr_map, color='darkorange',
                   linestyle=':', linewidth=1.2, alpha=0.6)

        # ── Always-HR baseline ─────────────────────────────────────────────
        ax.scatter([hr_baseline['energy_saved']], [hr_map],
                   marker='^', s=130, color='green', zorder=6,
                   label=f"Always-HR (100% pixels)\n"
                         f"mAP={hr_map:.4f}")
        ax.axhline(hr_map, color='green',
                   linestyle='--', linewidth=1.2, alpha=0.6)

        # ── ±2.5% acceptable band around HR ───────────────────────────────
        ax.axhspan(hr_map - 0.025, hr_map + 0.025,
                   alpha=0.08, color='green',
                   label='±2.5% of HR accuracy')

        ax.set_xlabel("Sensor Front-End Energy Saved (%)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Adaptive CIS: Energy vs {ylabel}\n"
                     f"({DATASET_LABEL}, YOLO11n)", fontsize=11)
        ax.legend(fontsize=8.5, loc='lower left')
        ax.set_xlim(-3, 103)
        all_vals = list(df[metric]) + [hr_map, lr_map]
        ax.set_ylim(max(0, min(all_vals)-0.05), min(1, max(all_vals)+0.05))
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.grid(True, alpha=0.35)

    plt.suptitle(
        f"Step 2: Adaptive LR+HR Pipeline — {DATASET_LABEL}  |  YOLO11n\n"
        f"All three systems evaluated with SAME custom per-image AP evaluator",
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    out = RESULTS_DIR / "energy_accuracy_step2.png"
    plt.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[✓] Plot saved: {out}")


def plot_qualitative(n_samples: int = 4) -> None:
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
    cls_names = C.NAMES

    # Pick images that have at least one labeled object
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
        print("[WARN] No labeled test images found.")
        return

    fig, axes = plt.subplots(len(img_paths), 3,
                             figsize=(16, 5*len(img_paths)), squeeze=False)
    col_titles = [
        '(a) Original HR + Ground Truth',
        '(b) LR Readout + LR Detections',
        '(c) Adaptive: HR Crops + Final Detections',
    ]

    for row_idx, img_path in enumerate(img_paths):
        hr_img = cv2.imread(str(img_path))
        H, W   = hr_img.shape[:2]
        lbl    = TEST_LBL_DIR / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(lbl, W, H)
        dets, energy_info, debug = pipeline.process_frame(
            hr_img, hr_trigger_conf=C.ADAPTIVE["hr_trigger_conf"]
        )

        # ── Panel (a): HR + GT ─────────────────────────────────────────────
        ax = axes[row_idx][0]
        disp = cv2.cvtColor(hr_img.copy(), cv2.COLOR_BGR2RGB)
        for gt in gt_boxes:
            x1,y1,x2,y2 = [int(v) for v in gt['box']]
            cv2.rectangle(disp,(x1,y1),(x2,y2),(0,200,0),2)
            lname = cls_names[gt['cls']] if gt['cls'] < len(cls_names) else str(gt['cls'])
            cv2.putText(disp, lname, (x1, max(0,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
        ax.imshow(disp); ax.axis('off')
        if row_idx == 0: ax.set_title(col_titles[0], fontsize=10, pad=6)
        ax.set_ylabel(f"Sample {row_idx+1}", fontsize=9)

        # ── Panel (b): LR + LR detections ─────────────────────────────────
        ax = axes[row_idx][1]
        lr_disp = cv2.cvtColor(debug['lr_img'].copy(), cv2.COLOR_BGR2RGB)
        for d in debug['lr_dets']:
            x1,y1,x2,y2,conf,cls_id = d
            cls_id = int(cls_id)
            lname = cls_names[cls_id] if cls_id < len(cls_names) else str(cls_id)
            cv2.rectangle(lr_disp,(int(x1),int(y1)),(int(x2),int(y2)),(255,165,0),2)
            cv2.putText(lr_disp, f"{lname} {conf:.2f}",
                        (int(x1), max(0,int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,165,0), 1)
        ax.imshow(lr_disp); ax.axis('off')
        if row_idx == 0: ax.set_title(col_titles[1], fontsize=10, pad=6)
        ax.text(4, H-10,
                f"LR pixels: {energy_info['lr_pct']:.1f}%  ({C.ADAPTIVE['row_skip']+1}× skip)",
                color='white', fontsize=8,
                bbox=dict(facecolor='darkorange', alpha=0.7, pad=2))

        # ── Panel (c): Adaptive — HR crops + final detections ─────────────
        ax = axes[row_idx][2]
        adapt_disp = cv2.cvtColor(hr_img.copy(), cv2.COLOR_BGR2RGB)
        overlay = adapt_disp.copy()
        for (cx1,cy1,cx2,cy2) in debug['hr_crops']:
            cv2.rectangle(overlay,(cx1,cy1),(cx2,cy2),(100,149,237),-1)
        adapt_disp = cv2.addWeighted(overlay, 0.25, adapt_disp, 0.75, 0)
        for (cx1,cy1,cx2,cy2) in debug['hr_crops']:
            cv2.rectangle(adapt_disp,(cx1,cy1),(cx2,cy2),(30,100,255),2)
        for d in dets:
            x1,y1,x2,y2,conf,cls_id = d
            cls_id = int(cls_id)
            lname = cls_names[cls_id] if cls_id < len(cls_names) else str(cls_id)
            cv2.rectangle(adapt_disp,(int(x1),int(y1)),(int(x2),int(y2)),(220,30,30),2)
            cv2.putText(adapt_disp, f"{lname} {conf:.2f}",
                        (int(x1), max(0,int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220,30,30), 1)
        ax.imshow(adapt_disp); ax.axis('off')
        if row_idx == 0: ax.set_title(col_titles[2], fontsize=10, pad=6)

        # FIX: cap total_pct display at 100%
        total_display = min(energy_info['total_pct'], 100.0)
        ax.text(4, H-10,
                f"Total: {total_display:.1f}%  "
                f"(LR:{energy_info['lr_pct']:.1f}% + HR:{energy_info['hr_pct']:.1f}%)",
                color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.7, pad=2))

    plt.suptitle(
        f"Qualitative Results — Adaptive CIS  ({DATASET_LABEL})\n"
        "Green=GT  |  Orange=LR detections  |  "
        "Blue=HR-activated regions  |  Red=Final detections",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out = RESULTS_DIR / "qualitative_samples.png"
    plt.savefig(str(out), dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[✓] Qualitative plot saved: {out}")


def print_latex_table(df: pd.DataFrame,
                      hr_baseline: dict,
                      lr_baseline: dict) -> None:
    print("\n" + "="*70)
    print("  LaTeX Table")
    print("="*70)

    rows = [
        ("Always-HR (full sensor)",
         hr_baseline['energy_pct'], hr_baseline['energy_saved'],
         hr_baseline['mAP50'],     hr_baseline['mAP5095']),
        ("Always-LR (×2 skip, ours)",
         lr_baseline['energy_pct'], lr_baseline['energy_saved'],
         lr_baseline['mAP50'],      lr_baseline['mAP5095']),
    ]
    for _, r in df.iterrows():
        rows.append((
            f"Adaptive $\\tau$={r['hr_trigger_conf']:.2f} (ours)",
            round(r['avg_energy_pct'], 1),
            round(r['avg_energy_saved'], 1),
            r['mAP50'], r['mAP5095'],
        ))

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(f"\\caption{{Adaptive CIS results on {DATASET_LABEL} (YOLO11n). "
          r"All systems evaluated with same per-image AP protocol. "
          r"$\tau$ = HR trigger confidence threshold.}}")
    print(r"\label{tab:adaptive_results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Pixels (\%)} & \textbf{Energy Saved (\%)} "
          r"& \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} \\")
    print(r"\midrule")
    for name, px, saved, m50, m5095 in rows:
        print(f"  {name} & {px:.1f} & {saved:.1f} & {m50:.4f} & {m5095:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Run evaluation (baselines + adaptive sweep)
    df, hr_baseline, lr_baseline = run_evaluation()

    # 2. Print results
    print("\n" + "="*65)
    print("  RESULTS SUMMARY")
    print("="*65)
    print(f"\n  Always-HR  : mAP@0.5={hr_baseline['mAP50']:.4f}  "
          f"mAP@0.5:0.95={hr_baseline['mAP5095']:.4f}  saved=0%")
    print(f"  Always-LR  : mAP@0.5={lr_baseline['mAP50']:.4f}  "
          f"mAP@0.5:0.95={lr_baseline['mAP5095']:.4f}  saved=75%")
    print(f"\n  Adaptive sweep:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(df.to_string(index=False))

    # 3. Generate figures
    print("\n[→] Generating paper figures...")
    plot_energy_accuracy(df, hr_baseline, lr_baseline)
    plot_qualitative(n_samples=4)
    print_latex_table(df, hr_baseline, lr_baseline)

    # 4. Interpretation guide
    print(f"\n{'='*65}")
    print("  WHAT TO LOOK FOR IN THE NEW PLOT")
    print(f"{'='*65}")
    hr_map = hr_baseline['mAP5095']
    print(f"\n  HR baseline (same evaluator): {hr_map:.4f}")
    print(f"\n  Adaptive thresholds within ±2.5% of HR:")
    found = False
    for _, r in df.iterrows():
        if r['mAP5095'] >= hr_map - 0.025:
            print(f"    τ={r['hr_trigger_conf']:.2f} → "
                  f"mAP@0.5:0.95={r['mAP5095']:.4f}  "
                  f"saved={r['avg_energy_saved']:.1f}%  ← paper claim ✓")
            found = True
    if not found:
        print("    None within ±2.5% — adaptive is below HR. "
              "Consider increasing HR crop context (padding_ratio=0.15).")

    print(f"\n[✓] All outputs in: {RESULTS_DIR}")
