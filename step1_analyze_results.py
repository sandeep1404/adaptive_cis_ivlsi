"""
Step 1C: Collect all trained model metrics, compute energy savings,
         and produce paper-ready figures and tables.

Outputs (all in results/step1/):
    metrics_table.csv              ← full table for the paper
    energy_accuracy_tradeoff.png   ← the main paper figure (mAP vs energy saved)
    mAP50_bar_chart.png            ← bar chart comparing all variants
    native_vs_upsampled.png        ← key comparison: native LR vs blocky upsample
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from pathlib import Path
from config import (VARIANTS, RUNS_DIR, RESULTS_DIR,
                    pixel_ratio, energy_savings_pct)

sns.set_theme(style="whitegrid", palette="colorblind")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load metrics from Ultralytics results.csv ─────────────────────────────

def load_best_metrics(variant_name: str) -> dict:
    """
    Extract best-epoch metrics from Ultralytics training output.
    Ultralytics saves: runs/step1/{variant_name}/results.csv

    Returns dict with mAP50, mAP50_95, precision, recall, or None if missing.
    """
    results_csv = RUNS_DIR / variant_name / "results.csv"

    if not results_csv.exists():
        print(f"  [WARN] No results.csv for {variant_name}")
        return None

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Ultralytics adds spaces to col names

    # Ultralytics column names (may vary slightly by version; handle both)
    col_map = {
        "metrics/mAP50(B)":    "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
        "metrics/precision(B)":"precision",
        "metrics/recall(B)":   "recall",
    }

    # Get best row (highest mAP50)
    map50_col = next((c for c in df.columns if "mAP50" in c and "95" not in c), None)
    if map50_col is None:
        print(f"  [WARN] mAP50 column not found in {results_csv}. Columns: {list(df.columns)}")
        return None

    best_idx = df[map50_col].idxmax()
    best_row = df.loc[best_idx]

    out = {}
    for orig_col, clean_name in col_map.items():
        # fuzzy match because Ultralytics adds spaces
        matched = next((c for c in df.columns if orig_col.strip() in c.strip()), None)
        if matched:
            out[clean_name] = round(float(best_row[matched]), 4)
        else:
            out[clean_name] = None

    out["best_epoch"] = int(best_row.get("epoch", best_idx))
    return out


# ── 2. Build the master metrics table ─────────────────────────────────────────

def build_metrics_table() -> pd.DataFrame:
    rows = []
    for vname, vcfg in VARIANTS.items():
        metrics = load_best_metrics(vname)
        if metrics is None:
            continue

        rs  = vcfg["row_skip"]
        cs  = vcfg["col_skip"]
        ups = vcfg["upsample"]
        pr  = pixel_ratio(rs, cs)
        es  = energy_savings_pct(rs, cs)

        # Determine skip pattern label for the table
        if rs == 0 and cs == 0:
            pattern = "None (HR)"
        elif cs == 0:
            pattern = f"Row-only ×{rs+1}"
        else:
            pattern = f"Row+Col ×{rs+1}"

        row = {
            "Variant":          vname,
            "Label":            vcfg["label"],
            "Skip Pattern":     pattern,
            "Upsample":         "Yes" if ups else "No",
            "Training imgsz":   vcfg["imgsz"],
            "Pixels Read (%)":  round(pr * 100, 2),
            "Energy Saved (%)": round(es, 2),
            "mAP@0.5":          metrics.get("mAP50"),
            "mAP@0.5:0.95":     metrics.get("mAP50_95"),
            "Precision":        metrics.get("precision"),
            "Recall":           metrics.get("recall"),
            "Best Epoch":       metrics.get("best_epoch"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] No metrics found. Train models first (step1_train_all.py).")
        return df

    # Sort by energy saved ascending (HR first → most aggressive skip last)
    df = df.sort_values("Energy Saved (%)", ascending=True).reset_index(drop=True)

    out_path = RESULTS_DIR / "metrics_table.csv"
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved: {out_path}")
    return df


# ── 3. Paper Figure 1 — Energy vs Accuracy Trade-off ─────────────────────────

def plot_energy_accuracy(df: pd.DataFrame) -> None:
    """
    Main paper figure: mAP@0.5 vs Sensor Energy Saved (%).
    Separate curves for upsampled vs native LR.
    """
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Separate by upsample mode
    df_up  = df[df["Upsample"] == "Yes"].copy()
    df_nat = df[df["Upsample"] == "No"].copy()

    # Plot upsampled variants
    if not df_up.empty:
        ax.plot(df_up["Energy Saved (%)"], df_up["mAP@0.5"],
                marker="o", linewidth=2, markersize=8,
                label="Upsampled to original size", color="steelblue")
        for _, r in df_up.iterrows():
            ax.annotate(r["Skip Pattern"],
                        (r["Energy Saved (%)"], r["mAP@0.5"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7.5, color="steelblue")

    # Plot native LR variants
    if not df_nat.empty:
        ax.plot(df_nat["Energy Saved (%)"], df_nat["mAP@0.5"],
                marker="s", linewidth=2, markersize=8,
                linestyle="--",
                label="Native LR resolution", color="darkorange")
        for _, r in df_nat.iterrows():
            ax.annotate(r["Skip Pattern"],
                        (r["Energy Saved (%)"], r["mAP@0.5"]),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=7.5, color="darkorange")

    # HR baseline reference line
    hr_row = df[df["Variant"] == "hr_640"]
    if not hr_row.empty:
        hr_map = hr_row["mAP@0.5"].values[0]
        ax.axhline(hr_map, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
                   label=f"HR Baseline mAP@0.5 = {hr_map:.3f}")
        ax.axhspan(hr_map - 0.025, hr_map + 0.025, alpha=0.05, color="gray")
        ax.text(2, hr_map + 0.005, "±2.5% acceptable accuracy range",
                fontsize=8, color="gray")

    ax.set_xlabel("Sensor Front-End Energy Saved (%)", fontsize=12)
    ax.set_ylabel("mAP @ IoU=0.5", fontsize=12)
    ax.set_title("Resolution vs Detection Accuracy Trade-off\n"
                 "(Penn-Fudan dataset, YOLO11n)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(left=-2)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out = RESULTS_DIR / "energy_accuracy_tradeoff.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out}")


# ── 4. mAP Bar Chart ──────────────────────────────────────────────────────────

def plot_map_bar(df: pd.DataFrame) -> None:
    """Bar chart of mAP@0.5 and mAP@0.5:0.95 for all variants."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(df))
    width  = 0.35

    bars1 = ax.bar(x - width/2, df["mAP@0.5"],      width, label="mAP@0.5",      alpha=0.85)
    bars2 = ax.bar(x + width/2, df["mAP@0.5:0.95"], width, label="mAP@0.5:0.95", alpha=0.85)

    # Color HR baseline differently
    hr_indices = df[df["Variant"] == "hr_640"].index.tolist()
    for i in hr_indices:
        bars1[i].set_color("green")
        bars2[i].set_color("darkgreen")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Skip Pattern']}\n({'Up' if r['Upsample']=='Yes' else 'Native'})"
                        for _, r in df.iterrows()],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mAP Score", fontsize=12)
    ax.set_title("Detection Accuracy Across All LR Variants\n"
                 "(Penn-Fudan, YOLO11n)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.4)

    # Annotate pixel read % on top of each bar pair
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, max(row["mAP@0.5"], row["mAP@0.5:0.95"]) + 0.02,
                f"{row['Pixels Read (%)']:.1f}%\npx",
                ha="center", va="bottom", fontsize=6.5, color="dimgray")

    plt.tight_layout()
    out = RESULTS_DIR / "mAP50_bar_chart.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out}")


# ── 5. Native vs Upsampled Comparison ────────────────────────────────────────

def plot_native_vs_upsampled(df: pd.DataFrame) -> None:
    """
    Head-to-head: for the SAME skip factor,
    compare accuracy of native LR model vs upsampled-to-640 model.
    """
    if df.empty:
        return

    skip_pairs = [
        ("row_skip1_native",  "row_skip1_640",  "Row-skip×2"),
        ("row_skip3_native",  "row_skip3_640",  "Row-skip×4"),
        ("both_skip1_native", "both_skip1_640", "Row+Col-skip×2"),
        ("both_skip3_native", "both_skip3_640", "Row+Col-skip×4"),
    ]

    pair_labels, native_maps, up_maps = [], [], []

    for nat_name, up_name, label in skip_pairs:
        nat_row = df[df["Variant"] == nat_name]
        up_row  = df[df["Variant"] == up_name]
        if nat_row.empty or up_row.empty:
            continue
        pair_labels.append(label)
        native_maps.append(nat_row["mAP@0.5"].values[0])
        up_maps.append(up_row["mAP@0.5"].values[0])

    if not pair_labels:
        return

    x     = np.arange(len(pair_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, native_maps, width, label="Native LR (true sensor resolution)",
           color="darkorange", alpha=0.85)
    ax.bar(x + width/2, up_maps,     width, label="Upsampled LR (blocky to original size)",
           color="steelblue", alpha=0.85)

    # HR baseline
    hr_row = df[df["Variant"] == "hr_640"]
    if not hr_row.empty:
        ax.axhline(hr_row["mAP@0.5"].values[0], color="green", linestyle="--",
                   linewidth=1.5, label="HR Baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=11)
    ax.set_ylabel("mAP @ IoU=0.5", fontsize=12)
    ax.set_title("Native LR vs Upsampled LR: Which is better?\n"
                 "(Same skip pattern, different training/inference resolution)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    out = RESULTS_DIR / "native_vs_upsampled.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out}")


# ── 6. Print Paper-ready Table ────────────────────────────────────────────────

def print_latex_table(df: pd.DataFrame) -> None:
    """Print a LaTeX table snippet you can paste into the paper."""
    if df.empty:
        return

    print("\n" + "="*60)
    print("  LaTeX Table Snippet (paste into paper)")
    print("="*60)

    cols = ["Skip Pattern", "Upsample", "Pixels Read (%)",
            "Energy Saved (%)", "mAP@0.5", "mAP@0.5:0.95"]

    sub = df[cols].copy()
    latex = sub.to_latex(
        index=False,
        float_format="%.3f",
        caption="Resolution sweep results on Penn-Fudan. "
                "mAP@0.5 and mAP@0.5:0.95 (YOLOv11n). "
                "Energy Saved = fraction of sensor pixels not read.",
        label="tab:resolution_sweep",
        escape=False,
    )
    print(latex)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Step 1C: Analyzing Results")
    print("=" * 60)

    df = build_metrics_table()

    if df.empty:
        print("[!] No results yet. Complete training first.")
    else:
        print("\nMetrics Table:")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(df.to_string(index=False))

        plot_energy_accuracy(df)
        plot_map_bar(df)
        plot_native_vs_upsampled(df)
        print_latex_table(df)

        print(f"\n[✓] All outputs saved to: {RESULTS_DIR}")
        print("\n  KEY QUESTION FOR YOUR PAPER:")
        hr_row = df[df["Variant"] == "hr_640"]
        if not hr_row.empty:
            hr_map = hr_row["mAP@0.5"].values[0]
            for _, row in df.iterrows():
                if row["Energy Saved (%)"] > 30 and row["mAP@0.5"] is not None:
                    drop = hr_map - row["mAP@0.5"]
                    print(f"  {row['Variant']:<28} "
                          f"saves {row['Energy Saved (%)']:>5.1f}% energy  "
                          f"with mAP drop = {drop:.4f} "
                          f"({'✓ acceptable' if drop < 0.025 else '✗ too high'})")
