"""
config.py — Adaptive CIS experiment configuration.
Switch DATASET = "pennfudan" | "bdd100k" | "coco_person_car" | "pascal_person_car"
ALL paths, class names, variants, and training params update automatically.
Results saved to separate namespaced folders — no cross-contamination.
"""
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CHANGE THIS ONE LINE TO SWITCH DATASET
# ══════════════════════════════════════════════════════════════════════════════
DATASET = "pascal_person_car"   # "pennfudan" | "bdd100k" | "coco_person_car" | "pascal_person_car"


# ── Paths (auto-namespaced per dataset, no cross-contamination) ────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"


SOURCE_YOLO       = DATA_DIR / f"{DATASET}_yolo"
OUTPUT_DIR        = DATA_DIR / "lr_variants"  / DATASET
RUNS_DIR          = ROOT     / "runs"  / "step1" / DATASET
RESULTS_DIR       = ROOT     / "results" / "step1" / DATASET
RESULTS_DIR_STEP2 = ROOT     / "results" / "step2" / DATASET


# Test split paths (used by step2_evaluate.py)
TEST_IMG_DIR = SOURCE_YOLO / "images" / "test"
TEST_LBL_DIR = SOURCE_YOLO / "labels" / "test"


# ── Dataset metadata ───────────────────────────────────────────────────────────
DATASET_META = {
    "pennfudan": {
        "nc":       1,
        "names":    ["person"],
        "img_w":    640,
        "img_h":    640,
        "batch":    16,
        "epochs":   100,
        "patience": 20,
    },
    "bdd100k": {
        "nc":       10,
        "names":    [
            "pedestrian", "rider", "car", "truck", "bus",
            "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
        ],
        "img_w":    1280,
        "img_h":    720,
        "batch":    16,
        "epochs":   50,
        "patience": 10,
    },
    "coco_person_car": {
        "nc":       2,
        "names":    ["person", "car"],
        "img_w":    640,
        "img_h":    640,
        "batch":    16,
        "epochs":   80,
        "patience": 30,
    },
    "pascal_person_car": {        # ← NEW
        "nc":       2,
        "names":    ["person", "car"],
        "img_w":    640,
        "img_h":    640,
        "batch":    16,
        "epochs":   100,
        "patience": 20,
    },
}


_meta    = DATASET_META[DATASET]
NC       = _meta["nc"]
NAMES    = _meta["names"]
SPLITS   = ["train", "val", "test"]
EPOCHS   = _meta["epochs"]
PATIENCE = _meta["patience"]
BATCH    = _meta["batch"]


# ── LR Variant Definitions ─────────────────────────────────────────────────────
_PENNFUDAN_VARIANTS = {
    "hr_640":            dict(row_skip=0, col_skip=0, upsample=True,  imgsz=640, label="HR Baseline"),
    "row_skip1_640":     dict(row_skip=1, col_skip=0, upsample=True,  imgsz=640, label="Row-skip×2, Up"),
    "row_skip1_native":  dict(row_skip=1, col_skip=0, upsample=False, imgsz=320, label="Row-skip×2, Native"),
    "row_skip3_640":     dict(row_skip=3, col_skip=0, upsample=True,  imgsz=640, label="Row-skip×4, Up"),
    "row_skip3_native":  dict(row_skip=3, col_skip=0, upsample=False, imgsz=160, label="Row-skip×4, Native"),
    "both_skip1_640":    dict(row_skip=1, col_skip=1, upsample=True,  imgsz=640, label="Row+Col-skip×2, Up"),
    "both_skip1_native": dict(row_skip=1, col_skip=1, upsample=False, imgsz=320, label="Row+Col-skip×2, Native"),
    "both_skip3_640":    dict(row_skip=3, col_skip=3, upsample=True,  imgsz=640, label="Row+Col-skip×4, Up"),
    "both_skip3_native": dict(row_skip=3, col_skip=3, upsample=False, imgsz=160, label="Row+Col-skip×4, Native"),
    "both_skip7_640":    dict(row_skip=7, col_skip=7, upsample=True,  imgsz=640, label="Row+Col-skip×8, Up"),
}


_BDD100K_VARIANTS = {
    "hr_640":         dict(row_skip=0, col_skip=0, upsample=True, imgsz=640, label="HR Baseline (100% pixels)"),
    "both_skip1_640": dict(row_skip=1, col_skip=1, upsample=True, imgsz=640, label="LR ×2 skip (25% pixels)"),
    "both_skip3_640": dict(row_skip=3, col_skip=3, upsample=True, imgsz=640, label="LR ×4 skip (6.25% pixels)"),
    "both_skip7_640": dict(row_skip=7, col_skip=7, upsample=True, imgsz=640, label="LR ×8 skip (1.56% pixels)"),
}


_COCO_VARIANTS = {
    "hr_640":         dict(row_skip=0, col_skip=0, upsample=True, imgsz=640, label="HR Baseline (100% pixels)"),
    "both_skip1_640": dict(row_skip=1, col_skip=1, upsample=True, imgsz=640, label="LR ×2 skip (25% pixels)"),
    "both_skip3_640": dict(row_skip=3, col_skip=3, upsample=True, imgsz=640, label="LR ×4 skip (6.25% pixels)"),
    "both_skip7_640": dict(row_skip=7, col_skip=7, upsample=True, imgsz=640, label="LR ×8 skip (1.56% pixels)"),
}


# Pascal VOC person+car: same 4-variant set as COCO for direct comparison
# - Same 2 classes (person=0, car=1), diverse non-driving scenes
# - 1324 train / 1256 val images (filtered from VOC 2007's 20 classes)
# - val reused as test (no official test.txt in trainval tarball)
_PASCAL_VARIANTS = {                                                    # ← NEW
    "hr_640":         dict(row_skip=0, col_skip=0, upsample=True, imgsz=640, label="HR Baseline (100% pixels)"),
    "both_skip1_640": dict(row_skip=1, col_skip=1, upsample=True, imgsz=640, label="LR ×2 skip (25% pixels)"),
    "both_skip3_640": dict(row_skip=3, col_skip=3, upsample=True, imgsz=640, label="LR ×4 skip (6.25% pixels)"),
    "both_skip7_640": dict(row_skip=7, col_skip=7, upsample=True, imgsz=640, label="LR ×8 skip (1.56% pixels)"),
}


# ── VARIANT SELECTOR ───────────────────────────────────────────────────────────
if DATASET == "pennfudan":
    VARIANTS = _PENNFUDAN_VARIANTS
elif DATASET == "bdd100k":
    VARIANTS = _BDD100K_VARIANTS
elif DATASET == "coco_person_car":
    VARIANTS = _COCO_VARIANTS
elif DATASET == "pascal_person_car":                                    # ← NEW
    VARIANTS = _PASCAL_VARIANTS
else:
    raise ValueError(f"Unknown DATASET: '{DATASET}'. "
                     f"Choose 'pennfudan', 'bdd100k', 'coco_person_car', or 'pascal_person_car'.")


# ── Sensor Energy Model ────────────────────────────────────────────────────────
def pixel_ratio(row_skip: int, col_skip: int) -> float:
    return 1.0 / ((row_skip + 1) * (col_skip + 1))


def energy_savings_pct(row_skip: int, col_skip: int) -> float:
    return (1.0 - pixel_ratio(row_skip, col_skip)) * 100.0


# ── Model + Hardware ───────────────────────────────────────────────────────────
MODEL   = "yolo11n.pt"
DEVICE  = "0"
WORKERS = 4


# ── Adaptive Pipeline Parameters (Step 2) ─────────────────────────────────────
ADAPTIVE = {
    "lr_detect_conf":  0.15,
    "hr_trigger_conf": 0.35,
    "padding_ratio":   0.07,
    "nms_iou":         0.40,
    "min_crop_px":     24,
    "row_skip":        3,
    "col_skip":        3,
}
