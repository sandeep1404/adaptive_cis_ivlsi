"""
step2_pipeline.py  —  v3  COMPOSITE IMAGE APPROACH
=====================================================
FUNDAMENTAL FIX:
  OLD: LR detect → crop bbox → HR model on tiny crop  (WRONG - out-of-dist)
  NEW: LR detect → paste HR pixels into full LR frame
       → HR model on FULL COMPOSITE IMAGE              (CORRECT)

This matches how a real sensor works:
  - Sensor reads full frame at LR (25% pixels)
  - For triggered ROIs: sensor reads those pixels at HR
  - Result: composite image (HR in ROIs, LR elsewhere)
  - YOLO processes the full composite → correct context + accurate boxes
"""

import cv2
import numpy as np
from pathlib import Path


class AdaptiveCISPipeline:

    def __init__(self,
                 lr_model_path:  str,
                 hr_model_path:  str,
                 row_skip:       int   = 3,
                 col_skip:       int   = 3,
                 padding_ratio:  float = 0.07,
                 min_crop_px:    int   = 32,
                 hr_conf:        float = 0.50,
                 nms_iou:        float = 0.40,
                 lr_detect_conf: float = 0.10):

        from ultralytics import YOLO
        print(f"[AdaptiveCIS] Loading LR model: {lr_model_path}")
        self.lr_model = YOLO(lr_model_path)
        print(f"[AdaptiveCIS] Loading HR model: {hr_model_path}")
        self.hr_model = YOLO(hr_model_path)

        self.row_skip       = row_skip
        self.col_skip       = col_skip
        self.step_r         = row_skip + 1
        self.step_c         = col_skip + 1
        self.padding_ratio  = padding_ratio
        self.min_crop_px    = min_crop_px
        self.hr_conf        = hr_conf
        self.nms_iou        = nms_iou
        self.lr_detect_conf = lr_detect_conf

        lr_frac   = 1.0 / (self.step_r * self.step_c)
        hr_extra  = 1.0 - lr_frac
        print(f"[AdaptiveCIS] LR pixel fraction : {lr_frac*100:.1f}%  "
              f"(row_skip={row_skip}, col_skip={col_skip})")
        print(f"[AdaptiveCIS] HR extra fraction : {hr_extra*100:.1f}%  "
              f"(pixels added per HR crop vs crop area)")
        print(f"[AdaptiveCIS] padding={padding_ratio*100:.0f}%  "
              f"nms_iou={nms_iou}  lr_detect_conf={lr_detect_conf}")

    # ─────────────────────────────────────────────────────────────────────────
    def _simulate_lr(self, img: np.ndarray) -> np.ndarray:
        """Sensor LR readout: skip rows/cols → upsample to original size."""
        H, W = img.shape[:2]
        lr_native = img[::self.step_r, ::self.step_c]
        return cv2.resize(lr_native, (W, H),
                          interpolation=cv2.INTER_NEAREST)

    def _run_model(self, model, img: np.ndarray, conf: float) -> list:
        """Run YOLO on img, return [[x1,y1,x2,y2,conf,cls], ...]."""
        H, W = img.shape[:2]
        result = model.predict(img, conf=conf, iou=self.nms_iou,
                               imgsz=640, verbose=False)[0]
        dets = []
        if result.boxes is not None and len(result.boxes) > 0:
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(float)
                x1 = max(0.0, min(float(W), x1))
                y1 = max(0.0, min(float(H), y1))
                x2 = max(0.0, min(float(W), x2))
                y2 = max(0.0, min(float(H), y2))
                dets.append([x1, y1, x2, y2,
                              float(b.conf[0].cpu()),
                              int(b.cls[0].cpu())])
        return dets

    def _pad_crop_box(self, x1, y1, x2, y2, W, H) -> tuple:
        """Expand bbox by padding_ratio, clamp to image bounds."""
        bw  = x2 - x1
        bh  = y2 - y1
        px  = bw * self.padding_ratio
        py  = bh * self.padding_ratio
        cx1 = max(0,   int(x1 - px))
        cy1 = max(0,   int(y1 - py))
        cx2 = min(W-1, int(x2 + px))
        cy2 = min(H-1, int(y2 + py))
        return cx1, cy1, cx2, cy2

    def _compute_energy(self, img_shape: tuple,
                         hr_crop_boxes: list) -> dict:
        """Energy = LR fraction + additional HR pixels (deduplicated)."""
        H, W      = img_shape[:2]
        total_px  = H * W
        lr_pixels = (H // self.step_r) * (W // self.step_c)
        lr_frac   = 1.0 / (self.step_r * self.step_c)
        hr_extra_frac = 1.0 - lr_frac  # 0.75 for 2× skip

        # Build a pixel mask to DEDUPLICATE overlapping HR crops
        if hr_crop_boxes:
            hr_mask = np.zeros((H, W), dtype=np.uint8)
            for (cx1, cy1, cx2, cy2) in hr_crop_boxes:
                hr_mask[cy1:cy2, cx1:cx2] = 1
            hr_area     = int(hr_mask.sum())
            hr_pixels   = int(hr_area * hr_extra_frac)
        else:
            hr_pixels   = 0
            hr_area     = 0

        total_read   = lr_pixels + hr_pixels
        energy_ratio = min(total_read / total_px, 1.0)

        return {
            'energy_ratio': round(energy_ratio,                  4),
            'total_pct':    round(min(100.0*total_read/total_px, 100.0), 2),
            'lr_pct':       round(100.0 * lr_pixels  / total_px, 2),
            'hr_pct':       round(100.0 * hr_pixels  / total_px, 2),
            'hr_area_pct':  round(100.0 * hr_area    / total_px, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def process_frame(self, img: np.ndarray,
                      hr_trigger_conf: float = None) -> tuple:
        """
        Adaptive LR+HR pipeline — COMPOSITE IMAGE APPROACH.

        Sensor physics simulation:
          1. Read full frame at LR (25% pixels)    → lr_img
          2. Run LR model on lr_img                → lr_dets (screening)
          3. For dets with conf ≥ τ:
               - Mark those pixel regions for HR readout
               - Paste original HR pixels into those regions of lr_img
               → composite_img  (HR in ROIs, LR elsewhere)
          4. Run HR model on FULL composite_img    → final_dets

        Why this is correct:
          - HR model always sees full 640×640 scene context (same as training)
          - ROI pixels are true HR quality → better localization
          - Background is LR but provides scene context → model not confused
          - Low τ: most pixels are HR → composite ≈ full HR → mAP ≈ HR ✓
          - High τ: few HR pixels → composite ≈ full LR → mAP ≈ LR ✓
          - Curve is now MONOTONICALLY DECREASING as τ increases ✓
        """
        tau = hr_trigger_conf if hr_trigger_conf is not None else self.hr_conf
        H, W = img.shape[:2]

        # ── Step 1: LR readout simulation ────────────────────────────────────
        lr_img = self._simulate_lr(img)

        # ── Step 2: LR screening detection ───────────────────────────────────
        lr_dets = self._run_model(self.lr_model, lr_img,
                                  conf=self.lr_detect_conf)

        # ── Step 3: Identify triggered regions ───────────────────────────────
        hr_crop_boxes  = []
        triggered_idxs = []

        for i, det in enumerate(lr_dets):
            x1, y1, x2, y2, conf, cls = det
            if conf < tau:
                continue
            cx1, cy1, cx2, cy2 = self._pad_crop_box(x1, y1, x2, y2, W, H)
            if (cx2 - cx1) < self.min_crop_px or (cy2 - cy1) < self.min_crop_px:
                continue
            hr_crop_boxes.append((cx1, cy1, cx2, cy2))
            triggered_idxs.append(i)

        # ── Step 4: Build composite image ────────────────────────────────────
        #
        # THE FUNDAMENTAL FIX:
        #   Start with the blurry LR image (full size).
        #   Paste the ORIGINAL HR pixels for every triggered ROI.
        #   → HR model sees full scene + sharp pixels where it matters.
        #
        if hr_crop_boxes:
            composite = lr_img.copy()
            for (cx1, cy1, cx2, cy2) in hr_crop_boxes:
                composite[cy1:cy2, cx1:cx2] = img[cy1:cy2, cx1:cx2]
        else:
            composite = lr_img  # no triggers → pure LR image

        # ── Step 5: Run HR model on full composite image ──────────────────────
        if hr_crop_boxes:
            final_dets = self._run_model(self.hr_model, composite,
                                         conf=self.lr_detect_conf)  ## instead of self.hr_model
        else:
            # Nothing triggered → use LR detections directly
            final_dets = lr_dets

        # ── Step 6: Energy accounting ─────────────────────────────────────────
        energy_info = self._compute_energy(img.shape, hr_crop_boxes)

        # ── Debug info for visualization ──────────────────────────────────────
        debug = {
            'lr_img':        lr_img,
            'lr_dets':       lr_dets,
            'triggered_idx': triggered_idxs,
            'hr_crops':      hr_crop_boxes,
            'hr_dets':       final_dets,
            'composite':     composite if hr_crop_boxes else lr_img,
            'n_triggered':   len(triggered_idxs),
        }

        return final_dets, energy_info, debug
