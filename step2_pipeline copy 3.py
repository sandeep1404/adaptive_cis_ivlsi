"""
step2_pipeline.py — Fixed v2
============================
KEY FIX: HR detections REPLACE triggered LR detections.
Previously: final = NMS(ALL_LR + HR)  ← both kept → duplicate FPs
Now:        final = NMS(NON_TRIGGERED_LR + HR)  ← HR replaces LR ✓
"""

import cv2
import numpy as np
from pathlib import Path


class AdaptiveCISPipeline:

    def __init__(self,
                 lr_model_path:  str,
                 hr_model_path:  str,
                 row_skip:       int   = 1,
                 col_skip:       int   = 1,
                 padding_ratio:  float = 0.07,
                 min_crop_px:    int   = 32,
                 hr_conf:        float = 0.50,
                 nms_iou:        float = 0.40,
                 lr_detect_conf: float = 0.15):

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

        lr_frac = 1.0 / (self.step_r * self.step_c)
        hr_extra = 1.0 - lr_frac
        print(f"[AdaptiveCIS] LR pixel fraction : {lr_frac*100:.1f}%  "
              f"(row_skip={row_skip}, col_skip={col_skip})")
        print(f"[AdaptiveCIS] HR extra fraction : {hr_extra*100:.1f}%  "
              f"(pixels added per HR crop vs crop area)")
        print(f"[AdaptiveCIS] padding={padding_ratio*100:.0f}%  "
              f"nms_iou={nms_iou}  lr_detect_conf={lr_detect_conf}")

    # ─────────────────────────────────────────────────────────────────────────
    def _simulate_lr(self, img: np.ndarray) -> np.ndarray:
        """Downsample by skipping rows/cols, upsample back to original size."""
        H, W = img.shape[:2]
        lr_native = img[::self.step_r, ::self.step_c]
        return cv2.resize(lr_native, (W, H), interpolation=cv2.INTER_NEAREST)

    def _run_model(self, model, img: np.ndarray, conf: float) -> list:
        """Run YOLO model, return list of [x1,y1,x2,y2,conf,cls]."""
        H, W = img.shape[:2]
        result = model.predict(img, conf=conf, iou=self.nms_iou,
                               imgsz=640, verbose=False)[0]
        dets = []
        if result.boxes is not None and len(result.boxes) > 0:
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(float)
                # clamp to image bounds
                x1 = max(0.0, min(float(W), x1))
                y1 = max(0.0, min(float(H), y1))
                x2 = max(0.0, min(float(W), x2))
                y2 = max(0.0, min(float(H), y2))
                dets.append([x1, y1, x2, y2,
                              float(b.conf[0].cpu()),
                              int(b.cls[0].cpu())])
        return dets

    def _pad_crop_box(self, x1, y1, x2, y2, W, H) -> tuple:
        """Add padding around a bounding box, clamp to image bounds."""
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * self.padding_ratio
        pad_y = bh * self.padding_ratio
        cx1 = max(0,   int(x1 - pad_x))
        cy1 = max(0,   int(y1 - pad_y))
        cx2 = min(W-1, int(x2 + pad_x))
        cy2 = min(H-1, int(y2 + pad_y))
        return cx1, cy1, cx2, cy2

    def _map_dets_to_full_image(self, dets: list,
                                 crop_x1: int, crop_y1: int) -> list:
        """Map crop-relative detection boxes back to full-image coordinates."""
        mapped = []
        for d in dets:
            x1, y1, x2, y2, conf, cls = d
            mapped.append([
                x1 + crop_x1,
                y1 + crop_y1,
                x2 + crop_x1,
                y2 + crop_y1,
                conf, cls
            ])
        return mapped

    def _nms(self, dets: list, iou_thresh: float = None) -> list:
        """Simple class-aware NMS."""
        if not dets:
            return []
        iou_thresh = iou_thresh or self.nms_iou
        dets_sorted = sorted(dets, key=lambda x: x[4], reverse=True)
        kept = []
        suppressed = [False] * len(dets_sorted)
        for i in range(len(dets_sorted)):
            if suppressed[i]:
                continue
            kept.append(dets_sorted[i])
            for j in range(i + 1, len(dets_sorted)):
                if suppressed[j]:
                    continue
                if dets_sorted[i][5] != dets_sorted[j][5]:
                    continue  # different class, skip
                iou = self._compute_iou(dets_sorted[i][:4],
                                        dets_sorted[j][:4])
                if iou > iou_thresh:
                    suppressed[j] = True
        return kept

    @staticmethod
    def _compute_iou(b1, b2) -> float:
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 1e-6 else 0.0

    def _compute_energy(self, img_shape: tuple,
                         hr_crop_boxes: list) -> dict:
        """Compute pixel read fraction for this frame."""
        H, W      = img_shape[:2]
        total_px  = H * W
        lr_pixels = (H // self.step_r) * (W // self.step_c)
        lr_frac   = 1.0 / (self.step_r * self.step_c)
        hr_extra_frac = 1.0 - lr_frac  # HR reads (1 - 0.25) = 75% more pixels

        hr_pixels = 0
        for (cx1, cy1, cx2, cy2) in hr_crop_boxes:
            crop_area = (cx2 - cx1) * (cy2 - cy1)
            hr_pixels += int(crop_area * hr_extra_frac)

        total_read   = lr_pixels + hr_pixels
        energy_ratio = min(total_read / total_px, 1.0)

        return {
            'energy_ratio': round(energy_ratio, 4),
            'total_pct':    round(100.0 * total_read / total_px, 2),
            'lr_pct':       round(100.0 * lr_pixels / total_px,  2),
            'hr_pct':       round(100.0 * hr_pixels / total_px,  2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def process_frame(self, img: np.ndarray,
                      hr_trigger_conf: float = None) -> tuple:
        """
        Process one frame through the adaptive LR+HR pipeline.

        THE FIX IS HERE:
          OLD: final = NMS(ALL_lr_dets + hr_dets)
               → triggered LR dets kept AND HR dets added → duplicates + FPs
          NEW: final = NMS(NON_TRIGGERED_lr_dets + hr_dets)
               → triggered LR dets REMOVED, replaced by HR dets
               → HR dets refine/replace, not pollute

        Returns:
            final_dets  : list of [x1,y1,x2,y2,conf,cls]
            energy_info : dict with energy metrics
            debug       : dict with intermediate results for visualization
        """
        tau = hr_trigger_conf if hr_trigger_conf is not None else self.hr_conf
        H, W = img.shape[:2]

        # ── Step 1: Simulate LR readout ───────────────────────────────────────
        lr_img = self._simulate_lr(img)

        # ── Step 2: Run LR model ──────────────────────────────────────────────
        lr_dets = self._run_model(self.lr_model, lr_img,
                                  conf=self.lr_detect_conf)

        # ── Step 3: Decide which LR dets trigger HR ───────────────────────────
        triggered_indices = []   # indices of LR dets that trigger HR
        kept_lr_dets      = []   # LR dets that do NOT trigger HR (kept as-is)

        for i, det in enumerate(lr_dets):
            x1, y1, x2, y2, conf, cls = det
            crop_w = x2 - x1
            crop_h = y2 - y1
            # Trigger if: confidence ≥ tau AND crop is large enough to matter
            if (conf >= tau and
                    crop_w >= self.min_crop_px and
                    crop_h >= self.min_crop_px):
                triggered_indices.append(i)
            else:
                kept_lr_dets.append(det)  # ← NOT triggered: keep LR det

        # ── Step 4: Run HR model on triggered crops ───────────────────────────
        hr_dets_all  = []    # all HR detections mapped to full-image coords
        hr_crop_boxes = []   # crop boxes (for energy calculation + visualization)

        for i in triggered_indices:
            x1, y1, x2, y2, conf, cls = lr_dets[i]
            cx1, cy1, cx2, cy2 = self._pad_crop_box(x1, y1, x2, y2, W, H)

            # Skip crops that are too small after padding
            if (cx2 - cx1) < self.min_crop_px or (cy2 - cy1) < self.min_crop_px:
                kept_lr_dets.append(lr_dets[i])  # fallback: keep LR det
                continue

            # Crop from the ORIGINAL full-resolution image
            hr_crop = img[cy1:cy2, cx1:cx2].copy()
            if hr_crop.size == 0:
                kept_lr_dets.append(lr_dets[i])  # fallback: keep LR det
                continue

            # Run HR model on the crop
            # Use same lr_detect_conf so we don't miss objects the LR found
            crop_dets = self._run_model(self.hr_model, hr_crop,
                                        conf=self.lr_detect_conf)

            # Map crop coordinates back to full image
            mapped = self._map_dets_to_full_image(crop_dets, cx1, cy1)

            if mapped:
                # HR found something → add HR dets (they replace this LR det)
                hr_dets_all.extend(mapped)
            else:
                # HR found nothing in the crop → FALLBACK: keep original LR det
                # This prevents losing objects when HR model misses them on crops
                kept_lr_dets.append(lr_dets[i])

            hr_crop_boxes.append((cx1, cy1, cx2, cy2))

        # ── Step 5: Merge  ────────────────────────────────────────────────────
        #
        # THE KEY FIX:
        #   kept_lr_dets  = LR dets that did NOT trigger HR (unchanged)
        #                 + LR dets where HR found NOTHING (fallback)
        #   hr_dets_all   = HR dets from all triggered crops (replacements)
        #
        #   We do NOT include triggered LR dets that had successful HR matches.
        #   HR dets REPLACE triggered LR dets.
        #
        all_dets_before_nms = kept_lr_dets + hr_dets_all
        final_dets = self._nms(all_dets_before_nms, iou_thresh=self.nms_iou)

        # ── Step 6: Energy ────────────────────────────────────────────────────
        energy_info = self._compute_energy(img.shape, hr_crop_boxes)

        # ── Debug info for visualization ──────────────────────────────────────
        debug = {
            'lr_img':        lr_img,
            'lr_dets':       lr_dets,           # all LR detections (before split)
            'triggered_idx': triggered_indices,  # which LR dets triggered HR
            'hr_crops':      hr_crop_boxes,      # crop boxes used for HR
            'hr_dets':       hr_dets_all,        # raw HR detections
            'kept_lr_dets':  kept_lr_dets,       # LR dets that weren't replaced
            'n_triggered':   len(triggered_indices),
            'n_hr_found':    len(hr_dets_all),
            'n_hr_fallback': len(triggered_indices) - len(hr_crop_boxes),
        }

        return final_dets, energy_info, debug
