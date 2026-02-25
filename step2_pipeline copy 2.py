"""
step2_pipeline.py  —  Adaptive CIS LR+HR Pipeline  (CORRECTED v3)

FIXES vs v2:
  [FIX 6] Energy model corrected — HR crops only charged for ADDITIONAL pixels
           not already read in the LR pass.

           Old (wrong):
             hr_pixels = sum(crop_w × crop_h)
             → double-counts LR pixels inside HR crops
             → can exceed 100% total energy (physically impossible)

           New (correct):
             lr_fraction = 1 / (step_r × step_c)            # fraction already read
             hr_additional = 1 - lr_fraction                 # fraction still to read
             hr_pixels = sum(crop_area) × hr_additional
             → for both_skip×2: additional = 75% of crop area
             → total energy always in [25%, 100%]

           Physical meaning:
             LR pass:  full frame, every step_r-th row and step_c-th col → 25%
             HR pass:  for activated crops, fill in the missing (step_r×step_c - 1)
                       pixels per grid cell → 75% of crop area

Prior fixes from v2 (unchanged):
  [FIX 1] Dual-threshold: lr_detect_conf (keep) vs hr_trigger_conf (activate HR)
  [FIX 2] padding_ratio 0.15 → 0.07
  [FIX 3] NMS iou_threshold 0.50 → 0.40
  [FIX 4] HR box clamp before full-frame mapping
  [FIX 5] hr_trigger_conf as sweep parameter in process_frame()
"""

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# NMS helper
# ─────────────────────────────────────────────────────────────────────────────

def _iou_xyxy(b1: list, b2: list) -> float:
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(0.0, b1[2]-b1[0]) * max(0.0, b1[3]-b1[1])
    a2 = max(0.0, b2[2]-b2[0]) * max(0.0, b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 1e-6 else 0.0


def apply_nms(detections: list, iou_threshold: float = 0.40) -> list:
    """
    Per-class greedy NMS over list of [x1, y1, x2, y2, conf, cls].
    iou_threshold=0.40 lets closely-spaced pedestrians coexist.
    """
    if len(detections) <= 1:
        return detections

    kept = []
    for cls in set(int(d[5]) for d in detections):
        cls_dets = sorted(
            [d for d in detections if int(d[5]) == cls],
            key=lambda x: x[4], reverse=True
        )
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            cls_dets = [
                d for d in cls_dets
                if _iou_xyxy(best[:4], d[:4]) < iou_threshold
            ]
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Core Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveCISPipeline:
    """
    Adaptive CMOS Image Sensor pipeline.

    Every frame:
      1. Full-frame LR readout  → 25% of pixels (both_skip×2)
      2. LR YOLO detects above lr_detect_conf
      3. Above hr_trigger_conf  → sensor reads ADDITIONAL HR pixels for that ROI
         Below hr_trigger_conf  → kept as LR detection (not dropped)
      4. HR YOLO on each activated crop
      5. Map HR boxes back to full-frame coords
      6. NMS fusion
      7. Energy = LR frame + additional HR pixels only (no double-counting)
    """

    def __init__(self,
                 lr_model_path:   str,
                 hr_model_path:   str,
                 row_skip:        int   = 1,
                 col_skip:        int   = 1,
                 lr_detect_conf:  float = 0.15,
                 hr_conf:         float = 0.25,
                 padding_ratio:   float = 0.07,
                 min_crop_px:     int   = 24,
                 nms_iou:         float = 0.40):

        print(f"[AdaptiveCIS] Loading LR model: {lr_model_path}")
        self.lr_model = YOLO(lr_model_path)

        print(f"[AdaptiveCIS] Loading HR model: {hr_model_path}")
        self.hr_model = YOLO(hr_model_path)

        self.row_skip       = row_skip
        self.col_skip       = col_skip
        self.step_r         = row_skip + 1
        self.step_c         = col_skip + 1
        self.lr_detect_conf = lr_detect_conf
        self.hr_conf        = hr_conf
        self.padding_ratio  = padding_ratio
        self.min_crop_px    = min_crop_px
        self.nms_iou        = nms_iou

        # [FIX 6] Pre-compute energy fractions (used in every frame)
        self.lr_pixel_fraction  = 1.0 / (self.step_r * self.step_c)  # 0.25 for ×2
        self.hr_extra_fraction  = 1.0 - self.lr_pixel_fraction        # 0.75 for ×2

        print(f"[AdaptiveCIS] LR pixel fraction : {self.lr_pixel_fraction*100:.1f}%  "
              f"(row_skip={row_skip}, col_skip={col_skip})")
        print(f"[AdaptiveCIS] HR extra fraction : {self.hr_extra_fraction*100:.1f}%  "
              f"(pixels added per HR crop vs crop area)")
        print(f"[AdaptiveCIS] padding={padding_ratio*100:.0f}%  "
              f"nms_iou={nms_iou}  lr_detect_conf={lr_detect_conf}")

    # ── Sensor simulation ──────────────────────────────────────────────────────

    def simulate_lr_readout(self, hr_img: np.ndarray) -> np.ndarray:
        """
        Simulate sensor LR readout.
        Sub-samples every step_r-th row and step_c-th col,
        then NN-upsamples back to original (W, H).
        Matches exactly how both_skip1_640 training images were generated —
        so LR YOLO boxes are already in full-frame pixel coordinates.
        """
        H, W      = hr_img.shape[:2]
        lr_native = hr_img[::self.step_r, ::self.step_c]
        lr_up     = cv2.resize(lr_native, (W, H), interpolation=cv2.INTER_NEAREST)
        return lr_up

    # ── Energy model (CORRECTED v3) ────────────────────────────────────────────

    def compute_energy(self,
                       hr_img:        np.ndarray,
                       hr_crop_boxes: list) -> dict:
        """
        Physically correct per-frame sensor pixel budget.

        LR pass  : full frame at skip rate       → lr_pixel_fraction × H × W
        HR pass  : for each activated ROI, only  → hr_extra_fraction × crop_area
                   the pixels NOT already read in the LR pass are newly read.

        Example (both_skip×2, one 100×100 HR crop):
          LR pixels  = 640×640 × 0.25 = 102,400
          HR extra   = 100×100 × 0.75 =   7,500  (not 10,000!)
          Total      = 109,900 / 409,600 = 26.83%
          Energy saved = 73.17%

        Without HR crops: energy = 25% exactly (always-LR bound)
        With 100% HR coverage: energy = 25% + 75% = 100% (always-HR bound) ✓
        Old model with 100% HR: 25% + 100% = 125% ✗  (physically impossible)
        """
        H, W         = hr_img.shape[:2]
        total_pixels = H * W

        # LR: always the full-frame sub-sampled pass
        lr_pixels = (H // self.step_r) * (W // self.step_c)

        # HR: only the ADDITIONAL pixels not yet read by the LR pass
        # = crop_area × (1 - 1/(step_r × step_c))
        hr_pixels = int(sum(
            (x2 - x1) * (y2 - y1) * self.hr_extra_fraction
            for (x1, y1, x2, y2) in hr_crop_boxes
        ))

        total_read   = lr_pixels + hr_pixels
        energy_ratio = min(total_read / total_pixels, 1.0)  # hard cap at 1.0

        return {
            'total_pixels':  total_pixels,
            'lr_pixels':     lr_pixels,
            'hr_pixels':     hr_pixels,      # additional HR pixels only
            'total_read':    int(total_read),
            'energy_ratio':  round(energy_ratio, 4),
            'energy_saved':  round(1.0 - energy_ratio, 4),
            'lr_pct':        round(100.0 * lr_pixels  / total_pixels, 2),
            'hr_pct':        round(100.0 * hr_pixels  / total_pixels, 2),
            'total_pct':     round(100.0 * total_read / total_pixels, 2),
        }

    # ── Crop helpers ───────────────────────────────────────────────────────────

    def _padded_crop_coords(self,
                            x1: float, y1: float,
                            x2: float, y2: float,
                            W: int,    H: int) -> tuple:
        """Padded HR crop coords clamped to image bounds."""
        pw  = (x2 - x1) * self.padding_ratio
        ph  = (y2 - y1) * self.padding_ratio
        cx1 = max(0, int(x1 - pw))
        cy1 = max(0, int(y1 - ph))
        cx2 = min(W, int(x2 + pw))
        cy2 = min(H, int(y2 + ph))
        return cx1, cy1, cx2, cy2

    def _map_crop_to_fullframe(self,
                               hr_boxes_raw,
                               cx1: int, cy1: int,
                               cx2: int, cy2: int) -> list:
        """
        Map HR YOLO detections from crop-local coords → full-frame coords.
        [FIX 4] Clamp to crop bounds before offsetting.
        """
        full_dets = []
        crop_w    = cx2 - cx1
        crop_h    = cy2 - cy1

        for hb in hr_boxes_raw:
            hx1, hy1, hx2, hy2 = hb.xyxy[0].cpu().numpy().astype(float)

            # Clamp to crop bounds (avoids out-of-bound boxes near edges)
            hx1 = max(0.0, min(hx1, float(crop_w)))
            hy1 = max(0.0, min(hy1, float(crop_h)))
            hx2 = max(0.0, min(hx2, float(crop_w)))
            hy2 = max(0.0, min(hy2, float(crop_h)))

            hconf = float(hb.conf[0].cpu())
            hcls  = int(hb.cls[0].cpu())

            full_dets.append([
                cx1 + hx1, cy1 + hy1,
                cx1 + hx2, cy1 + hy2,
                hconf, hcls
            ])
        return full_dets

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def process_frame(self,
                      hr_img:          np.ndarray,
                      hr_trigger_conf: float = 0.35) -> tuple:
        """
        Run the full adaptive pipeline on one frame.

        Args:
            hr_img:          Full-resolution BGR image (simulates sensor HR buffer).
            hr_trigger_conf: HR activation gate (sweep this in evaluate.py).
                             ≥ hr_trigger_conf → HR crop activated
                             in [lr_detect_conf, hr_trigger_conf) → kept as LR det
                             < lr_detect_conf → discarded (true background)

        Returns:
            detections:  [[x1,y1,x2,y2,conf,cls], ...]  full-frame pixel coords
            energy_info: dict from compute_energy()
            debug:       {'lr_img', 'lr_dets', 'hr_crops'}
        """
        H, W = hr_img.shape[:2]

        # ── Step 1: LR sensor readout ──────────────────────────────────────────
        lr_img = self.simulate_lr_readout(hr_img)

        # ── Step 2: LR detection ───────────────────────────────────────────────
        lr_result    = self.lr_model.predict(
            lr_img, conf=self.lr_detect_conf, iou=0.50, verbose=False
        )[0]
        lr_boxes_raw = lr_result.boxes

        final_dets    = []
        hr_crop_boxes = []
        debug_lr_dets = []

        if lr_boxes_raw is None or len(lr_boxes_raw) == 0:
            energy_info = self.compute_energy(hr_img, [])
            return final_dets, energy_info, {
                'lr_img': lr_img, 'lr_dets': [], 'hr_crops': []
            }

        # ── Steps 3–5: Decide LR-keep or HR-activate ──────────────────────────
        for box in lr_boxes_raw:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            conf = float(box.conf[0].cpu())
            cls  = int(box.cls[0].cpu())

            debug_lr_dets.append([x1, y1, x2, y2, conf, cls])

            if conf < hr_trigger_conf:
                # Below HR gate → keep LR detection as-is (never drop)
                final_dets.append([x1, y1, x2, y2, conf, cls])
                continue

            # Above HR gate → activate HR for this ROI
            cx1, cy1, cx2, cy2 = self._padded_crop_coords(x1, y1, x2, y2, W, H)

            if (cx2 - cx1) < self.min_crop_px or (cy2 - cy1) < self.min_crop_px:
                # Crop too tiny → fall back to LR detection
                final_dets.append([x1, y1, x2, y2, conf, cls])
                continue

            hr_crop_boxes.append((cx1, cy1, cx2, cy2))

            # Step 4: Read real HR pixels from sensor (no upsampling here)
            hr_crop = hr_img[cy1:cy2, cx1:cx2]

            # Step 5: HR YOLO on crop
            hr_result    = self.hr_model.predict(
                hr_crop, conf=self.hr_conf, iou=0.45, verbose=False
            )[0]
            hr_boxes_raw = hr_result.boxes

            if hr_boxes_raw is not None and len(hr_boxes_raw) > 0:
                refined = self._map_crop_to_fullframe(
                    hr_boxes_raw, cx1, cy1, cx2, cy2
                )
                final_dets.extend(refined)
            else:
                # HR found nothing → fall back to LR box
                final_dets.append([x1, y1, x2, y2, conf, cls])

        # ── Step 6: NMS ───────────────────────────────────────────────────────
        final_dets = apply_nms(final_dets, iou_threshold=self.nms_iou)

        # ── Step 7: Energy (CORRECTED v3) ─────────────────────────────────────
        energy_info = self.compute_energy(hr_img, hr_crop_boxes)

        debug = {
            'lr_img':   lr_img,
            'lr_dets':  debug_lr_dets,
            'hr_crops': hr_crop_boxes,
        }

        return final_dets, energy_info, debug
