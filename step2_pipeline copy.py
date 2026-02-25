"""
step2_pipeline.py  —  Adaptive CIS LR+HR Pipeline  (CORRECTED v2)

FIXES vs v1:
  [FIX 1] Dual-threshold logic:
            - lr_detect_conf (low, e.g. 0.15): minimum confidence to keep ANY
              detection in final output. Objects below this are truly background.
            - hr_trigger_conf (tunable sweep param): minimum confidence to
              ACTIVATE HR mode for that region. Objects below hr_trigger_conf
              but above lr_detect_conf are KEPT as LR detections (never lost).
            Previously, objects with conf < lr_conf were silently dropped,
            causing mAP to fall as the threshold rose.

  [FIX 2] padding_ratio reduced 0.15 → 0.07
            Smaller padding → tighter HR crops → boxes match GT better →
            higher IoU → better mAP@0.5:0.95.

  [FIX 3] Final NMS iou_threshold 0.5 → 0.40
            Allows closely-spaced people to survive NMS without merging into
            one large box (was swallowing adjacent pedestrians in Penn-Fudan).

  [FIX 4] HR crop result remapping — clamp HR boxes to valid range before
            mapping back to full-frame coords (avoids rare out-of-bound boxes
            from YOLO on tiny crops near image edges).

  [FIX 5] process_frame() now accepts hr_trigger_conf as the sweep parameter
            (not lr_conf). This separates detection sensitivity from HR activation.

Energy Model (unchanged, correct):
  E_lr_frame  = (H // step_r) × (W // step_c)         [always paid, full frame]
  E_hr_crops  = Σ (roi_w × roi_h)                      [paid only for HR regions]
  energy_ratio = (E_lr_frame + E_hr_crops) / (H × W)
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# NMS helper
# ─────────────────────────────────────────────────────────────────────────────

def _iou_xyxy(b1: list, b2: list) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1    = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2    = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 1e-6 else 0.0


def apply_nms(detections: list, iou_threshold: float = 0.40) -> list:
    """
    Per-class greedy NMS over list of [x1, y1, x2, y2, conf, cls].

    iou_threshold = 0.40 (was 0.50):
      Lower threshold lets nearby pedestrians coexist without being merged
      into one large box when their bounding boxes partially overlap.
    """
    if len(detections) <= 1:
        return detections

    classes = set(int(d[5]) for d in detections)
    kept = []

    for cls in classes:
        cls_dets = sorted(
            [d for d in detections if int(d[5]) == cls],
            key=lambda x: x[4],
            reverse=True       # highest confidence first
        )
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            # suppress boxes that overlap too much with 'best'
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
      1. Full-frame LR readout (line-skip, both_skip×2 → 25% pixels)
      2. LR YOLO detects all objects above lr_detect_conf
      3. For objects above hr_trigger_conf → sensor activates HR for that ROI
         For objects below hr_trigger_conf → kept as LR detections (not dropped)
      4. HR YOLO on each activated crop → precise boxes
      5. Map HR boxes back to full-frame coords
      6. NMS fusion
      7. Energy accounting
    """

    def __init__(self,
                 lr_model_path:   str,
                 hr_model_path:   str,
                 row_skip:        int   = 1,
                 col_skip:        int   = 1,
                 lr_detect_conf:  float = 0.15,   # [FIX 1] always keep above this
                 hr_conf:         float = 0.25,   # confidence for HR YOLO on crops
                 padding_ratio:   float = 0.07,   # [FIX 2] was 0.15
                 min_crop_px:     int   = 24,
                 nms_iou:         float = 0.40):  # [FIX 3] was 0.50
        """
        Args:
            lr_model_path:  best.pt from Step 1 both_skip1_640 training
            hr_model_path:  best.pt from Step 1 hr_640 training
            row_skip:       rows to skip between sensor reads (1 = every 2nd row)
            col_skip:       cols to skip between sensor reads (1 = every 2nd col)
            lr_detect_conf: minimum LR confidence to keep a detection at all.
                            Set low (0.15) so no real object is ever silently dropped.
            hr_conf:        confidence threshold inside the HR YOLO pass on crops.
            padding_ratio:  fraction of box size added as padding when cropping HR ROI.
                            0.07 = 7% on each side — enough to avoid truncation without
                            making the box oversized.
            min_crop_px:    skip HR refinement for crops smaller than this (pixels).
            nms_iou:        IoU threshold for final NMS fusion (0.40 allows adjacent
                            pedestrians to coexist).
        """
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

        lr_pct = 100.0 / (self.step_r * self.step_c)
        print(f"[AdaptiveCIS] LR pixel budget (full frame): {lr_pct:.1f}%  "
              f"(row_skip={row_skip}, col_skip={col_skip})")
        print(f"[AdaptiveCIS] padding={padding_ratio*100:.0f}%  "
              f"nms_iou={nms_iou}  lr_detect_conf={lr_detect_conf}")

    # ── Sensor simulation ──────────────────────────────────────────────────────

    def simulate_lr_readout(self, hr_img: np.ndarray) -> np.ndarray:
        """
        Simulate sensor LR readout.

        Sub-samples every (step_r)-th row and (step_c)-th col,
        then nearest-neighbour upsamples back to original (W, H).

        This matches exactly how both_skip1_640 training images were created,
        so coordinates from the LR YOLO model are directly in (W, H) pixel space —
        NO coordinate scaling needed to map LR boxes onto the HR image.
        """
        H, W    = hr_img.shape[:2]
        lr_native = hr_img[::self.step_r, ::self.step_c]
        lr_up     = cv2.resize(lr_native, (W, H), interpolation=cv2.INTER_NEAREST)
        return lr_up

    # ── Energy model ───────────────────────────────────────────────────────────

    def compute_energy(self,
                       hr_img:        np.ndarray,
                       hr_crop_boxes: list) -> dict:
        """
        Per-frame sensor pixel budget.

        LR cost is always 25% (both_skip×2, full frame).
        HR cost is the sum of activated ROI areas.
        """
        H, W         = hr_img.shape[:2]
        total_pixels = H * W

        lr_pixels = (H // self.step_r) * (W // self.step_c)
        hr_pixels = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in hr_crop_boxes)

        total_read   = lr_pixels + hr_pixels
        energy_ratio = min(total_read / total_pixels, 1.0)

        return {
            'total_pixels':  total_pixels,
            'lr_pixels':     lr_pixels,
            'hr_pixels':     hr_pixels,
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
                            W:  int,   H:  int) -> tuple:
        """
        Compute padded HR crop coordinates, clamped to image bounds.
        padding_ratio = 0.07 adds 7% of box width/height on each side.
        """
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
        Map HR YOLO detections from crop-local coordinates → full-frame coordinates.

        [FIX 4] Added clamp to ensure mapped boxes stay within crop boundaries
        before offsetting, preventing rare out-of-bound boxes near image edges.
        """
        full_dets = []
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        for hb in hr_boxes_raw:
            hx1, hy1, hx2, hy2 = hb.xyxy[0].cpu().numpy().astype(float)

            # [FIX 4] Clamp to crop bounds before offsetting
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
        Run the full adaptive pipeline on a single frame.

        Args:
            hr_img:          Original full-resolution BGR image.
            hr_trigger_conf: [FIX 1] Confidence threshold that GATES HR activation.
                             This is the parameter you sweep in evaluate.py.

                             Objects with LR conf >= hr_trigger_conf → HR crop activated
                             Objects with LR conf in [lr_detect_conf, hr_trigger_conf)
                               → kept in final output as LR detections (NOT dropped)
                             Objects with LR conf < lr_detect_conf → true background, ignored

                             Sweeping hr_trigger_conf gives the energy-accuracy curve:
                               low hr_trigger_conf  → many HR crops → higher accuracy, less saving
                               high hr_trigger_conf → few HR crops  → lower accuracy, more saving

        Returns:
            detections:  list of [x1, y1, x2, y2, conf, cls] (full-frame pixel coords)
            energy_info: dict from compute_energy()
            debug:       dict with LR image, LR detections, HR crop boxes (for visualization)
        """
        H, W = hr_img.shape[:2]

        # ── Step 1: LR sensor readout ──────────────────────────────────────────
        lr_img = self.simulate_lr_readout(hr_img)

        # ── Step 2: LR detection — use low lr_detect_conf to catch all objects ─
        # [FIX 1] lr_detect_conf is LOW (0.15); we never lose an object here.
        # hr_trigger_conf controls which of these we upgrade to HR — separately.
        lr_result    = self.lr_model.predict(
            lr_img,
            conf    = self.lr_detect_conf,   # catch everything
            iou     = 0.50,
            verbose = False
        )[0]
        lr_boxes_raw = lr_result.boxes

        final_dets    = []
        hr_crop_boxes = []
        debug_lr_dets = []

        # No detections at all → pure LR frame, no HR activation
        if lr_boxes_raw is None or len(lr_boxes_raw) == 0:
            energy_info = self.compute_energy(hr_img, [])
            return final_dets, energy_info, {
                'lr_img': lr_img, 'lr_dets': [], 'hr_crops': []
            }

        # ── Steps 3–5: For each LR detection decide LR-keep or HR-activate ────
        for box in lr_boxes_raw:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            conf = float(box.conf[0].cpu())
            cls  = int(box.cls[0].cpu())

            debug_lr_dets.append([x1, y1, x2, y2, conf, cls])

            # ── [FIX 1] GATE: should this detection trigger HR? ────────────────
            if conf < hr_trigger_conf:
                # Not confident enough for HR → keep LR box as final detection.
                # Object is NOT lost — it just doesn't get HR refinement.
                final_dets.append([x1, y1, x2, y2, conf, cls])
                continue

            # ── HR mode: compute padded crop coordinates ───────────────────────
            cx1, cy1, cx2, cy2 = self._padded_crop_coords(x1, y1, x2, y2, W, H)

            if (cx2 - cx1) < self.min_crop_px or (cy2 - cy1) < self.min_crop_px:
                # Crop too small for HR model → keep LR detection
                final_dets.append([x1, y1, x2, y2, conf, cls])
                continue

            hr_crop_boxes.append((cx1, cy1, cx2, cy2))

            # ── Step 4: Extract real HR pixels from original frame ─────────────
            # This is the sensor switching to full-pixel readout for this ROI.
            hr_crop = hr_img[cy1:cy2, cx1:cx2]

            # ── Step 5: HR YOLO on crop ────────────────────────────────────────
            hr_result    = self.hr_model.predict(
                hr_crop,
                conf    = self.hr_conf,
                iou     = 0.45,
                verbose = False
            )[0]
            hr_boxes_raw = hr_result.boxes

            if hr_boxes_raw is not None and len(hr_boxes_raw) > 0:
                # Map HR crop detections → full-frame coordinates [FIX 4]
                refined = self._map_crop_to_fullframe(
                    hr_boxes_raw, cx1, cy1, cx2, cy2
                )
                final_dets.extend(refined)
            else:
                # HR model found nothing in crop → fall back to LR detection
                # (object is still reported, just at LR precision)
                final_dets.append([x1, y1, x2, y2, conf, cls])

        # ── Step 6: NMS over fused detections [FIX 3] ─────────────────────────
        # iou_threshold=0.40 allows closely-spaced pedestrians to coexist
        final_dets = apply_nms(final_dets, iou_threshold=self.nms_iou)

        # ── Step 7: Energy accounting ──────────────────────────────────────────
        energy_info = self.compute_energy(hr_img, hr_crop_boxes)

        debug = {
            'lr_img':   lr_img,
            'lr_dets':  debug_lr_dets,
            'hr_crops': hr_crop_boxes,
        }

        return final_dets, energy_info, debug
