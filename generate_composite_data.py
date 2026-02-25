import os, cv2, shutil, numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def generate_composite_dataset(
    input_dir,          # original full-res dataset
    output_dir,         # where to save composite images
    lr_model_path,      # trained LR detector
    row_skip=3,         # ×4 skip
    col_skip=3,
    lr_conf=0.10,
    padding_ratio=0.07
):
    """
    For each training image:
      1. Simulate ×4 line-skipping  → blocky LR image
      2. Run LR detector             → get bboxes
      3. Paste real HR pixels        → composite image
      4. Save composite + ORIGINAL labels (GT, not LR predictions!)

    The ORIGINAL labels are used because:
      - LR detector may miss some objects
      - We want the composite detector to learn from GT truth
      - This avoids label noise from imperfect LR predictions
    """

    lr_model = YOLO(lr_model_path)
    step_r, step_c = row_skip + 1, col_skip + 1

    for split in ['train', 'val', 'test']:
        os.makedirs(f'{output_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{output_dir}/labels/{split}', exist_ok=True)

    for split in ['train', 'val', 'test']:
        img_dir   = f'{input_dir}/images/{split}'
        label_dir = f'{input_dir}/labels/{split}'
        if not os.path.exists(img_dir):
            continue

        files = [f for f in os.listdir(img_dir)
                 if f.endswith(('.jpg', '.png', '.jpeg'))]

        for fname in tqdm(files, desc=split):
            img = cv2.imread(f'{img_dir}/{fname}')
            if img is None: continue

            img = cv2.resize(img, (640, 640))
            H, W = img.shape[:2]

            # Step 1: LR simulation
            lr_img = cv2.resize(
                img[::step_r, ::step_c],
                (W, H), interpolation=cv2.INTER_NEAREST
            )

            # Step 2: LR detections (for deciding WHERE to paste HR)
            result = lr_model.predict(lr_img, conf=lr_conf,
                                      imgsz=640, verbose=False)[0]

            # Step 3: Build composite
            composite = lr_img.copy()
            if result.boxes is not None and len(result.boxes) > 0:
                for b in result.boxes:
                    x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
                    bw, bh = x2-x1, y2-y1
                    px, py = int(bw*padding_ratio), int(bh*padding_ratio)
                    cx1 = max(0, x1-px); cy1 = max(0, y1-py)
                    cx2 = min(W-1, x2+px); cy2 = min(H-1, y2+py)
                    # Paste REAL HR pixels into the composite
                    composite[cy1:cy2, cx1:cx2] = img[cy1:cy2, cx1:cx2]

            # Step 4: Save composite image
            cv2.imwrite(f'{output_dir}/images/{split}/{fname}', composite)

            # Step 5: Copy ORIGINAL GT labels (not LR predictions!)
            label_file = fname.rsplit('.', 1)[0] + '.txt'
            src = f'{label_dir}/{label_file}'
            dst = f'{output_dir}/labels/{split}/{label_file}'
            if os.path.exists(src):
                shutil.copy(src, dst)

    # Write data.yaml
    with open(f'{output_dir}/data.yaml', 'w') as f:
        f.write(f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test
names:
  0: person
  1: car
nc: 2
""")
    print(f"✅ Composite dataset saved to: {output_dir}")


generate_composite_dataset(
    input_dir='data/pascal_person_car_yolo',
    output_dir='data/pascal_composite_4x',
    lr_model_path='/home/sandeep/Desktop/adaptive_cis/runs/step1/pascal_person_car/both_skip3_640/weights/best.pt',
    row_skip=3, col_skip=3,
    lr_conf=0.10
)
