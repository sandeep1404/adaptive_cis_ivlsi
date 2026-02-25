import os
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copy2

# Paths
VOC_ROOT = Path("data/voc2007/VOCdevkit2007/VOC2007")
ANN_DIR = VOC_ROOT / "Annotations"
IMG_DIR = VOC_ROOT / "JPEGImages"
SET_DIR = VOC_ROOT / "ImageSets" / "Main"

# Output root
OUT_ROOT = Path("voc_person_car_yolo")
for sub in ["images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"]:
    (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

# Class mapping
CLASS_MAP = {
    "person": 0,
    "car": 1,
}

def convert_box(size, box):
    w_img, h_img = size
    xmin, ymin, xmax, ymax = box
    cx = (xmin + xmax) / 2.0 / w_img
    cy = (ymin + ymax) / 2.0 / h_img
    w = (xmax - xmin) / w_img
    h = (ymax - ymin) / h_img
    return cx, cy, w, h

def process_split(split_name, out_img_sub, out_label_sub):
    split_file = SET_DIR / f"{split_name}.txt"
    out_img_dir = OUT_ROOT / out_img_sub
    out_label_dir = OUT_ROOT / out_label_sub

    with open(split_file, "r") as f:
        ids = [line.strip() for line in f.readlines()]

    kept = 0
    for img_id in ids:
        xml_path = ANN_DIR / f"{img_id}.xml"
        if not xml_path.exists():
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        w_img = int(root.find("size/width").text)
        h_img = int(root.find("size/height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_MAP:
                continue

            cls_id = CLASS_MAP[name]
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            cx, cy, w, h = convert_box((w_img, h_img), (xmin, ymin, xmax, ymax))
            # Filter out degenerate boxes
            if w <= 0 or h <= 0:
                continue
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Skip if no person/car in this image
        if not yolo_lines:
            continue

        # Copy image
        src_img = IMG_DIR / f"{img_id}.jpg"
        if not src_img.exists():
            continue
        dst_img = out_img_dir / f"{img_id}.jpg"
        copy2(src_img, dst_img)

        # Write label
        dst_label = out_label_dir / f"{img_id}.txt"
        with open(dst_label, "w") as lf:
            lf.write("\n".join(yolo_lines))

        kept += 1

    print(f"{split_name}: kept {kept} images with person/car")

if __name__ == "__main__":
    process_split("train", "images/train", "labels/train")
    process_split("val", "images/val", "labels/val")
    # process_split("test", "images/test", "labels/test")

    test_split = SET_DIR / "test.txt"
    if test_split.exists():
        process_split("test", "images/test", "labels/test")
    else:
        print("No test.txt in ImageSets/Main, skipping test split.")
