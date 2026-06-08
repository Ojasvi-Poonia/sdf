"""
Stage 3: Convert images + YOLO pseudo-labels into a COCO-format dataset with
train/valid/test splits, laid out the way RF-DETR expects:

    data/dataset/
        train/  <images> + _annotations.coco.json
        valid/  <images> + _annotations.coco.json
        test/   <images> + _annotations.coco.json

YOLOX / other COCO trainers can use these same JSON files (point your Exp at them).

NOTE on category ids: COCO categories here are 1-indexed (id = class_index + 1),
which is the standard COCO convention. 0- vs 1-based category ids is the single
most common silent dataset bug -- if your trainer errors on data loading, check
this first.

Run:  python 03_build_dataset.py
"""
import os
import glob
import json
import random
import shutil

from PIL import Image

import config

random.seed(config.SEED)


def write_split(pairs, split):
    split_dir = os.path.join(config.DATASET_DIR, split)
    os.makedirs(split_dir, exist_ok=True)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(config.CLASSES)],
    }
    ann_id = 1
    for img_id, (img_path, lbl_path) in enumerate(pairs, start=1):
        with Image.open(img_path) as im:
            W, H = im.size
        fname = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(split_dir, fname))
        coco["images"].append({"id": img_id, "file_name": fname, "width": W, "height": H})
        if os.path.exists(lbl_path):
            for line in open(lbl_path):
                parts = line.split()
                if len(parts) != 5:
                    continue
                c = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                w, h = bw * W, bh * H
                x, y = cx * W - w / 2.0, cy * H - h / 2.0
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": c + 1,        # 1-indexed
                    "bbox": [x, y, w, h],         # COCO = [x, y, w, h] top-left
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1
    with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f)
    print(f"  {split:5s}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")


# Gather (image, label) pairs.
images = sorted(
    glob.glob(os.path.join(config.RAW_IMAGES_DIR, "*.jpg"))
    + glob.glob(os.path.join(config.RAW_IMAGES_DIR, "*.jpeg"))
    + glob.glob(os.path.join(config.RAW_IMAGES_DIR, "*.png"))
)
if not images:
    raise SystemExit(f"No images found in {config.RAW_IMAGES_DIR}. Run 01/02 first.")

pairs = []
for img in images:
    base = os.path.splitext(os.path.basename(img))[0]
    pairs.append((img, os.path.join(config.LABELS_DIR, base + ".txt")))

random.shuffle(pairs)
n_test = int(len(pairs) * config.TEST_FRACTION)
n_val = int(len(pairs) * config.VAL_FRACTION)

# Guard: trainers (RF-DETR/YOLOX) require a non-empty valid (and often test) split.
# With small datasets the fractions can round to 0, so force at least 1 each when possible.
if len(pairs) >= 5:
    n_val = max(n_val, 1)
    n_test = max(n_test, 1)
if n_val + n_test >= len(pairs):
    raise SystemExit(f"Only {len(pairs)} images — too few to split. Add more images first.")

test_pairs = pairs[:n_test]
val_pairs = pairs[n_test:n_test + n_val]
train_pairs = pairs[n_test + n_val:]

print(f"Building COCO dataset in {config.DATASET_DIR} ...")
write_split(train_pairs, "train")
write_split(val_pairs, "valid")
write_split(test_pairs, "test")
print("Done. Next: verify in FiftyOne (optional) then train the student (see GUIDE Stage 5).")
