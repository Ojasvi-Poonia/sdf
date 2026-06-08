"""
Stage 1 (OPTIONAL): Pull an Open Images V7 subset with human-drawn boxes for the
appliance classes that exist in its taxonomy, and write them straight into
config.RAW_IMAGES_DIR + config.LABELS_DIR using YOUR class indices, so they merge
seamlessly with the OWLv2 pseudo-labels.

Why: free, high-quality ground-truth boxes for common appliances.
Caveat: Open Images box ANNOTATIONS are CC-BY, but individual IMAGE licenses vary
        (many CC-BY, some not). Confirm image licenses for commercial training.

Run:  python 01_collect_openimages.py
Requires: pip install fiftyone
"""
import os
import shutil

import fiftyone as fo
import fiftyone.zoo as foz

import config

# Map Open Images class names (Title Case) -> your class names.
# Omit classes Open Images does not have (e.g. "air conditioner") -> OWLv2 covers those.
OI_CLASS_MAP = {
    "Television": "television",
    "Microwave oven": "microwave oven",
    "Washing machine": "washing machine",
    "Refrigerator": "refrigerator",
    "Dishwasher": "dishwasher",
    "Oven": "oven",
}

oi_names = list(OI_CLASS_MAP.keys())
name_to_idx = {name: i for i, name in enumerate(config.CLASSES)}

print("Downloading Open Images V7 subset (this can take a while)...")
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=oi_names,
    max_samples=3000,           # tune as you like
    dataset_name="oi_appliances",
)

# Field that holds the detections differs across FiftyOne versions.
label_field = "ground_truth" if dataset.has_field("ground_truth") else "detections"
print(f"Using label field: {label_field}")

os.makedirs(config.RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(config.LABELS_DIR, exist_ok=True)

written = 0
for sample in dataset.iter_samples(progress=True):
    dets = sample[label_field]
    if dets is None:
        continue
    rows = []
    for d in dets.detections:
        if d.label in OI_CLASS_MAP:
            cls_idx = name_to_idx[OI_CLASS_MAP[d.label]]
            # FiftyOne bounding_box = [top-left-x, top-left-y, width, height], normalized.
            x, y, w, h = d.bounding_box
            cx, cy = x + w / 2.0, y + h / 2.0
            rows.append((cls_idx, cx, cy, w, h))
    if not rows:
        continue
    fname = "oi_" + os.path.basename(sample.filepath)
    shutil.copy(sample.filepath, os.path.join(config.RAW_IMAGES_DIR, fname))
    base = os.path.splitext(fname)[0]
    with open(os.path.join(config.LABELS_DIR, base + ".txt"), "w") as f:
        for cls_idx, cx, cy, w, h in rows:
            f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    written += 1

print(f"Wrote {written} Open Images samples into the pipeline.")
print("Add your OWN captured images to data/raw_images too, then run 02_autolabel_owlv2.py")
print("(02 will label any image that does NOT already have a .txt; to re-label, delete its .txt)")
