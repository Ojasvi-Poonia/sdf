"""
Stage 2 (CORE): Auto-label images with the OWLv2 teacher.

Reads every image in config.RAW_IMAGES_DIR, runs OWLv2 with your appliance
prompts, applies a confidence threshold + per-class NMS, and writes one
YOLO-format label file per image into config.LABELS_DIR.

This is the heart of the "distillation": the teacher's detections become the
pseudo-ground-truth the small student will train on.

Run:  python 02_autolabel_owlv2.py
GPU strongly recommended (CPU works for small sets but is slow).
"""
import os
import glob

import torch
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import config

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {config.OWLV2_MODEL} on {device} ...")
processor = Owlv2Processor.from_pretrained(config.OWLV2_MODEL)
model = Owlv2ForObjectDetection.from_pretrained(config.OWLV2_MODEL).to(device).eval()

# Flatten prompts into one query list, remembering which class each prompt is.
flat_prompts, prompt_to_class = [], []
for class_idx, synonyms in enumerate(config.PROMPTS):
    for phrase in synonyms:
        flat_prompts.append(phrase)
        prompt_to_class.append(class_idx)

os.makedirs(config.LABELS_DIR, exist_ok=True)

image_paths = sorted(
    glob.glob(os.path.join(config.RAW_IMAGES_DIR, "*.jpg"))
    + glob.glob(os.path.join(config.RAW_IMAGES_DIR, "*.jpeg"))
    + glob.glob(os.path.join(config.RAW_IMAGES_DIR, "*.png"))
)
if not image_paths:
    raise SystemExit(f"No images found in {config.RAW_IMAGES_DIR}")

# Skip images that ALREADY have a label file (e.g. Open Images ground-truth from
# script 01, or a previous run). To force re-labeling, delete the image's .txt.
def label_path_for(img_path):
    return os.path.join(config.LABELS_DIR,
                        os.path.splitext(os.path.basename(img_path))[0] + ".txt")

to_label = [p for p in image_paths if not os.path.exists(label_path_for(p))]
print(f"{len(image_paths)} images total; {len(to_label)} need labeling "
      f"({len(image_paths) - len(to_label)} already labeled, skipping).")

n_boxes_total = 0
for img_path in tqdm(to_label, desc="OWLv2 labeling"):
    image = Image.open(img_path).convert("RGB")
    W, H = image.size

    inputs = processor(text=[flat_prompts], images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # target_sizes is (height, width); OWLv2 returns boxes in absolute pixels.
    target_sizes = torch.tensor([[H, W]], device=device)
    result = processor.post_process_object_detection(
        outputs, threshold=config.SCORE_THRESHOLD, target_sizes=target_sizes
    )[0]

    boxes = result["boxes"]            # (N, 4) xyxy in pixels
    scores = result["scores"]          # (N,)
    prompt_labels = result["labels"]   # (N,) index into flat_prompts

    if boxes.numel() == 0:
        # still write an (empty) label file so the image is treated as a negative
        open(os.path.join(config.LABELS_DIR,
                          os.path.splitext(os.path.basename(img_path))[0] + ".txt"), "w").close()
        continue

    # Map each detection's prompt index -> our class index.
    cls = torch.tensor([prompt_to_class[int(l)] for l in prompt_labels], device=device)

    # Per-class NMS to dedupe boxes produced by different synonyms of the same class.
    keep_idx = []
    for c in cls.unique():
        mask = cls == c
        sub = torch.nonzero(mask).squeeze(1)
        kept = nms(boxes[mask], scores[mask], config.NMS_IOU)
        keep_idx.append(sub[kept])
    keep = torch.cat(keep_idx)

    # Write YOLO format: "<class> <cx> <cy> <w> <h>" normalized to [0,1].
    base = os.path.splitext(os.path.basename(img_path))[0]
    with open(os.path.join(config.LABELS_DIR, base + ".txt"), "w") as f:
        for i in keep.tolist():
            x1, y1, x2, y2 = boxes[i].tolist()
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            # clamp to valid range
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
            f.write(f"{int(cls[i])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            n_boxes_total += 1

with open(os.path.join(config.LABELS_DIR, "classes.txt"), "w") as f:
    f.write("\n".join(config.CLASSES))

print(f"Done. {len(image_paths)} images, {n_boxes_total} pseudo-boxes -> {config.LABELS_DIR}")
print("Next: verify a sample of labels, then run 03_build_dataset.py")
