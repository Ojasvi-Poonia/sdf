"""
Central configuration for the appliance-detection distillation pipeline.
Edit this file (especially CLASSES / PROMPTS / paths) and every script picks it up.
"""

# --- Classes -------------------------------------------------------------
# Your final appliance set. Keep it CATEGORY-level (no brands/models).
# The student detector will output exactly these classes, in this order.
CLASSES = [
    "television",
    "microwave oven",
    "air conditioner",
    "washing machine",
    "refrigerator",
    "dishwasher",
    "oven",
]

# Text prompts handed to the OWLv2 teacher. Synonyms raise recall.
# IMPORTANT: PROMPTS must be the same length & order as CLASSES.
# Each inner list = the phrases that all map to that one class.
PROMPTS = [
    ["a television", "a tv", "a flat screen tv"],
    ["a microwave oven", "a microwave"],
    ["an air conditioner", "an ac unit", "a split air conditioner"],
    ["a washing machine", "a clothes washer"],
    ["a refrigerator", "a fridge"],
    ["a dishwasher"],
    ["an oven", "a wall oven"],
]

assert len(CLASSES) == len(PROMPTS), "CLASSES and PROMPTS must align 1:1"

# --- Paths ---------------------------------------------------------------
RAW_IMAGES_DIR = "data/raw_images"   # all training images land here (yours + Open Images)
LABELS_DIR     = "data/labels"       # YOLO-format labels land here (1 .txt per image)
DATASET_DIR    = "data/dataset"      # final COCO dataset (train/valid/test) for the student
CALIB_DIR      = "data/calib"        # ~200 representative images for INT8 calibration

# --- Teacher (OWLv2) -----------------------------------------------------
OWLV2_MODEL     = "google/owlv2-base-patch16-ensemble"  # explicit Apache-2.0 weights
SCORE_THRESHOLD = 0.20   # detection confidence cutoff. Tune 0.10-0.30 per your data.
NMS_IOU         = 0.50   # per-class NMS IoU to merge duplicate boxes from synonyms

# --- Student / dataset ---------------------------------------------------
IMG_SIZE      = 416   # large objects survive low res; 320 or 416 keeps it fast on-device
VAL_FRACTION  = 0.15
TEST_FRACTION = 0.10
SEED          = 42
