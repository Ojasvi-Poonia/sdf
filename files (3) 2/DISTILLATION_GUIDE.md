# Distillation Pipeline — Step-by-Step Execution Guide

Train a small, fast, **Apache-2.0** appliance detector with **no manual dataset**, by using
**OWLv2** as an offline auto-labeler ("pseudo-label distillation"), then quantizing and exporting
the student for Samsung edge devices.

```
OWLv2 (teacher, offline)  →  pseudo-labels  →  small student detector  →  INT8  →  TFLite/NCNN  →  device
```

> **What "distillation" means here:** the teacher labels your images; the student trains on those
> labels. Teacher and student have totally different architectures — that's fine, because only
> *labels* are transferred (not logits/features). This is the robust, standard approach for
> deployment.

---

## 0. Prerequisites

- **Python 3.10+**, and ideally a **CUDA GPU** for auto-labeling and training (CPU works for tiny
  sets but is slow).
- Install dependencies:
  ```bash
  cd appliance-distillation
  python -m venv .venv && source .venv/bin/activate      # (Windows: .venv\Scripts\activate)
  pip install -r requirements.txt
  ```
- Edit **`config.py`** — set your `CLASSES`, matching `PROMPTS`, and the input size (`IMG_SIZE`).
  Everything else reads from this file.

**Project layout**
```
appliance-distillation/
├── config.py                  # EDIT THIS: classes, prompts, paths, thresholds
├── requirements.txt
├── 01_collect_openimages.py   # OPTIONAL: free Open Images boxes
├── 02_autolabel_owlv2.py      # CORE: OWLv2 labels your images
├── 03_build_dataset.py        # YOLO labels -> COCO train/valid/test
├── 04_quantize_int8.py        # ONNX -> INT8
└── data/
    ├── raw_images/   ├── labels/   ├── dataset/   └── calib/
```

---

## 1. Collect images

You need images containing your appliances. **Best results = your own captured photos** taken the
way the deployment camera will see appliances (homes, store shelves, real angles, partial views).
Aim for a few hundred per class to start.

Put them in `data/raw_images/`.

**Optional free boost — Open Images V7** (human-drawn boxes for many appliance classes):
```bash
python 01_collect_openimages.py
```
This writes Open-Images samples + ground-truth labels straight into `data/raw_images/` and
`data/labels/` using *your* class indices, so they merge with the OWLv2 labels automatically.
(Box annotations are CC-BY; verify individual **image** licenses for commercial use.)

---

## 2. Auto-label with the OWLv2 teacher  ← the core step

```bash
python 02_autolabel_owlv2.py
```
- Runs OWLv2 over every image **without an existing label**, applies your confidence threshold and
  per-class NMS, and writes one YOLO `.txt` per image into `data/labels/`.
- It **skips** images already labeled (so Open-Images ground-truth is preserved). To force
  re-labeling an image, delete its `.txt`.
- **Tuning knobs in `config.py`:** lower `SCORE_THRESHOLD` (e.g. 0.10–0.15) if the teacher misses
  appliances; raise it (0.25–0.30) if you get junk boxes. Add synonyms to `PROMPTS` to improve
  recall on a stubborn class.

---

## 3. Build the COCO dataset

```bash
python 03_build_dataset.py
```
Converts images + YOLO labels into `data/dataset/{train,valid,test}/` with a
`_annotations.coco.json` per split (RF-DETR layout; YOLOX can use the same JSONs).

**Verify the pseudo-labels before training** (highly recommended — the student inherits the
teacher's mistakes):
```python
import fiftyone as fo
ds = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="data/dataset/train",
    labels_path="data/dataset/train/_annotations.coco.json",
)
fo.launch_app(ds)   # eyeball boxes; delete/fix bad images, then re-run step 3
```
Fixing boxes is ~5–10× faster than drawing them, and it's the single biggest lever on final
accuracy.

---

## 4. (Concept) The dataset IS the distilled knowledge

At this point the teacher's knowledge now lives in `data/dataset/` as labels. The teacher is done —
it never ships. Everything from here is standard small-detector training.

---

## 5. Train the student detector

Pick **one** Apache/MIT student. Train at **low resolution** (`IMG_SIZE` = 320–416) — large
appliances survive it and it keeps the model fast on-device.

### Option A — RF-DETR (recommended: simplest API, Apache-2.0, edge-focused)
```python
# train_rfdetr.py
from rfdetr import RFDETRBase

model = RFDETRBase()                      # small variant; COCO-pretrained backbone
model.train(
    dataset_dir="data/dataset",           # expects train/ valid/ test/ with _annotations.coco.json
    epochs=100,
    batch_size=16,
    grad_accum_steps=1,
    lr=1e-4,
    resolution=448,                        # multiple of 56 for RF-DETR; 392/448 are good low-res
    output_dir="output",
)
```
```bash
python train_rfdetr.py
```

### Option B — YOLOX-Nano/Tiny (most battle-tested mobile export path)
```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX && cd YOLOX && pip install -e . && cd ..
```
Create `exps/appliance_yolox_nano.py`:
```python
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 7                       # = len(CLASSES)
        self.depth, self.width = 0.33, 0.25        # nano scale (tiny: 0.33, 0.375)
        self.input_size = self.test_size = (416, 416)
        self.data_dir = "data/dataset"
        self.train_ann = "train/_annotations.coco.json"
        self.val_ann   = "valid/_annotations.coco.json"
        self.max_epoch, self.eval_interval = 100, 5
```
```bash
python tools/train.py -f exps/appliance_yolox_nano.py -d 1 -b 16 --fp16 -c yolox_nano.pth
```
> YOLOX's COCO loader expects the image folder name to match where your images live; if it can't
> find images, point `self.data_dir` / annotation paths at the exact locations from step 3, or
> symlink them into YOLOX's expected `train2017/`-style folders.

**Starting from COCO-pretrained weights** (the `-c` flag / `RFDETRBase()` default) is strongly
recommended — it transfers general object features and cuts training time sharply.

---

## 6. Evaluate

Check mAP on the held-out **`test/`** split. RF-DETR and YOLOX both report COCO mAP during/after
training. For a trustworthy number, hand-verify the boxes in the test split first (a small clean
test set beats a large noisy one). If mAP is low, the usual cause is noisy pseudo-labels — go back
to step 3's verification, or adjust the teacher threshold in step 2.

---

## 7. Quantize to INT8

Export the trained student to **ONNX**, then quantize:
```python
# RF-DETR ONNX export
from rfdetr import RFDETRBase
m = RFDETRBase(pretrain_weights="output/checkpoint_best_total.pth")
m.export()        # writes ONNX into output/
```
(YOLOX: `python tools/export_onnx.py -f exps/appliance_yolox_nano.py -c output/.../best_ckpt.pth`)

Then:
```bash
# point ONNX_FP32 in 04_quantize_int8.py at your exported file, and put ~200
# representative images in data/calib/, then:
python 04_quantize_int8.py
```
> **Critical:** the `preprocess()` in `04_quantize_int8.py` must match your student's training
> normalization exactly (DETR-family use ImageNet mean/std; many YOLO variants use plain `/255`).
> Mismatched preprocessing is the #1 cause of "INT8 destroyed my accuracy."

---

## 8. Export & deploy to Samsung edge

From the INT8 ONNX, convert to the on-device runtime:

| Target | Path | Notes |
|--------|------|-------|
| **LiteRT / TFLite** | `onnx2tf -i model_int8.onnx -o tflite_out` | Then run with the **NNAPI** or GPU delegate |
| **NCNN / MNN** | ONNX → `pnnx` / `onnx2ncnn` | Popular, fast mobile CV runtimes |
| **Samsung NPU (Exynos)** | feed TFLite/ONNX to **ENN SDK** | Exynos Neural Network framework |
| **Snapdragon NPU** | **Qualcomm QNN / SNPE** | For Snapdragon-based Galaxies |

Then:
1. Run inference on-device, decode boxes, apply a final NMS.
2. **Benchmark latency/FPS on the real device** and adjust `IMG_SIZE` / student scale to hit your
   budget. A quantized YOLOX-Tiny / small RF-DETR at 320–416 px should be comfortably real-time on a
   modern Galaxy NPU.
3. **Frame-skip** for free headroom — appliances don't move, so detect every Nth frame and hold the
   result.

---

## 9. Extend without annotation

New appliance later? Add it to `CLASSES` + `PROMPTS` in `config.py`, drop in some images, re-run
steps 2 → 7. No manual labeling. The open-vocabulary flexibility stays in the offline pipeline; the
shipped model remains a tight, fast, fixed-class detector.

---

## Compute & time expectations (rough)

| Step | Hardware | Ballpark |
|------|----------|----------|
| OWLv2 labeling (a few thousand imgs) | 1 GPU | minutes–~1 hr |
| Student training (low-res, pretrained) | 1 GPU | ~1–4 hrs |
| INT8 quantization | CPU/GPU | minutes |
| TFLite/NCNN conversion | CPU | minutes |

The whole thing is a one-time offline cost on a single GPU — trivial for a product, and far cheaper
than annotating a dataset.

---

## Common pitfalls (read this)

1. **Category id 0 vs 1.** This pipeline writes 1-indexed COCO categories. If your trainer errors on
   data loading, this is the first thing to check.
2. **INT8 preprocessing mismatch.** Calibration/inference normalization must equal training
   normalization. See Stage 7.
3. **ONNX input name.** `04_quantize_int8.py` auto-prints it; make sure it's used.
4. **Pseudo-label noise.** The student can't exceed the teacher. Verify a sample (Stage 3) — it's
   the highest-leverage fix.
5. **Domain gap.** If your camera sees appliances very differently from the teacher's training data
   (extreme close-ups, heavy occlusion), label quality drops there. Capture in-domain images.
6. **Open Images class names.** Must match its exact taxonomy spelling (Title Case). "air
   conditioner" isn't a distinct Open Images class — OWLv2 covers it.
7. **Image licensing ≠ model licensing.** Source training images for commercial use; have legal
   confirm. OWLv2 / RF-DETR / YOLOX licenses are clean, but your *images* are a separate question.

---

*Not legal advice. Verify all third-party licenses with Samsung's OSS/legal team before production.*
