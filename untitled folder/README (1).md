# Real-Time Open-Vocabulary Appliance Detection on Edge Devices

A technical research summary and implementation guide for detecting large home appliances
(TV, microwave, AC, washing machine, refrigerator, dishwasher, oven, etc.) in real time on
mobile/edge hardware, **without a labeled dataset** and using **only commercially-permissive
(Apache-2.0 / MIT) models**.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Constraints](#constraints)
3. [TL;DR — Recommended Solution](#tldr--recommended-solution)
4. [The Key Reframe](#the-key-reframe)
5. [Licensing Landscape (the decisive factor)](#licensing-landscape-the-decisive-factor)
6. [Approaches Considered](#approaches-considered)
7. [Recommended Path B — Distillation (production-optimal)](#recommended-path-b--distillation-production-optimal)
8. [No-Training Path A — Run an Open-Vocab Detector Directly](#no-training-path-a--run-an-open-vocab-detector-directly)
9. [Deployment Notes (Samsung edge)](#deployment-notes-samsung-edge)
10. [Key Insights](#key-insights)
11. [Caveats & Risks](#caveats--risks)
12. [Open Questions / Next Steps](#open-questions--next-steps)
13. [References](#references)

---

## Problem Statement

Detect a fixed set of **large home appliances** in real time. Some target classes are **not in
the COCO dataset**, so a vanilla COCO-trained detector is insufficient. There is **no labeled
training dataset** for these appliances, and collecting/annotating one is expensive. The solution
must run on **mobile / edge devices** (e.g., Samsung Galaxy NPUs) and must use **models that are
licensed for commercial use** in a Samsung product (Apache-2.0 or MIT).

---

## Constraints

| # | Constraint | Implication |
|---|------------|-------------|
| 1 | Real-time on mobile/edge | Model must be small, quantizable, NPU-friendly |
| 2 | No labeled dataset | Need zero-shot / open-vocab, or auto-labeling |
| 3 | Classes outside COCO | Can't rely on a fixed COCO class list |
| 4 | Apache-2.0 / MIT only | Eliminates GPL/AGPL and research-only models |

---

## TL;DR — Recommended Solution

> **Use a heavyweight open-vocabulary detector OFFLINE as an auto-labeler, then distill its
> knowledge into a tiny, fast, Apache-licensed detector that ships on the phone.**

- **Teacher (offline only):** Grounding DINO or OWLv2 — both Apache-2.0 — prompted with your
  appliance class names. Generates bounding-box pseudo-labels with zero manual annotation.
- **Free labels:** Pull existing bounding boxes for common appliances from Open Images V7 (box
  annotations are CC-BY) to bootstrap the dataset.
- **Student (ships on device):** YOLOX-Tiny / NanoDet-Plus / PP-YOLOE / RF-DETR — all Apache-2.0 —
  trained at low input resolution (320–416 px), then INT8-quantized.
- **Result:** A small fixed-class detector that is real-time on a Samsung NPU, needs no manual
  dataset, and is clean on licensing end-to-end. Nothing GPL/AGPL/research-only ever ships.

This is also the approach the edge open-vocabulary literature converges on (see References:
Frontiers 2025 survey).

A **no-training** alternative exists (Path A below) and is license-clean, but it is **not
phone-real-time** — it is best for prototyping or edge-GPU hardware.

---

## The Key Reframe

The problem statement says "open-vocabulary model," but the *actual* need is:

> **Detect a fixed, known, small set of common appliances — without a labeled dataset.**

That is **not** the same as needing open-vocabulary detection *at inference time*. Because the
class list is fixed and the appliances are common (they exist in LVIS / Objects365 training sets),
you can let the "open-vocabulary" capability live entirely in your **offline labeling pipeline**,
and ship a tight **fixed-class** detector that is far faster and lighter on-device.

---

## Licensing Landscape (the decisive factor)

Licensing — not capability — is what eliminates the most obvious options. The fast open-vocab
detectors are copyleft; the permissively-licensed open-vocab detectors are too heavy for phone
real-time.

### Open-vocabulary detectors

| Model | Vendor | License | Ship in product? | Notes |
|-------|--------|---------|------------------|-------|
| **OWLv2 / OWL-ViT** | Google | **Apache-2.0** | ✅ | Zero-shot from text prompts; ViT-based, heavy |
| **Grounding DINO** | IDEA Research | **Apache-2.0** | ✅ | Strong boxes; heavy. Ideal offline teacher |
| **Grounded SAM** | IDEA Research | **Apache-2.0** | ✅ | Open-vocab segmentation (boxes + masks) |
| YOLO-World | Tencent | **GPL-3.0** | ❌ | Commercial license available from Tencent only |
| YOLOE ("Seeing Anything") | Tsinghua / Ultralytics | **AGPL-3.0** | ❌ | Built on Ultralytics; enterprise license needed |

### CLIP-style image–text models

| Model | Vendor | License | Ship in product? | Notes |
|-------|--------|---------|------------------|-------|
| **SigLIP / SigLIP2** | Google | **Apache-2.0** | ✅ | Recommended CLIP substitute |
| **OpenCLIP (LAION)** | LAION | MIT / Apache (varies by checkpoint) | ✅ | Verify the specific checkpoint license |
| MobileCLIP — **code** | Apple | **MIT** | ✅ (code only) | Inference/training code only |
| MobileCLIP — **weights** | Apple | **Apple ML Research Model License** | ❌ | Research-only, revocable. Covers checkpoints AND derivatives (fine-tunes/distillations) |
| MobileCLIP — **training data (DataCompDR)** | Apple | **CC-BY-NC-ND** | ❌ | Non-commercial; blocks reproducing the weights yourself |

> ⚠️ **MobileCLIP cannot be used in a commercial product** — neither embedded directly nor as a
> distillation teacher, because a distilled student would be a "Model Derivative" under Apple's
> research-only license. Use SigLIP/OpenCLIP instead if you need a CLIP component.

### Student detectors (what actually ships)

| Model | Vendor | License | Ship in product? | Notes |
|-------|--------|---------|------------------|-------|
| **YOLOX (Nano/Tiny)** | Megvii | **Apache-2.0** | ✅ | Mobile-friendly, strong export tooling |
| **NanoDet-Plus** | open-source | **Apache-2.0** | ✅ | Purpose-built for mobile, extremely small |
| **PP-YOLOE** | Baidu / PaddleDetection | **Apache-2.0** | ✅ | Good Paddle Lite mobile path |
| **RT-DETR (original)** | Baidu / PaddlePaddle | **Apache-2.0** | ✅ | Use the PaddlePaddle version, NOT the Ultralytics port |
| **RF-DETR** | Roboflow | **Apache-2.0** | ✅ | Real-time transformer detector, edge-focused (2025) |
| Ultralytics YOLOv5/8/11, RT-DETR port | Ultralytics | **AGPL-3.0** | ❌ | Enterprise license needed |

### Proposers / segmenters (for the two-stage no-training option)

| Model | License | Ship in product? | Notes |
|-------|---------|------------------|-------|
| **SAM** (Segment Anything) | **Apache-2.0** | ✅ | Heavy; class-agnostic masks |
| **MobileSAM** | **Apache-2.0** | ✅ | ~60× smaller than SAM, ~10 ms/image |
| EdgeSAM | (verify before use) | ⚠️ | Mobile-optimized SAM variant — confirm license |

### Offline tooling (does not ship — license is low-risk)

| Tool | License | Purpose |
|------|---------|---------|
| **Autodistill** | **Apache-2.0** | Orchestrates "auto-label with a foundation model → distill" |
| **FiftyOne** (Voxel51) | **Apache-2.0** | Dataset curation; pulls Open Images subsets by class |

> **"Ship in product?"** refers to embedding the model's weights/code in a proprietary product.
> Tools used **offline only** (teachers used purely for labeling, Autodistill, FiftyOne) do not
> ship, so their licenses are even lower-risk. **All license interpretations must be confirmed by
> Samsung's OSS/legal team before relying on them.**

---

## Approaches Considered

### A. Quadrant + MobileCLIP (the original idea) — ❌ not recommended

Divide the frame into quadrants, classify each with CLIP, recurse into promising quadrants.

- ❌ CLIP/MobileCLIP is a **whole-image classifier, not a detector** — unreliable on partial /
  off-center crops; large appliances get cut across quadrant boundaries.
- ❌ **No real bounding boxes** — localization is quantized to the grid.
- ❌ **Latency explodes** — many forward passes per frame; won't hit real-time on a phone.
- ❌ **Threshold fragility** — "is an object here?" via cosine similarity drifts across scenes.
- ❌ **MobileCLIP weights are research-only** — license-blocked regardless.

### B. Ship a real-time open-vocab detector directly (YOLO-World / YOLOE) — ❌ license-blocked

The natural fit (fast + open-vocab + edge-capable), but **GPL-3 / AGPL-3** — cannot ship in a
proprietary Samsung product without a commercial license.

### C. No-training: run OWLv2 / Grounding DINO on-device — ✅ clean, ⚠️ slow

Apache-2.0, zero training, returns boxes from text prompts. **But** ViT-based and not phone-real-time
(a few FPS at best). Good for prototyping or edge-GPU (Jetson-class) hardware. See **Path A**.

### D. No-training two-stage: MobileSAM + SigLIP/OpenCLIP — ✅ clean, ⚠️ slow

Class-agnostic proposals (MobileSAM) → crop → classify with SigLIP/OpenCLIP. A "done-right" version
of the quadrant idea (real proposals, full-object crops). All Apache, no training, but more moving
parts and still not phone-real-time.

### E. ⭐ Distillation: open-vocab teacher → tiny student — ✅ RECOMMENDED

Auto-label offline with an Apache teacher, distill to a tiny Apache student that ships. Hits all
four constraints. See **Path B**.

| Approach | Real-time on phone | No dataset | Non-COCO classes | Apache/MIT clean |
|----------|:------------------:|:----------:|:----------------:|:----------------:|
| A. Quadrant + MobileCLIP | ❌ | ✅ | ✅ | ❌ |
| B. Ship YOLO-World/YOLOE | ✅ | ✅ | ✅ | ❌ |
| C. OWLv2/GDINO on device | ❌ (~few FPS) | ✅ | ✅ | ✅ |
| D. MobileSAM + SigLIP | ❌ (~few FPS) | ✅ | ✅ | ✅ |
| **E. Distillation** | ✅ | ✅ | ✅ | ✅ |

---

## Recommended Path B — Distillation (production-optimal)

A one-time offline training run buys a model that is an order of magnitude faster and lighter
on-device than any open-vocab model. The "training" is offline (hours on a single GPU), not a
runtime or per-device cost.

### Stage 1 — Classes & prompts
List your ~10–20 appliance classes; write text prompts with synonyms per class
(e.g., "television / TV / flat-screen TV", "air conditioner / AC unit / split AC"). Keep it
**category-level**, not brand/model-level.

### Stage 2 — Gather images (two cheap sources)

**(a) Free labeled boxes from Open Images V7** (box annotations CC-BY; verify per-image licenses):

```python
import fiftyone.zoo as foz

ds = foz.load_zoo_dataset(
    "open-images-v7", split="train", label_types=["detections"],
    classes=["Television", "Microwave oven", "Refrigerator",
             "Washing machine", "Dishwasher", "Oven"],
    max_samples=3000,
)
```

**(b) Your own captured photos** of appliances as the deployment camera will see them (homes,
store shelves, real angles). Most license-clean and domain-matched. A few hundred per class.

### Stage 3 — Auto-label your own images (Grounding DINO via Autodistill)

```python
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

ontology = CaptionOntology({
    "television": "tv", "microwave oven": "microwave",
    "air conditioner": "ac", "washing machine": "washing_machine",
    "refrigerator": "fridge", "dishwasher": "dishwasher",
})
GroundingDINO(ontology=ontology).label(
    input_folder="./my_images", output_folder="./dataset", extension=".jpg")
```

Then do a **quick human verification pass** (CVAT or FiftyOne). Fixing boxes is ~5–10× faster than
drawing them and raises the accuracy ceiling, since the student inherits the teacher's mistakes.

### Stage 4 — Train the tiny student
Pick one Apache/MIT detector and train at **low input resolution (320–416 px)** — large appliances
survive downscaling, which keeps inference cheap:

- **RF-DETR** (simplest Python training API; real-time transformer, edge-focused)
- **YOLOX-Nano/Tiny** (well-trodden mobile export path)
- **NanoDet-Plus** (smallest, mobile-first)
- **PP-YOLOE** (strong Paddle Lite path)

Train on merged Open Images + auto-labeled data. Because labels carry some noise, keep augmentation
modest; consider confidence-weighting.

### Stage 5 — Optimize & export
- **INT8 post-training quantization** with a small calibration set (use QAT only if accuracy drops too much).
- Export **ONNX → LiteRT (TFLite)** with NNAPI/GPU delegate, or **ONNX → NCNN / MNN**.
- Target the NPU via **Samsung ENN** (Exynos) or **Qualcomm QNN / SNPE** (Snapdragon).
- A quantized YOLOX-Tiny / NanoDet at 320 px comfortably hits real-time on a modern Galaxy NPU.

### Stage 6 — Extend without annotation
New appliance later? Add a prompt, re-run the teacher on fresh images, re-distill. The open-vocab
flexibility stays in the offline pipeline; the shipped model stays small and fixed.

---

## No-Training Path A — Run an Open-Vocab Detector Directly

License-clean and stands up in ~a day. **Not phone-real-time** — expect a few FPS on a phone NPU,
better on an edge GPU. Use it for prototyping, or as the labeling engine that feeds Path B.

### Step 1 — Install
```bash
pip install transformers torch pillow
```

### Step 2 — Run it on a frame (this is the whole core)
```python
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
from PIL import Image

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()

prompts = [["a television", "a microwave oven", "an air conditioner",
            "a washing machine", "a refrigerator", "a dishwasher"]]
image = Image.open("frame.jpg")
inputs = processor(text=prompts, images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_object_detection(
    outputs, threshold=0.2, target_sizes=torch.tensor([image.size[::-1]]))[0]
# results["boxes"], results["scores"], results["labels"]
```

### Step 3 — Tune
Add synonyms to prompts, lower the threshold (0.1–0.25) for hard classes, apply NMS to dedupe boxes.

### Step 4 — Make it as fast as possible on-device
- **Cache text embeddings.** Classes are fixed → run the text encoder once, offline; feed cached
  embeddings to the detection head at runtime. Removes the entire text tower from inference.
- **INT8-quantize the image encoder** and reduce input resolution.
- **Frame-skip.** Appliances don't move; run detection on 1 of every N frames and hold the result.

### Step 5 — Deploy
Export the (text-embedding-baked) image model to **ONNX → ONNX Runtime Mobile**, or target the NPU
via **ENN / QNN**. Note: ViT export to mobile NPUs often hits unsupported ops — expect GPU-delegate
fallback for some layers, and do real on-device latency testing.

---

## Deployment Notes (Samsung edge)

- **Quantization:** INT8 PTQ first; QAT only if needed. Tiny CNN detectors quantize and run on
  mobile NPUs far better than ViT-based open-vocab models.
- **Runtimes:** LiteRT (TFLite) + NNAPI/GPU delegate, ONNX Runtime Mobile, NCNN, MNN.
- **NPU SDKs:** Samsung ENN (Exynos), Qualcomm QNN/SNPE (Snapdragon Galaxies).
- **Resolution:** 320–416 px is plenty for large appliances and keeps latency/power low.
- **Always benchmark on the real target device** and trade off resolution vs. model size to hit
  the FPS/power budget.

---

## Key Insights

1. **"Large objects" is an advantage, not a problem.** Big appliances survive aggressive
   downscaling, so a small student at low resolution is fast and sufficient. The quadrant recursion
   exists to recover *small* objects at high res — a problem you don't have.
2. **You don't need runtime open-vocabulary.** A fixed, known appliance set means the open-vocab
   capability can live offline (in labeling), and the shipped model can be a fast fixed-class detector.
3. **No-training and distillation are not either/or.** The no-training model (OWLv2/Grounding DINO)
   *is* the offline labeling engine for the distilled model. Prototype with Path A, ship Path B.
4. **Licensing is the real gate.** Capability-wise, YOLO-World/YOLOE are ideal — but GPL/AGPL kills
   them for a proprietary product. The clean stack is Apache teacher + Apache student.

---

## Caveats & Risks

- **Pseudo-label quality:** the student inherits the teacher's errors and won't meaningfully exceed
  it. A small human verification pass is the highest-leverage quality lever.
- **Domain gap:** if the camera sees appliances very differently from the teacher's training
  distribution (extreme close-ups, occlusion, odd angles), label quality drops — bias image
  collection toward in-domain captures.
- **Image licensing ≠ model licensing:** training images must be sourced for commercial use (own
  captures, or properly-licensed datasets like Open Images). A separate legal question.
- **Brand/model-level recognition is out of scope** for this approach (it detects generic
  categories). "Is this a Samsung Bespoke fridge?" would need a fine-grained classifier on labeled
  brand data.
- **Final legal sign-off:** Apple/Tencent/Ultralytics terms can change, and license interpretation
  is Samsung OSS/legal's call. Verify before relying on anything here.

---

## Open Questions / Next Steps

To turn this into concrete end-to-end code, three decisions are needed:

1. **Final class list** (how many appliances, and which).
2. **Target chipset** — Exynos (→ ENN) vs. Snapdragon (→ QNN/SNPE).
3. **Granularity** — category-level (this plan) vs. brand/model-level (needs extra work).

**Suggested first move:** stand up Path A this week to prove feasibility and produce a baseline +
labels, then run Path B to produce the shipping model.

---

## References

- YOLO-World (Tencent, GPL-3.0): https://github.com/AILab-CVC/YOLO-World — paper: https://arxiv.org/abs/2401.17270
- YOLOE (Tsinghua/Ultralytics, AGPL-3.0): https://github.com/THU-MIG/yoloe — paper: https://arxiv.org/abs/2503.07465
- OWLv2 (Google, Apache-2.0): https://huggingface.co/docs/transformers/model_doc/owlv2 — weights: https://huggingface.co/google/owlv2-base-patch16-ensemble
- OWL-ViT (Google, Apache-2.0): https://huggingface.co/docs/transformers/model_doc/owlvit
- Grounding DINO (IDEA Research, Apache-2.0) — via Autodistill: https://github.com/autodistill/autodistill-grounding-dino
- Grounded SAM (IDEA Research, Apache-2.0): https://github.com/IDEA-Research/Grounded-Segment-Anything
- MobileCLIP (Apple — code MIT / weights research-only / data CC-BY-NC-ND): https://github.com/apple/ml-mobileclip
- SigLIP / SigLIP2 (Google, Apache-2.0): https://huggingface.co/docs/transformers/model_doc/siglip
- YOLOX (Megvii, Apache-2.0): https://github.com/Megvii-BaseDetection/YOLOX
- NanoDet (Apache-2.0): https://github.com/RangiLyu/nanodet
- PP-YOLOE / PaddleDetection (Baidu, Apache-2.0): https://github.com/PaddlePaddle/PaddleDetection
- RF-DETR (Roboflow, Apache-2.0): https://github.com/roboflow/rf-detr
- SAM (Meta, Apache-2.0): https://github.com/facebookresearch/segment-anything
- MobileSAM (Apache-2.0): https://github.com/ChaoningZhang/MobileSAM
- Autodistill (Apache-2.0): https://github.com/autodistill/autodistill
- FiftyOne / Open Images V7 (Voxel51, Apache-2.0): https://docs.voxel51.com/
- Edge open-vocab perception survey (Frontiers in Robotics and AI, Oct 2025): https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1693988/full

---

*This document summarizes research and engineering recommendations only. It is not legal advice.
All third-party license terms must be independently verified with Samsung's OSS/legal team before
use in a commercial product.*
