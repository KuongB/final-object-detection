"""Single source of truth for paths, classes and per-model training settings.

Everything that needs to know *where* something lives, or *what* the five
classes are, imports it from here - so a path only ever has to change in one
place, and the three training pipelines cannot silently disagree about the
class order.

Path resolution is deliberately environment-aware: `PROJECT_ROOT` is derived
from this file's location, never hard-coded, so the same package runs from
`C:\\Users\\...\\final-object-detection` on the laptop and from
`/content/final-object-detection` on Colab without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root - this file is at <root>/src/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Allow an override so Colab/Kaggle can point at a copied dataset without
#: touching the code:  set  OBJDET_DATA_DIR=/content/data
DATA_DIR = Path(os.environ.get("OBJDET_DATA_DIR", PROJECT_ROOT / "data")).resolve()

IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MANIFEST_PATH = DATA_DIR / "manifest.json"
SPLITS_INDEX_PATH = DATA_DIR / "splits.json"
DATA_YAML = DATA_DIR / "data.yaml"

RUNS_DIR = Path(os.environ.get("OBJDET_RUNS_DIR", PROJECT_ROOT / "runs")).resolve()
WEIGHTS_DIR = PROJECT_ROOT / "weights"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

#: Pretrained checkpoints ultralytics / torchvision download on first use.
CHECKPOINT_DIR = PROJECT_ROOT / "weights" / "pretrained"


def images_dir(split: str) -> Path:
    """`data/images/<split>/`."""
    return IMAGES_DIR / split


def labels_dir(split: str) -> Path:
    """`data/labels/<split>/` - YOLO `.txt` files."""
    return LABELS_DIR / split


def ann_path(split: str) -> Path:
    """`data/annotations/instances_<split>.json` - COCO format."""
    return ANNOTATIONS_DIR / f"instances_{split}.json"


def run_dir(model_key: str, tag: str = "") -> Path:
    """Where one training run writes its weights, logs and metrics."""
    name = f"{model_key}_{tag}" if tag else model_key
    return RUNS_DIR / name


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #

#: Canonical class order. Index in this tuple *is* the YOLO class id, and is
#: used verbatim by every model. Never reorder without rebuilding the labels.
CLASSES: tuple[str, ...] = ("apple", "banana", "broccoli", "carrot", "orange")

NUM_CLASSES = len(CLASSES)

#: YOLO / D-FINE convention: 0-indexed, no background class.
IDX_TO_CLASS: dict[int, str] = dict(enumerate(CLASSES))
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in IDX_TO_CLASS.items()}

#: COCO JSON convention: 1-indexed category ids (0 is reserved for background).
#: These are *our* compact ids 1..5, not the original COCO 80-class ids.
CLASS_TO_COCO_ID: dict[str, int] = {name: idx + 1 for idx, name in enumerate(CLASSES)}
COCO_ID_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_COCO_ID.items()}

#: torchvision detection convention: label 0 == background, so our classes
#: occupy 1..5 - which happens to be identical to the COCO ids above, and is
#: why `predict.py` can pass torchvision's labels through unchanged.
TV_NUM_CLASSES = NUM_CLASSES + 1  # +1 background

#: Where each class sits in the original 91-entry COCO label list that
#: torchvision's pretrained detectors were trained on. Used to warm-start the
#: classification head instead of throwing the pretrained weights away.
TORCHVISION_COCO91_INDEX: dict[str, int] = {
    "apple": 53,
    "banana": 52,
    "broccoli": 56,
    "carrot": 57,
    "orange": 55,
}

#: Same idea for the contiguous 80-class COCO ordering used by HuggingFace
#: detection checkpoints (D-FINE, RT-DETR, DETR).
HF_COCO80_INDEX: dict[str, int] = {
    "apple": 47,
    "banana": 46,
    "broccoli": 50,
    "carrot": 51,
    "orange": 49,
}

#: Colour-blind-safe, consistent across every figure and the web app.
CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "apple": (214, 39, 40),      # red
    "banana": (255, 187, 39),    # amber
    "broccoli": (44, 160, 44),   # green
    "carrot": (255, 127, 14),    # orange
    "orange": (148, 103, 189),   # purple - deliberately NOT orange, so the
}                                # 'orange' class never blends into 'carrot'

# --------------------------------------------------------------------------- #
# Splits & determinism
# --------------------------------------------------------------------------- #

SPLITS: tuple[str, ...] = ("train", "val", "test")

RANDOM_SEED = 42

#: train/val come from COCO train2017; test is the untouched COCO val2017
#: subset, so the test set is a genuine held-out distribution.
VAL_FRACTION = 0.15

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
#
# One entry per architecture under comparison. `train_kwargs` holds the
# hyper-parameters that were tuned for an 8 GB RTX 4060 Laptop GPU; the
# training scripts read them from here so the report and the code can never
# drift apart.
#
# On `workers`: 2, not the more usual 4. Windows spawns dataloader workers
# instead of forking them, so each one re-imports torch and re-opens the
# dataset - expensive, and fragile under the I/O load of a 640px pipeline with
# real-time antivirus scanning every JPEG. Measured on this machine: with 4
# workers YOLO's loader died mid-epoch ("DataLoader worker exited unexpectedly")
# and SSDLite ran at 0.201 s/step; with 2 it is stable and *faster*, 0.162
# s/step - the extra workers were contending, not helping. Raise it on Linux.

MODELS: dict[str, dict] = {
    "ssdlite": {
        "display_name": "SSDLite320-MobileNetV3-Large",
        "family": "CNN (one-stage, anchor-based)",
        "framework": "torchvision",
        "checkpoint": "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
        "imgsz": 320,
        "train_kwargs": {
            # The most epochs of the three, because SSDLite is both the cheapest
            # (50 s/epoch measured) and the furthest behind at the start: its
            # warm-started head scores ~0.05 mAP before training, against
            # D-FINE's 0.33. SSD's fixed anchor matching simply needs more
            # passes to adapt.
            "epochs": 120,
            "batch_size": 32,
            "optimizer": "sgd",
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 4e-5,
            "warmup_epochs": 3,
            "scheduler": "cosine",
            "amp": True,
            "workers": 2,
            "ema_decay": 0.999,
            "clip_grad_norm": 10.0,
        },
    },
    "yolo11s": {
        "display_name": "YOLO11s",
        "family": "YOLO (one-stage, anchor-free)",
        "framework": "ultralytics",
        "checkpoint": "yolo11s.pt",
        "imgsz": 640,
        "train_kwargs": {
            # 60 rather than 80: `patience=20` stops the run once val mAP stops
            # improving, so this is an upper bound rather than a fixed cost, and
            # 163 s/epoch makes each extra epoch the most expensive of the three.
            "epochs": 60,
            "batch": 16,
            "optimizer": "auto",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 5e-4,
            "warmup_epochs": 3.0,
            "cos_lr": True,
            "amp": True,
            "workers": 2,
            "patience": 20,
            "close_mosaic": 10,
        },
    },
    # Huan luyen tren Kaggle bang notebooks/04_kaggle_yolo26m_openimages.ipynb,
    # tren Open Images V7 chu khong phai tap con COCO nhu ba model kia. Dang ky o
    # day de duong danh gia nap duoc no; `TRAINERS` co tinh khong co key nay vi
    # `train_yolo.py` gan cung MODEL_KEY = "yolo11s", nen `--model yolo26m` khi
    # huan luyen se bao loi ro rang thay vi am tham train nham model.
    "yolo26m": {
        "display_name": "YOLO26m (Open Images)",
        "family": "YOLO (one-stage, anchor-free, NMS-free)",
        "framework": "ultralytics",
        "checkpoint": "yolo26m.pt",
        "imgsz": 640,
        "train_kwargs": {
            "epochs": 60,
            "batch": 16,
            # Dat thang thay vi "auto": ultralytics bo qua lr0/momentum khi
            # optimizer="auto", nen cong thuc cua YOLO26 khong co tac dung.
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.948,
            "weight_decay": 0.00027,
            "warmup_epochs": 0.99,
            "cos_lr": True,
            "amp": True,
            "workers": 8,
            "patience": 20,
            "close_mosaic": 10,
        },
    },
    "dfine": {
        "display_name": "D-FINE-N",
        "family": "Transformer (DETR-based, end-to-end)",
        "framework": "transformers",
        "checkpoint": "ustc-community/dfine-nano-coco",
        "imgsz": 640,
        "train_kwargs": {
            # The fewest of the three, despite DETRs' reputation for slow
            # convergence: the warm-started head already scores 0.33 mAP before
            # a single gradient step, because the COCO checkpoint was trained on
            # these exact five classes. There is far less to learn than the
            # usual "DETR needs 300 epochs" figure assumes.
            "epochs": 50,
            # 16, not the 8 an 8 GB card suggests. Measured: batch 8 peaks at
            # only 1.6 GB and runs at 32.6 img/s, batch 16 peaks at 3.1 GB and
            # reaches 47.4 img/s - the GPU was simply idling between batches too
            # small to fill it. Also matches D-FINE's own published recipe, so
            # the learning rate below needs no rescaling.
            "batch_size": 16,
            "optimizer": "adamw",
            "lr": 2.5e-4,
            "backbone_lr": 2.5e-5,
            "weight_decay": 1e-4,
            "warmup_epochs": 2,
            "scheduler": "cosine",
            "amp": True,
            "workers": 2,
            "ema_decay": 0.999,
            "clip_grad_norm": 0.1,
        },
    },
}

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

#: Detections below this score never enter the COCO detections file. Low on
#: purpose: mAP integrates the full precision/recall curve, so pruning early
#: only costs recall. The web app uses a much higher threshold.
EVAL_SCORE_THRESHOLD = 0.001

#: COCO's own cap - at most 100 detections per image count towards mAP.
EVAL_MAX_DETECTIONS = 100

#: Confidence used for the qualitative figures and the web application.
DISPLAY_SCORE_THRESHOLD = 0.35


def ensure_dirs() -> None:
    """Create the output directories a run writes into."""
    for d in (RUNS_DIR, WEIGHTS_DIR, FIGURES_DIR, RESULTS_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)
