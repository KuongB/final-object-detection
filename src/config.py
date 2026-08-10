"""Central configuration: paths, class definitions, split parameters.

This module is the single source of truth for the project. Every script and
notebook imports from here so that paths and class-index mappings can never
drift apart between the data pipeline, the three training pipelines, the
evaluation code and the web application.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

# One shared image tree serves BOTH annotation formats, so the three models are
# guaranteed to train and evaluate on byte-identical images:
#
#   data/images/{train,val,test}/*.jpg          <- shared images
#   data/labels/{train,val,test}/*.txt          <- YOLO  (ultralytics finds these
#                                                  by swapping /images/ -> /labels/)
#   data/annotations/instances_{split}.json     <- COCO JSON (torchvision SSD, COCOeval)
#
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
ANN_DIR = DATA_DIR / "annotations"

DATA_YAML = DATA_DIR / "data.yaml"  # ultralytics dataset config
MANIFEST_PATH = DATA_DIR / "manifest.json"  # canonical dump from FiftyOne


def images_dir(split: str) -> Path:
    return IMAGES_DIR / split


def labels_dir(split: str) -> Path:
    return LABELS_DIR / split


def ann_path(split: str) -> Path:
    return ANN_DIR / f"instances_{split}.json"

CONFIGS_DIR = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RUNS_DIR = PROJECT_ROOT / "runs"  # raw training outputs (per-model)
WEIGHTS_DIR = PROJECT_ROOT / "weights"  # curated best checkpoints
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

WEBAPP_DIR = PROJECT_ROOT / "webapp"

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------
# Kept in alphabetical order so the index mapping is deterministic and
# reproducible regardless of the order FiftyOne happens to return.
CLASSES = ["apple", "banana", "broccoli", "carrot", "orange"]
NUM_CLASSES = len(CLASSES)

# YOLO / ultralytics convention: contiguous, 0-based.
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

# COCO JSON convention: category_id is 1-based. torchvision detection models
# reserve index 0 for background, so `category_id == model output label`.
CLASS_TO_COCO_ID = {name: i + 1 for i, name in enumerate(CLASSES)}
COCO_ID_TO_CLASS = {i + 1: name for i, name in enumerate(CLASSES)}

# torchvision detection heads are built with num_classes = len(classes) + 1
NUM_CLASSES_WITH_BACKGROUND = NUM_CLASSES + 1

# Vietnamese display names, used by the web application UI.
CLASS_DISPLAY_VI = {
    "apple": "Táo",
    "banana": "Chuối",
    "broccoli": "Súp lơ xanh",
    "carrot": "Cà rốt",
    "orange": "Cam",
}

# Stable per-class colours, used everywhere: EDA plots, bbox overlays, web app.
#
# NOTE - do not "fix" these to look like the fruit. The obvious mapping (carrot
# = orange, broccoli = green, apple = red) was measured and rejected: orange
# #ff7f0e against green #2ca02c is ΔE 0.7 under protanopia, i.e. the two classes
# are the same colour for a red-green colourblind reader, and roughly 1 in 12 men
# are. These five hues were picked from a validated categorical ramp and checked
# with the palette validator over ALL pairs (not just neighbouring bars, because
# all five classes appear at once on a bbox overlay):
#
#   worst pair, normal vision : ΔE 16.3  (floor 15)  PASS
#   worst pair, CVD           : ΔE  6.9  (target 8)  WARN -> allowed because
#                               every mark also carries a text label
#
# Two of the five (banana, broccoli) sit below 3:1 contrast on white, so charts
# using them must show values or labels rather than relying on the fill alone.
CLASS_COLORS_HEX = {
    "apple": "#e34948",  # red
    "banana": "#eda100",  # yellow
    "broccoli": "#1baf7a",  # aqua
    "carrot": "#2a78d6",  # blue
    "orange": "#4a3aa7",  # violet
}
CLASS_COLORS_RGB = {
    name: tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))
    for name, h in CLASS_COLORS_HEX.items()
}
CLASS_COLORS_BGR = {name: rgb[::-1] for name, rgb in CLASS_COLORS_RGB.items()}

# Chart chrome, so every figure in the report looks like one system.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"  # axis tick labels
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#ffffff"

# Single-hue ramp for magnitude (heatmaps, density) - never a rainbow.
SEQUENTIAL_HUE = "#2a78d6"
SEQUENTIAL_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------
# COCO source splits we pull from.
#   coco-2017/train      -> split internally into our train + val
#   coco-2017/validation -> used untouched as our held-out test set
SPLITS = ("train", "val", "test")
VAL_FRACTION = 0.15  # fraction of the coco-train subset held out for val
RANDOM_SEED = 42

# Image extensions we accept when scanning directories.
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
# torchvision's SSD normalises INSIDE its own forward() using these values
# (GeneralizedRCNNTransform with image_std = 1/255), so the data pipeline must
# hand it raw [0, 1] tensors and must NOT normalise them itself. This constant
# is here only so the zoom-out padding colour matches the mean SSD subtracts.
# Ultralytics models likewise normalise internally and need nothing from us.
SSD_PIXEL_MEAN = [0.48235, 0.45882, 0.40784]

# COCO class ids as each pretrained checkpoint numbers them. Needed to evaluate
# the OFF-THE-SHELF models on our data, before any fine-tuning: the five classes
# are already COCO classes, so a zero-shot baseline is available and shows how
# much fine-tuning actually contributes.
#   torchvision SSD -> 91-class COCO ids
#   ultralytics     -> 80-class contiguous ids
# Both are verified against the checkpoint metadata at runtime rather than
# trusted from here.
TORCHVISION_COCO91_IDS = {
    "apple": 53, "banana": 52, "broccoli": 56, "carrot": 57, "orange": 55,
}
ULTRALYTICS_COCO80_IDS = {
    "apple": 47, "banana": 46, "broccoli": 50, "carrot": 51, "orange": 49,
}

# Shared across all three models so the comparison is like-for-like.
#
# 15 rather than 40 is a deliberate compute-budget decision, not a claim that
# the models have converged. At the point the 40-epoch attempt was stopped,
# SSD300 was still improving on every epoch (val mAP 0.170 and rising at epoch
# 13), so it in particular is under-trained here. The report must say so: SSD is
# the model that suffers most, because it is the only one whose classification
# head is randomly initialised on top of a backbone that transfers less
# directly than the YOLO/RT-DETR necks do. Any conclusion of the form "the CNN
# approach is weaker" therefore has to be qualified by this budget.
EPOCHS = 15
# Patience must scale with the budget: 10 on a 15-epoch run could never fire
# before epoch 11 and would effectively disable early stopping.
PATIENCE = 5

# Per-model input resolution. These are each model's native/pretrained setting;
# forcing a common size would handicap whichever model was not designed for it.
# The difference must be stated whenever FPS numbers are compared.
INPUT_SIZES = {"ssd300_vgg16": 300, "yolov8m": 640, "rtdetr-l": 640}

# Exactly three models, one per architecture family, as the assignment requires:
#   CNN one-stage  SSD300-VGG16   24.3M params
#   YOLO           YOLOv8m        25.9M params
#   Transformer    RT-DETR-l      32.8M params
#
# YOLOv8m rather than the lighter v8s so that all three sit in the same 24-33M
# band. With v8s (11.2M) a loss for YOLO could always be blamed on capacity
# rather than on architecture, which would weaken the comparison the report is
# built on.
MODEL_NAMES = ("ssd300_vgg16", "yolov8m", "rtdetr-l")

# Batch sizes that fit the 8 GB RTX 4060 Laptop at the sizes above. They differ
# per model out of necessity, so they are recorded in every run config and must
# be quoted alongside any throughput number.
#
# RT-DETR peak VRAM measured on this GPU, one full smoke run per setting:
#
#   batch 4 -> 3869 MiB (47%)   original setting, GPU only 55% utilised
#   batch 6 -> 5683 MiB (69%)   chosen
#   batch 8 -> 7233 MiB (88%)   runs, but leaves only 955 MiB spare
#
# Batch 8 fits and does not OOM in a short run, but 955 MiB is too thin to hold
# for two or three hours: under Windows WDDM the display driver, a
# hardware-accelerated browser or the editor can claim several hundred MiB at
# any moment, and allocator fragmentation grows over a long run. An OOM in hour
# two costs the whole run, so batch 6 keeps a 2.5 GB margin instead.
#
# Ultralytics holds the effective batch at nbs=64 by adjusting gradient
# accumulation, so the learning rate does not need rescaling with this.
BATCH_SIZES = {"ssd300_vgg16": 16, "yolov8m": 16, "rtdetr-l": 6}

# --- ultralytics schedule parameters that are measured IN EPOCHS -------------
# These are the trap of shortening the run: both are counted in epochs, so
# leaving the library defaults while cutting 40 epochs to 15 silently changes
# the training recipe.
#
#   close_mosaic counts BACKWARDS from the end. At the default 10 with 15
#   epochs, mosaic would be active for only the first 5 epochs (33%) instead of
#   the 30 of 40 (75%) the original plan assumed. Mosaic is ultralytics'
#   strongest augmentation and it applies to YOLOv8/RT-DETR but not to SSD, so
#   getting this wrong skews the very comparison the project rests on.
CLOSE_MOSAIC = 3  # -> mosaic on for 12 of 15 epochs (80%)
#   warmup_epochs at the default 3.0 would consume 20% of a 15-epoch budget,
#   against 7.5% of a 40-epoch one.
WARMUP_EPOCHS = 1.5


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes into."""
    paths = [ANN_DIR, RUNS_DIR, WEIGHTS_DIR, FIGURES_DIR, RESULTS_DIR]
    for split in SPLITS:
        paths += [images_dir(split), labels_dir(split)]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"CLASSES      = {CLASSES}")
    print(f"CLASS_TO_IDX = {CLASS_TO_IDX}")
    print(f"CLASS_TO_COCO_ID = {CLASS_TO_COCO_ID}")
    ensure_dirs()
    print("All output directories created.")
