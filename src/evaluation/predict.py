"""Run any of the three models over a split and emit COCO-format detections.

One output format for all three, so `src/evaluation/coco_eval.py` scores them
with the same COCOeval call. Anything model-specific - class-id conventions,
input normalisation, how boxes are mapped back to original pixels - is dealt
with here and nowhere else.

Also provides `measure_fps`, which is the only honest way to compare inference
speed: the same images, the same machine, the same warm-up, timed end to end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.config import (
    CLASS_TO_COCO_ID,
    CLASSES,
    TORCHVISION_COCO91_IDS,
    ann_path,
    images_dir,
)


# ---------------------------------------------------------------------------
# split listing
# ---------------------------------------------------------------------------
def list_split_images(split: str) -> list[dict]:
    """[{image_id, file_name, path, width, height}, ...] in COCO json order."""
    coco = json.loads(ann_path(split).read_text(encoding="utf-8"))
    directory = images_dir(split)
    return [
        {
            "image_id": img["id"],
            "file_name": img["file_name"],
            "path": str(directory / img["file_name"]),
            "width": img["width"],
            "height": img["height"],
        }
        for img in coco["images"]
    ]


# ---------------------------------------------------------------------------
# class-id mapping
# ---------------------------------------------------------------------------
def ultralytics_class_map(names: dict) -> dict[int, int]:
    """model class index -> our COCO category_id (1..5).

    Derived from the checkpoint's own `names`, never hard-coded: a fine-tuned
    model has 5 classes in our alphabetical order, an off-the-shelf one has the
    80 COCO classes, and guessing which is which is how silent label scrambles
    happen.
    """
    lookup = {}
    for idx, name in names.items():
        if name in CLASS_TO_COCO_ID:
            lookup[int(idx)] = CLASS_TO_COCO_ID[name]
    missing = set(CLASSES) - {n for n in names.values() if n in CLASS_TO_COCO_ID}
    if missing:
        raise ValueError(f"checkpoint does not contain classes {sorted(missing)}")
    return lookup


def torchvision_pretrained_class_map(weights=None) -> dict[int, int]:
    """91-class COCO label -> our category_id, for the off-the-shelf SSD."""
    if weights is not None:
        cats = weights.meta["categories"]
        return {cats.index(name): CLASS_TO_COCO_ID[name] for name in CLASSES}
    return {v: CLASS_TO_COCO_ID[k] for k, v in TORCHVISION_COCO91_IDS.items()}


def identity_class_map() -> dict[int, int]:
    """Our fine-tuned SSD already predicts labels 1..5 == category_id."""
    return {v: v for v in CLASS_TO_COCO_ID.values()}


# ---------------------------------------------------------------------------
# ultralytics
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_ultralytics(
    model,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    device: str = "0",
    batch: int = 8,
    class_map: dict[int, int] | None = None,
) -> list[dict]:
    """YOLOv8 / RT-DETR -> COCO detections in original-image pixels.

    `conf` is 0.001, far below any sensible deployment threshold, because
    COCOeval integrates precision across the whole recall range: cutting
    low-confidence boxes truncates the PR curve and understates AP.
    """
    images = list_split_images(split)
    if class_map is None:
        class_map = ultralytics_class_map(model.names)

    detections: list[dict] = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        results = model.predict(
            [c["path"] for c in chunk],
            imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
            device=device, verbose=False,
        )
        for meta, result in zip(chunk, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            # ultralytics already returns xyxy in ORIGINAL image coordinates
            xyxy = boxes.xyxy.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            for (x0, y0, x1, y1), score, cls in zip(xyxy, scores, classes):
                category_id = class_map.get(int(cls))
                if category_id is None:
                    continue  # a COCO class we do not model
                detections.append({
                    "image_id": meta["image_id"],
                    "category_id": category_id,
                    "bbox": [round(x0, 2), round(y0, 2),
                             round(x1 - x0, 2), round(y1 - y0, 2)],
                    "score": round(float(score), 5),
                })
    return detections


# ---------------------------------------------------------------------------
# torchvision SSD
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_ssd(
    model,
    split: str = "test",
    device: torch.device | str = "cuda",
    batch: int = 8,
    class_map: dict[int, int] | None = None,
    amp: bool = True,
    normalize_imagenet: bool = False,
) -> list[dict]:
    """torchvision SSD -> COCO detections in original-image pixels.

    Images go in as raw float [0, 1] at their ORIGINAL resolution. The model's
    own GeneralizedRCNNTransform resizes to 300x300, normalises, and maps the
    predicted boxes back to the original size on the way out - so there is no
    manual rescaling here to get wrong.

    `normalize_imagenet` exists only to reproduce the double-normalisation bug
    for the A/B check in `scripts/20_baseline_pretrained.py`. Leave it False.
    """
    from PIL import Image
    from torchvision.transforms.v2 import functional as F

    device = torch.device(device)
    model = model.to(device).eval()
    images = list_split_images(split)
    if class_map is None:
        class_map = identity_class_map()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    detections: list[dict] = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        tensors = []
        for meta in chunk:
            with Image.open(meta["path"]) as img:
                tensor = F.to_image(img.convert("RGB"))
            tensor = F.to_dtype(tensor, torch.float32, scale=True).to(device)
            if normalize_imagenet:
                tensor = (tensor - mean) / std
            tensors.append(tensor)

        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            outputs = model(tensors)

        for meta, output in zip(chunk, outputs):
            boxes = output["boxes"].float().cpu().tolist()
            scores = output["scores"].float().cpu().tolist()
            labels = output["labels"].cpu().tolist()
            for (x0, y0, x1, y1), score, label in zip(boxes, scores, labels):
                category_id = class_map.get(int(label))
                if category_id is None:
                    continue
                detections.append({
                    "image_id": meta["image_id"],
                    "category_id": category_id,
                    "bbox": [round(x0, 2), round(y0, 2),
                             round(x1 - x0, 2), round(y1 - y0, 2)],
                    "score": round(float(score), 5),
                })
    return detections


# ---------------------------------------------------------------------------
# speed
# ---------------------------------------------------------------------------
def measure_fps(
    predict_one,
    paths: list[str],
    warmup: int = 10,
    repeats: int = 1,
    device: str = "cuda",
) -> dict:
    """Time single-image inference, the way the web application will run it.

    Batch-1 rather than batched throughput, because the deployed app processes
    one upload at a time - batched numbers would flatter the models with heavier
    per-image cost. CUDA is synchronised around the timer, otherwise the kernels
    are still queued when the clock stops and the result is meaningless.
    """
    sync = (lambda: torch.cuda.synchronize()) if (
        device == "cuda" and torch.cuda.is_available()
    ) else (lambda: None)

    for path in paths[:warmup]:
        predict_one(path)
    sync()

    times: list[float] = []
    for _ in range(repeats):
        for path in paths:
            sync()
            started = time.perf_counter()
            predict_one(path)
            sync()
            times.append(time.perf_counter() - started)

    times_sorted = sorted(times)
    n = len(times_sorted)
    return {
        "n_images": n,
        "mean_ms": round(1000 * sum(times) / n, 2),
        "median_ms": round(1000 * times_sorted[n // 2], 2),
        "p90_ms": round(1000 * times_sorted[int(0.9 * n)], 2),
        "fps_mean": round(n / sum(times), 1),
        "fps_median": round(1.0 / times_sorted[n // 2], 1),
    }


__all__ = [
    "identity_class_map",
    "list_split_images",
    "measure_fps",
    "predict_ssd",
    "predict_ultralytics",
    "torchvision_pretrained_class_map",
    "ultralytics_class_map",
]
