"""Turn each model's native output into one common COCO detections list.

This is the adapter layer that makes `coco_eval` possible. Each framework
returns something different:

    torchvision   list[dict] with xyxy boxes and labels already in 1..5
    ultralytics   Results objects with xyxy boxes and 0-indexed classes
    transformers  normalised cxcywh, post-processed back to xyxy, 0-indexed

and each has to end up as `{image_id, category_id, bbox: [x, y, w, h], score}`
with `category_id` in our 1..5 COCO ids and `bbox` in absolute pixels of the
*original* image.

The subtle part is coordinates. Every model sees a resized image (320x320 or
640x640) and predicts in that frame; if the mapping back to original pixels is
off, mAP collapses in a way that looks like a training failure. Each function
below therefore uses the framework's own inverse transform rather than
rescaling by hand.
"""

from __future__ import annotations

import time

import torch

from src.config import EVAL_MAX_DETECTIONS, EVAL_SCORE_THRESHOLD


def _to_coco_records(
    image_id: int,
    boxes_xyxy,
    scores,
    category_ids,
    score_threshold: float,
    max_detections: int,
    canvas: tuple[int, int] | None = None,
) -> list[dict]:
    """Shared tail: threshold, cap, clip, convert xyxy -> xywh, emit plain floats.

    `canvas` is the original image's (height, width). Clipping to it matters
    because a box that overhangs the border has an inflated area, and COCO's
    small/medium/large AP buckets are decided by area - an unclipped detection
    can be scored in the wrong size bucket.
    """
    records: list[dict] = []

    order = scores.argsort(descending=True)[:max_detections]
    for i in order.tolist():
        score = float(scores[i])
        if score < score_threshold:
            continue
        x0, y0, x1, y1 = (float(v) for v in boxes_xyxy[i])
        if canvas is not None:
            img_h, img_w = canvas
            x0, x1 = max(0.0, min(x0, img_w)), max(0.0, min(x1, img_w))
            y0, y1 = max(0.0, min(y0, img_h)), max(0.0, min(y1, img_h))
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0:
            continue
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_ids[i]),
                "bbox": [round(x0, 2), round(y0, 2), round(width, 2), round(height, 2)],
                "score": round(score, 5),
            }
        )
    return records


# --------------------------------------------------------------------------- #
# SSDLite (torchvision)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict_ssdlite(
    model,
    split: str = "test",
    device: torch.device | None = None,
    batch_size: int = 16,
    workers: int = 4,
    score_threshold: float = EVAL_SCORE_THRESHOLD,
    limit_batches: int | None = None,
) -> tuple[list[dict], dict]:
    """Run SSDLite over a split and collect COCO detections.

    torchvision's detector does its own letterbox-free resize inside
    `GeneralizedRCNNTransform` and maps predictions back to input-image
    coordinates before returning, so the boxes here are already in original
    pixels. Its labels are 1..5, which is exactly our COCO category id range -
    no remapping needed.
    """
    from torch.utils.data import DataLoader

    from src.data.coco_dataset import TorchvisionDetectionDataset, torchvision_collate

    device = device or next(model.parameters()).device
    dataset = TorchvisionDetectionDataset(split, transforms=None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=torchvision_collate,
        pin_memory=device.type == "cuda",
    )

    sizes = {
        r["image_id"]: (r["height"], r["width"]) for r in dataset.data.records
    }

    model.eval()
    detections: list[dict] = []
    n_images = 0
    started = time.perf_counter()

    for step, (images, targets) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            detections += _to_coco_records(
                image_id=target["image_id"],
                boxes_xyxy=output["boxes"].cpu(),
                scores=output["scores"].cpu(),
                category_ids=output["labels"].cpu().tolist(),
                score_threshold=score_threshold,
                max_detections=EVAL_MAX_DETECTIONS,
                canvas=sizes.get(target["image_id"]),
            )
        n_images += len(images)

    return detections, {"n_images": n_images, "wall_seconds": time.perf_counter() - started}


# --------------------------------------------------------------------------- #
# YOLO11s (ultralytics)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict_yolo(
    model,
    split: str = "test",
    imgsz: int = 640,
    device: str | int = 0,
    batch_size: int = 16,
    score_threshold: float = EVAL_SCORE_THRESHOLD,
    iou_threshold: float = 0.7,
    class_map: dict[int, int] | None = None,
) -> tuple[list[dict], dict]:
    """Run YOLO11s over a split's image files.

    Ultralytics is handed the *file paths* rather than our tensors so that its
    own letterbox preprocessing runs exactly as it does at deployment time -
    the point of the comparison is each model as it would actually be used.
    Image ids come from the COCO JSON, matched on file name.

    `class_map` exists for the un-fine-tuned COCO checkpoint, whose head still
    has all 80 classes: it maps the COCO class index to our category id and
    doubles as the filter, so only the five classes we care about are kept.
    Fine-tuned checkpoints leave it `None` and take the 0-indexed +1 path.
    """
    from src.data.coco_dataset import CocoRecords

    records = CocoRecords(split)
    paths = [str(records.path(i)) for i in range(len(records))]
    image_ids = [records.records[i]["image_id"] for i in range(len(records))]

    detections: list[dict] = []
    started = time.perf_counter()

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        ids = image_ids[start : start + batch_size]
        results = model.predict(
            chunk,
            imgsz=imgsz,
            device=device,
            conf=score_threshold,
            iou=iou_threshold,
            max_det=EVAL_MAX_DETECTIONS,
            verbose=False,
            **({"classes": sorted(class_map)} if class_map else {}),
        )
        for result, image_id in zip(results, ids):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            if class_map:
                category_ids = [class_map[int(c)] for c in boxes.cls.cpu().tolist()]
            else:
                # +1: ultralytics classes are 0-indexed, COCO category ids are 1-indexed.
                category_ids = [int(c) + 1 for c in boxes.cls.cpu().tolist()]
            detections += _to_coco_records(
                image_id=image_id,
                boxes_xyxy=boxes.xyxy.cpu(),
                scores=boxes.conf.cpu(),
                category_ids=category_ids,
                score_threshold=score_threshold,
                max_detections=EVAL_MAX_DETECTIONS,
                canvas=result.orig_shape,  # (height, width)
            )

    return detections, {"n_images": len(paths), "wall_seconds": time.perf_counter() - started}


# --------------------------------------------------------------------------- #
# D-FINE (transformers)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict_dfine(
    model,
    processor,
    split: str = "test",
    device: torch.device | None = None,
    batch_size: int = 8,
    workers: int = 4,
    score_threshold: float = EVAL_SCORE_THRESHOLD,
    limit_batches: int | None = None,
) -> tuple[list[dict], dict]:
    """Run D-FINE over a split and collect COCO detections.

    D-FINE emits a fixed 300 queries per image with no NMS - the bipartite
    training objective is supposed to make duplicate suppression unnecessary.
    We simply keep the highest-scoring 100, which is COCO's own cap.
    """
    from torch.utils.data import DataLoader

    from src.data.coco_dataset import DFineDetectionDataset, build_dfine_collate

    device = device or next(model.parameters()).device
    dataset = DFineDetectionDataset(split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=build_dfine_collate(processor),
        pin_memory=device.type == "cuda",
    )

    model.eval()
    detections: list[dict] = []
    n_images = 0
    started = time.perf_counter()

    for step, batch in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        outputs = model(pixel_values=pixel_values)

        # target_sizes drives the inverse of the 640x640 resize, so boxes come
        # back in the original image's pixel coordinates.
        target_sizes = torch.tensor(batch["orig_sizes"], device=device)
        processed = processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )

        for result, image_id, orig_size in zip(
            processed, batch["image_ids"], batch["orig_sizes"]
        ):
            if result["scores"].numel() == 0:
                continue
            category_ids = [int(c) + 1 for c in result["labels"].cpu().tolist()]
            detections += _to_coco_records(
                image_id=image_id,
                boxes_xyxy=result["boxes"].cpu(),
                scores=result["scores"].cpu(),
                category_ids=category_ids,
                score_threshold=score_threshold,
                max_detections=EVAL_MAX_DETECTIONS,
                canvas=orig_size,  # (height, width)
            )
        n_images += pixel_values.shape[0]

    return detections, {"n_images": n_images, "wall_seconds": time.perf_counter() - started}
