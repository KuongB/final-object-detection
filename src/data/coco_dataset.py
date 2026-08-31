"""PyTorch datasets over `data/annotations/instances_<split>.json`.

Ultralytics builds its own loader straight from `data.yaml`, so only the two
hand-trained models need this: SSDLite (torchvision conventions) and D-FINE
(HuggingFace conventions). Both read the *same* COCO file, so any difference
in their scores comes from the architecture rather than from the data.

Crowd regions are dropped from every training target. They stay in the COCO
JSON because `COCOeval` uses them correctly at scoring time - it excludes a
detection that lands on a crowd region instead of counting it as a false
positive - but as a *training* target a crowd box is noise: it marks "a pile
of oranges is somewhere in this region", which no per-object detector can
reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import CLASS_TO_IDX, COCO_ID_TO_CLASS, ann_path, images_dir


class CocoRecords:
    """Loads one split's COCO JSON into a flat, index-addressable list.

    Kept separate from the `Dataset` classes so both model families share one
    parser - and so the EDA / evaluation code can reuse it without dragging in
    a transform pipeline.
    """

    def __init__(self, split: str, include_crowd: bool = False) -> None:
        self.split = split
        self.root = images_dir(split)
        raw = json.loads(ann_path(split).read_text(encoding="utf-8"))

        by_image: dict[int, list[dict]] = {}
        for ann in raw["annotations"]:
            if ann["iscrowd"] and not include_crowd:
                continue
            by_image.setdefault(ann["image_id"], []).append(ann)

        self.records: list[dict] = []
        for img in raw["images"]:
            self.records.append(
                {
                    "image_id": img["id"],
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                    "annotations": by_image.get(img["id"], []),
                }
            )

        self.categories = {c["id"]: c["name"] for c in raw["categories"]}

    def __len__(self) -> int:
        return len(self.records)

    def path(self, idx: int) -> Path:
        return self.root / self.records[idx]["file_name"]

    def image(self, idx: int) -> Image.Image:
        # `.convert("RGB")` is not optional: COCO ships a handful of greyscale
        # and CMYK JPEGs that would otherwise arrive with 1 or 4 channels.
        with Image.open(self.path(idx)) as im:
            return im.convert("RGB")

    def boxes_xyxy(self, idx: int) -> tuple[list[list[float]], list[int]]:
        """Absolute xyxy boxes plus 0-indexed class ids."""
        boxes, labels = [], []
        for ann in self.records[idx]["annotations"]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(CLASS_TO_IDX[COCO_ID_TO_CLASS[ann["category_id"]]])
        return boxes, labels


# --------------------------------------------------------------------------- #
# torchvision (SSDLite)
# --------------------------------------------------------------------------- #


class TorchvisionDetectionDataset(Dataset):
    """Yields `(image, target)` in the layout torchvision detectors expect.

    * image  - float tensor in [0, 1]; the model's own `GeneralizedRCNNTransform`
      does the resize to 320x320 and the mean/std normalisation, so doing either
      here would apply it twice.
    * target - `boxes` as absolute xyxy, `labels` in 1..5 because torchvision
      reserves label 0 for background.
    """

    def __init__(self, split: str, transforms=None) -> None:
        self.data = CocoRecords(split)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        from torchvision import tv_tensors
        from torchvision.transforms.v2 import functional as F

        image = self.data.image(idx)
        boxes, labels = self.data.boxes_xyxy(idx)
        record = self.data.records[idx]

        img = tv_tensors.Image(F.pil_to_tensor(image))
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                format="XYXY",
                canvas_size=(record["height"], record["width"]),
            ),
            # +1: shift our 0..4 onto torchvision's 1..5 (0 == background).
            "labels": torch.as_tensor(labels, dtype=torch.int64) + 1,
            "image_id": record["image_id"],
        }

        if self.transforms is None:
            return F.to_dtype(img, torch.float32, scale=True), target

        # RandomIoUCrop can legitimately crop every box out of frame. An empty
        # target makes SSD's hard-negative mining divide by zero, so retry the
        # random pipeline a few times before falling back to the plain image.
        for _ in range(4):
            out_img, out_target = self.transforms(img, target)
            if out_target["boxes"].numel():
                return out_img, out_target
        return F.to_dtype(img, torch.float32, scale=True), target


def torchvision_collate(batch):
    """Detectors take a *list* of images - they are not a fixed-size stack."""
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- #
# HuggingFace (D-FINE)
# --------------------------------------------------------------------------- #


class DFineDetectionDataset(Dataset):
    """Yields raw images plus COCO-style annotations for `RTDetrImageProcessor`.

    Augmentation happens here, on absolute pixel boxes; the processor is then
    handed an already-augmented image and does only the deterministic part -
    resize to 640x640, rescale to [0, 1], and the conversion of boxes to the
    normalised cxcywh that the DETR-style loss expects.
    """

    def __init__(self, split: str, augment: bool = False) -> None:
        self.data = CocoRecords(split)
        self.augment = augment
        self._aug = DFineAugment() if augment else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        import numpy as np

        image = self.data.image(idx)
        boxes, labels = self.data.boxes_xyxy(idx)
        record = self.data.records[idx]

        arr = np.asarray(image)
        if self._aug is not None:
            arr, boxes, labels = self._aug(arr, boxes, labels)

        height, width = arr.shape[:2]
        annotations = [
            {
                "image_id": record["image_id"],
                "category_id": int(c),
                "bbox": [x0, y0, x1 - x0, y1 - y0],  # processor wants xywh
                "area": float((x1 - x0) * (y1 - y0)),
                "iscrowd": 0,
            }
            for (x0, y0, x1, y1), c in zip(boxes, labels)
        ]
        return {
            "image": arr,
            "target": {"image_id": record["image_id"], "annotations": annotations},
            "size": (height, width),
        }


class DFineAugment:
    """Horizontal flip + photometric jitter, applied to image and boxes together.

    Deliberately conservative. DETR-family models converge slowly and are
    sensitive to aggressive geometric augmentation on a small fine-tuning set;
    scale jitter is already covered by the processor's fixed 640x640 resize of
    variously-shaped source images.

    Written as a class rather than a closure because Windows spawns dataloader
    workers instead of forking them, which means the dataset - and everything
    it holds - has to survive `pickle`. A nested function does not.
    """

    def __init__(self) -> None:
        # Created lazily in `_get_rng`: a Generator built here would be pickled
        # *with its state*, so every worker would inherit the same stream and
        # apply an identical sequence of "random" augmentations.
        self._rng = None

    def _get_rng(self):
        import numpy as np

        if self._rng is None:
            import torch.utils.data as torch_data

            info = torch_data.get_worker_info()
            seed = info.seed if info is not None else None
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __call__(self, arr, boxes, labels):
        import numpy as np

        rng = self._get_rng()
        height, width = arr.shape[:2]

        if rng.random() < 0.5:
            arr = arr[:, ::-1].copy()
            boxes = [[width - x1, y0, width - x0, y1] for x0, y0, x1, y1 in boxes]

        if rng.random() < 0.5:
            # Brightness / contrast jitter in float space, then clip back to
            # uint8 so the processor's rescale still sees a valid image.
            brightness = rng.uniform(0.8, 1.2)
            contrast = rng.uniform(0.8, 1.2)
            f = arr.astype(np.float32)
            f = (f - f.mean()) * contrast + f.mean() * brightness
            arr = np.clip(f, 0, 255).astype(np.uint8)

        return arr, boxes, labels


class DFineCollate:
    """Batch collator that runs the HF processor over a list of samples.

    A class, not a closure, for the same reason `DFineAugment` is: Windows
    spawns dataloader workers, and `collate_fn` is pickled along with the
    dataset. A nested function raises
    `AttributeError: Can't pickle local object` the moment `num_workers > 0`.
    """

    def __init__(self, image_processor) -> None:
        self.image_processor = image_processor

    def __call__(self, samples):
        encoding = self.image_processor(
            images=[s["image"] for s in samples],
            annotations=[s["target"] for s in samples],
            return_tensors="pt",
        )
        return {
            "pixel_values": encoding["pixel_values"],
            "labels": encoding["labels"],
            # Kept out of the model call - evaluation needs them to map
            # normalised predictions back onto original image coordinates.
            "image_ids": [s["target"]["image_id"] for s in samples],
            "orig_sizes": [s["size"] for s in samples],
        }


def build_dfine_collate(image_processor) -> DFineCollate:
    """Kept as a function so call sites read the same as before."""
    return DFineCollate(image_processor)
