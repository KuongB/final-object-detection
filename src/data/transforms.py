"""Augmentation pipelines for the two hand-trained models.

Only SSDLite needs a real pipeline here - ultralytics owns YOLO11s's
augmentation (mosaic, HSV, scale, mixup) through its own hyper-parameters, and
D-FINE's lighter jitter lives next to its dataset in `coco_dataset.py` because
it has to run on numpy arrays before the HuggingFace processor sees them.

The SSD recipe is the one from the original SSD paper, which torchvision's
reference training still uses: photometric distortion, then "zoom out" (paste
the image into a larger canvas, manufacturing small objects), then an
IoU-constrained random crop, then a flip. The zoom-out/crop pair is what makes
SSD's fixed anchor grid scale-invariant, and it is worth far more on a 5.8k-image
fine-tune than any amount of extra epochs.
"""

from __future__ import annotations

import torch


def build_ssd_transforms(train: bool, zoom_out_prob: float = 0.35):
    """Return a `transforms.v2` pipeline for the SSDLite dataset.

    Boxes ride along automatically: v2 transforms dispatch on `tv_tensors`
    types, so `BoundingBoxes` are cropped, flipped and shifted in lockstep with
    the image.

    Note there is no `Resize` and no `Normalize` - torchvision's detection
    models carry their own `GeneralizedRCNNTransform` that resizes to 320x320
    and applies the ImageNet statistics internally.
    """
    from torchvision.transforms import v2

    if not train:
        return v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    # Zoom-out pads with the ImageNet mean so the manufactured border matches
    # what the model's own normalisation will map to zero.
    fill = [123.0, 117.0, 104.0]

    return v2.Compose(
        [
            v2.RandomPhotometricDistort(p=0.5),
            v2.RandomZoomOut(fill=fill, side_range=(1.0, 3.0), p=zoom_out_prob),
            v2.RandomIoUCrop(),
            v2.RandomHorizontalFlip(p=0.5),
            # After a crop, boxes can be partially or entirely outside the
            # canvas; this clips them and drops the degenerate leftovers.
            v2.SanitizeBoundingBoxes(min_size=2.0),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
