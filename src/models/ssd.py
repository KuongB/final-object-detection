"""SSD300-VGG16, adapted from 91-class COCO to our 5 classes."""

from __future__ import annotations

import torch
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

from src.config import NUM_CLASSES_WITH_BACKGROUND


def build_ssd300(
    num_classes: int = NUM_CLASSES_WITH_BACKGROUND,
    pretrained: bool = True,
    score_thresh: float = 0.01,
    nms_thresh: float = 0.45,
    detections_per_img: int = 100,
    warmstart_head: bool = True,
) -> torch.nn.Module:
    """SSD300 with a fresh classification head sized for `num_classes`.

    `num_classes` counts background: 5 fruit/vegetable classes -> 6.

    Only the classification head is replaced. The box-regression head is
    class-agnostic in SSD, so its COCO pretraining transfers directly and is
    worth keeping - re-initialising it would throw away a good box prior for no
    reason.

    Replacing the head rather than slicing the 91-class one keeps the comparison
    fair: ultralytics also discards the pretrained head when `nc` changes, so all
    three models start from a randomly initialised classifier over a pretrained
    backbone.

    `score_thresh` is deliberately low (0.01, torchvision's default is 0.01 as
    well): COCOeval integrates precision over the full recall range, so
    discarding low-confidence boxes early would truncate the PR curve and
    understate AP. The web application applies its own, much higher, threshold.

    `detections_per_img` is 100 rather than torchvision's 200 because COCOeval's
    primary metric caps at maxDets=100 - boxes 101-200 are scored by nothing and
    only inflate the detection list, which is what exhausted host RAM during
    evaluation on this 16 GB machine.
    """
    weights = SSD300_VGG16_Weights.COCO_V1 if pretrained else None
    model = ssd300_vgg16(
        weights=weights,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
    )

    old_head = model.head.classification_head
    in_channels = [layer.in_channels for layer in old_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    new_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    # Warm-start: all five classes are already COCO classes, so instead of
    # throwing the pretrained classifier away and relearning it from 5.8k
    # images, copy the rows that belong to our classes (plus background).
    # Verified in scripts/21_verify_warmstart.py: the transplanted head scores
    # 0.1315 mAP zero-shot against the original head's 0.1314.
    if warmstart_head and pretrained:
        from src.models.head_transfer import warmstart_ssd_head

        warmstart_ssd_head(new_head, old_head, weights.meta["categories"])

    model.head.classification_head = new_head
    return model


__all__ = ["build_ssd300"]
