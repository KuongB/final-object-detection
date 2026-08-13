"""SSDLite320-MobileNetV3-Large - the classical CNN baseline.

One-stage, anchor-based, and built for edge latency: a MobileNetV3 backbone
with depthwise-separable prediction heads on a six-level feature pyramid, all
at a fixed 320x320 input. It is the cheapest of the three models by a wide
margin, which is exactly why it belongs in the comparison - it sets the floor
for "how much accuracy does the extra compute actually buy?".
"""

from __future__ import annotations

from functools import partial

from torch import nn

from src.config import TV_NUM_CLASSES
from src.models.head_transfer import transfer_ssd_classification_head


def build_ssdlite(
    num_classes: int = TV_NUM_CLASSES,
    warm_start: bool = True,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int | None = None,
):
    """Build SSDLite with a 5-class (+background) head.

    Parameters
    ----------
    num_classes:
        Includes background, so 6 for this project.
    warm_start:
        Copy the pretrained apple/banana/broccoli/carrot/orange logits into the
        new head instead of re-initialising it randomly. See
        `src.models.head_transfer`.
    trainable_backbone_layers:
        `None` lets torchvision pick its default (6 of 6 trainable when
        pretrained). Lower it to freeze early layers if VRAM gets tight.
    """
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models.detection.ssdlite import (
        SSDLite320_MobileNet_V3_Large_Weights,
        SSDLiteClassificationHead,
    )

    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    old_head = model.head.classification_head
    old_num_classes = old_head.num_columns  # 91 for the COCO checkpoint

    # The head is a stack of per-pyramid-level prediction blocks; each block's
    # final 1x1 conv is the only part whose shape depends on the class count.
    in_channels = [block[-1].in_channels for block in old_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    # torchvision's own ssdlite builder uses these BatchNorm settings; matching
    # them keeps the new head statistically identical to the one we replace.
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    new_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=norm_layer,
    )

    transfer_stats = None
    if warm_start:
        transfer_stats = transfer_ssd_classification_head(
            old_head=old_head,
            new_head=new_head,
            num_anchors=num_anchors,
            old_num_classes=old_num_classes,
            new_num_classes=num_classes,
        )

    model.head.classification_head = new_head
    model.transform.min_size = (320,)
    model.transform.max_size = 320

    model.meta = {
        "key": "ssdlite",
        "num_classes": num_classes,
        "warm_start": warm_start,
        "transfer_stats": transfer_stats,
        "imgsz": 320,
        "params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return model


def ssdlite_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters so norm layers and biases skip weight decay.

    Applying L2 to BatchNorm scales and biases is a well-known small
    regression; on a short fine-tune it costs a fraction of a mAP point for
    free, so it is worth the six lines.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
