"""D-FINE-N - the Transformer / DETR-based entry.

D-FINE is a real-time DETR descendant. It keeps the end-to-end set-prediction
formulation - no anchors, no NMS, a fixed pool of 300 object queries matched to
ground truth by bipartite assignment - but replaces DETR's direct box
regression with Fine-grained Distribution Refinement: the decoder predicts a
*probability distribution* over corner offsets and sharpens it layer by layer,
which is what gives it DETR-quality localisation at a fraction of the cost.

The nano variant is chosen deliberately. Self-attention over 300 queries plus
denoising groups is the memory-hungry part of training a DETR, and on 8 GB of
VRAM the small/medium variants force a batch size low enough that the loss gets
noisy. Nano keeps batch 8 comfortable.
"""

from __future__ import annotations

from src.config import IDX_TO_CLASS, NUM_CLASSES
from src.models.head_transfer import transfer_dfine_class_heads


def build_dfine(
    checkpoint: str = "ustc-community/dfine-nano-coco",
    num_classes: int = NUM_CLASSES,
    warm_start: bool = True,
):
    """Load D-FINE-N and re-shape its classifiers to our five classes.

    Unlike SSD, D-FINE has no background class - DETR-style models score each
    query against the real classes only and let "no object" be the absence of a
    confident match - so `num_classes` is 5, not 6.
    """
    from transformers import AutoModelForObjectDetection

    label2id = {name: idx for idx, name in IDX_TO_CLASS.items()}

    # `ignore_mismatched_sizes` is what lets transformers build a 5-class model
    # from an 80-class checkpoint: everything that fits is loaded, and the
    # classifiers are left freshly initialised for us to fill in below.
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        id2label=dict(IDX_TO_CLASS),
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    transfer_stats = None
    if warm_start:
        reference = AutoModelForObjectDetection.from_pretrained(checkpoint)
        transfer_stats = transfer_dfine_class_heads(model, reference.state_dict())
        del reference

    model.meta = {
        "key": "dfine",
        "num_classes": num_classes,
        "warm_start": warm_start,
        "transfer_stats": transfer_stats,
        "imgsz": 640,
        "params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return model


def build_dfine_processor(checkpoint: str = "ustc-community/dfine-nano-coco", size: int = 640):
    """The image processor that owns resize, rescale and box conversion."""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(
        checkpoint,
        size={"height": size, "width": size},
        backend="torchvision",  # the fast, tensor-native path
    )


def dfine_param_groups(
    model, lr: float, backbone_lr: float, weight_decay: float
) -> list[dict]:
    """Give the pretrained backbone a 10x smaller learning rate.

    Standard practice for every DETR variant: the backbone arrives with good
    general features and only needs nudging, while the randomly-matched decoder
    heads need to move fast. Using one learning rate for both either destroys
    the backbone or starves the decoder.
    """
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "backbone" in name
        # Norm weights and biases are 1-D; decaying them is counter-productive.
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        if is_backbone:
            (backbone_no_decay if no_decay else backbone_decay).append(param)
        else:
            (head_no_decay if no_decay else head_decay).append(param)

    groups = [
        {"params": head_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


__all__ = ["build_dfine", "build_dfine_processor", "dfine_param_groups"]
