"""Warm-start a 5-class detection head from an 80/91-class COCO checkpoint.

All five of our classes - apple, banana, broccoli, carrot, orange - are COCO
classes. The pretrained checkpoints therefore already contain a trained
classifier row for each of them. The default fine-tuning path throws those rows
away and re-initialises the head randomly, which means the first epochs are
spent re-learning something the checkpoint already knew.

This module instead *selects* the five relevant rows (plus the background /
no-object row) out of the pretrained classifier and copies them into the new
5-class head. The backbone, neck and box regressor are untouched - they are
class-agnostic and transfer as-is.

The payoff is visible before training even starts: a warm-started model scores
a non-trivial mAP at epoch 0, whereas a randomly-initialised head scores ~0.
`python scripts/10_train.py --model <key> --no-warm-start` is the ablation.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import CLASSES, HF_COCO80_INDEX, TORCHVISION_COCO91_INDEX


# --------------------------------------------------------------------------- #
# torchvision / SSDLite
# --------------------------------------------------------------------------- #


def torchvision_source_rows() -> list[int]:
    """Rows to pull out of a 91-class torchvision head, background first.

    Index 0 of torchvision's COCO category list is `__background__`, and SSD
    keeps an explicit background logit, so it transfers too.
    """
    return [0] + [TORCHVISION_COCO91_INDEX[name] for name in CLASSES]


def transfer_ssd_classification_head(
    old_head: nn.Module,
    new_head: nn.Module,
    num_anchors: list[int],
    old_num_classes: int,
    new_num_classes: int,
) -> dict:
    """Copy the pretrained SSD class logits into a narrower head.

    Channel layout matters here. `SSDScoringHead.forward` reshapes each block's
    output with `view(N, A, K, H, W)`, so the convolution's output channels are
    anchor-major and class-minor: channel `a * K + k` is the logit for class `k`
    of anchor `a`. Selecting rows therefore means striding through the tensor
    once per anchor, not slicing a contiguous block.
    """
    rows = torchvision_source_rows()
    assert len(rows) == new_num_classes, (
        f"{len(rows)} source rows for {new_num_classes} classes"
    )

    stats = {"blocks": 0, "channels_copied": 0, "depthwise_copied": 0}

    with torch.no_grad():
        for block_idx, (old_block, new_block) in enumerate(
            zip(old_head.module_list, new_head.module_list)
        ):
            # Everything before the final 1x1 projection is class-independent
            # (a depthwise 3x3 + norm + ReLU6) and copies over verbatim.
            for old_layer, new_layer in zip(old_block[:-1], new_block[:-1]):
                new_layer.load_state_dict(old_layer.state_dict())
                stats["depthwise_copied"] += 1

            old_conv, new_conv = old_block[-1], new_block[-1]
            anchors = num_anchors[block_idx]

            for anchor in range(anchors):
                for new_k, old_k in enumerate(rows):
                    src = anchor * old_num_classes + old_k
                    dst = anchor * new_num_classes + new_k
                    new_conv.weight[dst].copy_(old_conv.weight[src])
                    new_conv.bias[dst].copy_(old_conv.bias[src])
                    stats["channels_copied"] += 1

            stats["blocks"] += 1

    return stats


# --------------------------------------------------------------------------- #
# HuggingFace / D-FINE
# --------------------------------------------------------------------------- #


def hf_source_rows() -> list[int]:
    """Rows to pull out of a contiguous 80-class HuggingFace head."""
    return [HF_COCO80_INDEX[name] for name in CLASSES]


def transfer_dfine_class_heads(
    model: nn.Module, pretrained_state: dict[str, torch.Tensor]
) -> dict:
    """Copy the five COCO class rows into every classifier D-FINE carries.

    D-FINE does not have one classifier - it has one per decoder layer
    (`decoder.class_embed.*`, all deep-supervised), one on the encoder that
    scores initial queries (`enc_score_head`), and a denoising embedding table
    that holds an extra row for the no-object class. Missing any of them would
    leave part of the model warm and part of it cold.
    """
    rows = hf_source_rows()
    index = torch.as_tensor(rows, dtype=torch.long)
    stats = {"linear_heads": 0, "denoising_tables": 0, "skipped": []}

    current = model.state_dict()

    with torch.no_grad():
        for key, new_tensor in current.items():
            if key not in pretrained_state:
                continue
            old_tensor = pretrained_state[key]
            if old_tensor.shape == new_tensor.shape:
                continue  # already loaded verbatim by from_pretrained

            is_class_head = (
                "class_embed" in key or "enc_score_head" in key
            ) and "denoising" not in key
            is_denoising = "denoising_class_embed" in key

            if is_class_head and old_tensor.shape[0] == 80:
                new_tensor.copy_(old_tensor.index_select(0, index))
                stats["linear_heads"] += 1
            elif is_denoising and old_tensor.shape[0] == 81:
                # Row 80 is the "no object" embedding and must stay last.
                dn_index = torch.cat([index, torch.tensor([80], dtype=torch.long)])
                new_tensor.copy_(old_tensor.index_select(0, dn_index))
                stats["denoising_tables"] += 1
            else:
                stats["skipped"].append((key, tuple(old_tensor.shape), tuple(new_tensor.shape)))

    model.load_state_dict(current)
    return stats
