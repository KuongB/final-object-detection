"""Warm-start a 5-class detection head from the pretrained 80/91-class one.

The problem this solves
-----------------------
All five target classes are already COCO classes, so the pretrained checkpoints
detect them out of the box (measured zero-shot on our test split: YOLOv8m 0.304
mAP, RT-DETR-l 0.319, SSD300 0.131). But changing `nc` from 80 to 5 makes every
framework discard the classification head, because its shape no longer matches:

    YOLOv8m    6 of 475 tensors dropped   (cv3.*.2.weight/bias)
    RT-DETR-l 15 of 941 tensors dropped   (enc/dec_score_head, denoising embed)
    SSD300     the SSDClassificationHead conv stack

Everything else - backbone, neck, and the class-agnostic box regression - still
transfers. So fine-tuning starts with excellent localisation and a *randomly
initialised* classifier, and has to relearn from 5.8k images what the original
head learned from 118k. On a short schedule that can end up WORSE than the
off-the-shelf model.

The fix
-------
Every dropped tensor is a row-indexed projection to per-class logits. Copying
the rows for the classes we keep reconstructs exactly the pretrained classifier,
restricted to our five classes:

    new_head.weight[our_index] = pretrained_head.weight[coco_index]

Verification
------------
`scripts/21_verify_warmstart.py` evaluates the transplanted 5-class model
zero-shot and compares it against the original 80-class model on the same
images. Matching scores prove the transplant is correct; that check is the
whole reason this can be trusted.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CLASSES


# ---------------------------------------------------------------------------
# index mapping
# ---------------------------------------------------------------------------
def coco_indices_for(names: dict | list) -> list[int]:
    """Index of each of our classes inside the pretrained checkpoint's classes.

    Returned in OUR order (alphabetical), so position i is the source row for
    target row i. Read from the checkpoint's own class list rather than
    hard-coded, because a wrong constant here silently scrambles labels while
    every shape still matches.
    """
    if isinstance(names, dict):
        lookup = {name: int(idx) for idx, name in names.items()}
    else:
        lookup = {name: i for i, name in enumerate(names)}

    missing = [c for c in CLASSES if c not in lookup]
    if missing:
        raise ValueError(
            f"pretrained checkpoint has no classes {missing}; "
            "warm-starting is only possible for classes it already knows"
        )
    return [lookup[c] for c in CLASSES]


def load_pretrained_ultralytics(weights_path: str):
    """Load a .pt checkpoint as a live nn.Module, across ultralytics versions.

    The helper was renamed: `attempt_load_one_weight` up to 8.3.x,
    `load_checkpoint` from 8.4. Both return (model, ckpt).
    """
    from ultralytics.nn import tasks

    for fn_name in ("load_checkpoint", "attempt_load_one_weight"):
        fn = getattr(tasks, fn_name, None)
        if fn is not None:
            result = fn(weights_path)
            return result[0] if isinstance(result, tuple) else result
    raise ImportError(
        "ultralytics exposes neither load_checkpoint nor "
        "attempt_load_one_weight; check the installed version"
    )


def _copy_rows(dst: torch.Tensor, src: torch.Tensor, rows: list[int]) -> None:
    """dst[i] <- src[rows[i]] along dim 0, in place."""
    if dst.shape[0] != len(rows):
        raise ValueError(f"destination has {dst.shape[0]} rows, expected {len(rows)}")
    with torch.no_grad():
        for i, source_row in enumerate(rows):
            dst[i].copy_(src[source_row].to(dst.device, dst.dtype))


# ---------------------------------------------------------------------------
# YOLOv8
# ---------------------------------------------------------------------------
def warmstart_yolo_head(new_model: nn.Module, pretrained_model: nn.Module) -> dict:
    """Copy the pretrained class logits into a freshly built YOLOv8 Detect head.

    The head is `model[-1]` (Detect). Its `cv3` branch produces class logits at
    three feature levels, each ending in Conv2d(c, nc, 1) - a per-class 1x1
    projection, so row i of the weight is entirely class i. The `cv2` (box) and
    `dfl` branches are class-independent and already transferred.
    """
    rows = coco_indices_for(pretrained_model.names)
    new_head = new_model.model[-1]
    old_head = pretrained_model.model[-1]

    copied = 0
    for level, (new_seq, old_seq) in enumerate(zip(new_head.cv3, old_head.cv3)):
        new_conv, old_conv = new_seq[-1], old_seq[-1]
        if not isinstance(new_conv, nn.Conv2d) or not isinstance(old_conv, nn.Conv2d):
            raise TypeError(f"cv3[{level}] does not end in Conv2d as expected")
        _copy_rows(new_conv.weight.data, old_conv.weight.data, rows)
        if new_conv.bias is not None and old_conv.bias is not None:
            _copy_rows(new_conv.bias.data, old_conv.bias.data, rows)
        copied += 1

    return {"model": "yolo", "levels_copied": copied, "source_rows": rows}


# ---------------------------------------------------------------------------
# RT-DETR
# ---------------------------------------------------------------------------
def warmstart_rtdetr_head(new_model: nn.Module, pretrained_model: nn.Module) -> dict:
    """Copy pretrained class logits into a freshly built RTDETRDecoder.

    Three class-dependent pieces, all Linear(hidden -> nc) or Embedding(nc, hd):
      enc_score_head        scores encoder queries during query selection
      dec_score_head[i]     per-decoder-layer classification
      denoising_class_embed embedding used by the denoising training trick
    """
    rows = coco_indices_for(pretrained_model.names)
    new_head = new_model.model[-1]
    old_head = pretrained_model.model[-1]

    copied: list[str] = []

    def copy_linear(name: str, new_mod, old_mod):
        _copy_rows(new_mod.weight.data, old_mod.weight.data, rows)
        if getattr(new_mod, "bias", None) is not None:
            _copy_rows(new_mod.bias.data, old_mod.bias.data, rows)
        copied.append(name)

    if hasattr(new_head, "enc_score_head"):
        copy_linear("enc_score_head", new_head.enc_score_head,
                    old_head.enc_score_head)

    if hasattr(new_head, "dec_score_head"):
        for i, (new_mod, old_mod) in enumerate(
            zip(new_head.dec_score_head, old_head.dec_score_head)
        ):
            copy_linear(f"dec_score_head.{i}", new_mod, old_mod)

    # nn.Embedding: weight is [nc, hd], so the same row indexing applies.
    if hasattr(new_head, "denoising_class_embed"):
        _copy_rows(
            new_head.denoising_class_embed.weight.data,
            old_head.denoising_class_embed.weight.data,
            rows,
        )
        copied.append("denoising_class_embed")

    return {"model": "rtdetr", "tensors_copied": copied, "source_rows": rows}


# ---------------------------------------------------------------------------
# torchvision SSD
# ---------------------------------------------------------------------------
def warmstart_ssd_head(
    new_head: nn.Module, old_head: nn.Module, categories: list[str]
) -> dict:
    """Copy pretrained class logits into a fresh SSDClassificationHead.

    Harder than the ultralytics cases because SSD packs anchors and classes into
    one channel axis. `SSDScoringHead.forward` reshapes the conv output with
    `view(N, -1, num_classes, H, W)`, so the layout is anchor-major:

        channel = anchor_index * num_classes + class_index

    The copy therefore has to reshape to [anchors, classes, ...] first, index
    the class axis, and flatten back. Index 0 is background in both heads and is
    carried over as-is, so the model keeps its calibrated "nothing here" prior.
    """
    # torchvision's 91-entry category list is indexed exactly the way the head's
    # class axis is, so `categories.index(name)` IS the class index. Row 0 is
    # background in both heads and carries over unchanged, which preserves the
    # calibrated "nothing here" prior.
    rows = [0] + [categories.index(c) for c in CLASSES]

    new_nc, old_nc = new_head.num_columns, old_head.num_columns

    copied = 0
    for level, (new_conv, old_conv) in enumerate(
        zip(new_head.module_list, old_head.module_list)
    ):
        anchors = old_conv.out_channels // old_nc
        if new_conv.out_channels != anchors * new_nc:
            raise ValueError(
                f"level {level}: expected {anchors * new_nc} out channels, "
                f"got {new_conv.out_channels}"
            )

        with torch.no_grad():
            ow = old_conv.weight.data.view(anchors, old_nc, *old_conv.weight.shape[1:])
            nw = new_conv.weight.data.view(anchors, new_nc, *new_conv.weight.shape[1:])
            for i, source in enumerate(rows):
                nw[:, i].copy_(ow[:, source])

            ob = old_conv.bias.data.view(anchors, old_nc)
            nb = new_conv.bias.data.view(anchors, new_nc)
            for i, source in enumerate(rows):
                nb[:, i].copy_(ob[:, source])
        copied += 1

    return {
        "model": "ssd", "levels_copied": copied, "source_rows": rows,
        "note": "row 0 is background",
    }


# ---------------------------------------------------------------------------
# ultralytics integration
# ---------------------------------------------------------------------------
def make_ultralytics_warmstart_callback(weights_path: str, verbose: bool = True):
    """Callback for `on_pretrain_routine_end` that transplants the head.

    Registered on the YOLO/RTDETR object before `.train()`. By this point
    ultralytics has built the nc=5 model, loaded every shape-compatible tensor,
    and built the optimizer. Copying in place with `.copy_()` keeps the same
    Parameter objects, so the optimizer's references stay valid.
    """

    def callback(trainer):
        import torch as _torch

        pretrained = load_pretrained_ultralytics(weights_path)
        model = trainer.model
        head_type = type(model.model[-1]).__name__

        if head_type == "RTDETRDecoder":
            info = warmstart_rtdetr_head(model, pretrained)
        elif head_type == "Detect":
            info = warmstart_yolo_head(model, pretrained)
        else:
            raise TypeError(f"no warm-start rule for head type {head_type}")

        del pretrained
        _torch.cuda.empty_cache()

        if verbose:
            print(f"\n[warm-start] {head_type}: copied pretrained class logits "
                  f"for {CLASSES} from COCO rows {info['source_rows']}")
        trainer.warmstart_info = info

    return callback


__all__ = [
    "coco_indices_for",
    "make_ultralytics_warmstart_callback",
    "warmstart_rtdetr_head",
    "warmstart_ssd_head",
    "warmstart_yolo_head",
]
