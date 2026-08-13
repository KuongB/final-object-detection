"""Training entry for D-FINE-N.

The loop itself is shared with SSDLite (`src.training.loop`) so the comparison
isolates the architecture rather than the training code. What differs is what a
DETR needs:

* **AdamW, not SGD.** Transformer blocks with LayerNorm do not train well under
  plain momentum SGD at these learning rates.
* **A 10x lower backbone LR.** See `dfine_param_groups`.
* **Tight gradient clipping (0.1).** DETR-family losses spike hard when the
  Hungarian matcher reassigns queries between steps; without clipping those
  spikes reach the weights.
* **The model computes its own loss.** We pass `labels=` into the forward call
  and the bipartite matcher, VFL classification loss and box losses all run
  inside `DFineForObjectDetection`.
"""

from __future__ import annotations

import re

import torch

from src.config import MODELS, NUM_CLASSES, RANDOM_SEED, run_dir
from src.data.coco_dataset import DFineDetectionDataset, build_dfine_collate
from src.evaluation.coco_eval import evaluate_detections
from src.evaluation.predict import predict_dfine
from src.models.dfine import build_dfine, build_dfine_processor, dfine_param_groups
from src.training.common import (
    ModelEMA,
    build_lr_lambda,
    get_device,
    seed_everything,
)
from src.training.loop import TrainSpec, run_loop

MODEL_KEY = "dfine"

#: `loss_giou_aux_3`, `loss_fgl_dn_1`, ... - the per-decoder-layer and
#: per-denoising-group copies of the three real loss terms.
_AUXILIARY_TERM = re.compile(r"_(aux|dn)_\d+$")


def build_train_loader(processor, batch_size: int, workers: int, device: torch.device):
    """Only the training loader - `predict_dfine` builds its own for validation."""
    from torch.utils.data import DataLoader

    return DataLoader(
        DFineDetectionDataset("train", augment=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=build_dfine_collate(processor),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _forward_losses(model, batch, device):
    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    labels = [
        {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
         for k, v in target.items()}
        for target in batch["labels"]
    ]
    outputs = model(pixel_values=pixel_values, labels=labels)

    # The loss dict carries one entry per decoder layer (`_aux_N`) and per
    # denoising group (`_dn_N`) on top of the three real terms - 19 entries in
    # total, which turns the epoch log line into a wall. Keep the primaries;
    # `outputs.loss` already sums all of them.
    components = {
        k: float(v.detach())
        for k, v in (outputs.loss_dict or {}).items()
        if not _AUXILIARY_TERM.search(k)
    }
    return outputs.loss, components, pixel_values.shape[0]


def run(
    epochs: int | None = None,
    batch_size: int | None = None,
    workers: int | None = None,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    warm_start: bool = True,
    tag: str = "",
    validate_every: int = 1,
    device_str: str = "auto",
    resume: bool = False,
    promote_weights: bool = True,
) -> dict:
    """Train D-FINE-N end to end and return the run summary."""
    cfg = dict(MODELS[MODEL_KEY]["train_kwargs"])
    if epochs is not None:
        cfg["epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if workers is not None:
        cfg["workers"] = workers

    seed_everything(RANDOM_SEED)
    device = get_device(device_str)
    out_dir = run_dir(MODEL_KEY, tag)

    checkpoint = MODELS[MODEL_KEY]["checkpoint"]
    imgsz = MODELS[MODEL_KEY]["imgsz"]

    print(f"[{MODEL_KEY}] device={device}  output={out_dir}")
    processor = build_dfine_processor(checkpoint, size=imgsz)
    model = build_dfine(checkpoint, num_classes=NUM_CLASSES, warm_start=warm_start).to(device)
    print(f"[{MODEL_KEY}] params={model.meta['params']:,}  warm_start={warm_start}")

    train_loader = build_train_loader(processor, cfg["batch_size"], cfg["workers"], device)
    steps_per_epoch = limit_train_batches or len(train_loader)

    optimizer = torch.optim.AdamW(
        dfine_param_groups(model, cfg["lr"], cfg["backbone_lr"], cfg["weight_decay"]),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(
            total_epochs=cfg["epochs"],
            warmup_epochs=cfg["warmup_epochs"],
            steps_per_epoch=steps_per_epoch,
            schedule=cfg["scheduler"],
        ),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"] and device.type == "cuda")
    ema = ModelEMA(model, decay=cfg["ema_decay"]) if cfg.get("ema_decay") else None

    def validate(eval_model):
        detections, info = predict_dfine(
            eval_model, processor, split="val", device=device,
            batch_size=cfg["batch_size"], workers=cfg["workers"],
            limit_batches=limit_val_batches,
        )
        metrics = evaluate_detections(detections, split="val", verbose=False)
        return metrics, info["wall_seconds"]

    def export_hf(best_model, run_directory):
        """Second copy in HuggingFace layout, so the web app can `from_pretrained`.

        `run_loop` loads the best checkpoint into the model before calling this,
        which is the whole point - exporting whatever happened to be in memory
        at the end of the last epoch would ship a model that scores worse than
        the number in the report.
        """
        best_model.save_pretrained(run_directory / "hf")
        processor.save_pretrained(run_directory / "hf")

    spec = TrainSpec(
        model_key=MODEL_KEY,
        display_name=MODELS[MODEL_KEY]["display_name"],
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        cfg=cfg,
        device=device,
        forward_losses=_forward_losses,
        validate=validate,
        checkpoint_meta={
            "framework": MODELS[MODEL_KEY]["framework"],
            "num_classes": NUM_CLASSES,
            "imgsz": imgsz,
            "base_checkpoint": checkpoint,
            "warm_start": warm_start,
            "params": model.meta["params"],
        },
        on_export=export_hf,
    )

    return run_loop(
        spec,
        out_dir,
        validate_every=validate_every,
        limit_train_batches=limit_train_batches,
        resume=resume,
        promote_weights=promote_weights,
    )
