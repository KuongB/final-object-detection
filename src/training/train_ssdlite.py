"""Training entry for SSDLite320-MobileNetV3-Large.

torchvision gives us the model and the losses but no trainer, so the loop is
hand-rolled - it lives in `src.training.loop`, shared with D-FINE. This module
only supplies the pieces that are specific to SSDLite: SGD with momentum over
`ssdlite_param_groups`, torchvision's list-of-images forward, and validation
through the same COCO scorer the final comparison uses.

That last point matters: the mAP printed each epoch is produced by
`pycocotools` on our own ground-truth file, so it is directly comparable to
D-FINE's and YOLO's - no framework-specific mAP variant enters the picture.
"""

from __future__ import annotations

import torch

from src.config import MODELS, RANDOM_SEED, TV_NUM_CLASSES, run_dir
from src.data.coco_dataset import TorchvisionDetectionDataset, torchvision_collate
from src.data.transforms import build_ssd_transforms
from src.evaluation.coco_eval import evaluate_detections
from src.evaluation.predict import predict_ssdlite
from src.models.ssdlite import build_ssdlite, ssdlite_param_groups
from src.training.common import (
    ModelEMA,
    build_lr_lambda,
    get_device,
    seed_everything,
)
from src.training.loop import TrainSpec, run_loop

MODEL_KEY = "ssdlite"


def build_train_loader(batch_size: int, workers: int, device: torch.device):
    """Only the training loader lives here.

    Validation deliberately does not get a loader from this function:
    `predict_ssdlite` builds its own (un-augmented, unshuffled) one, so a val
    loader created here would be dead weight - and with `persistent_workers`
    it would be dead weight holding worker processes open.
    """
    from torch.utils.data import DataLoader

    train_set = TorchvisionDetectionDataset("train", transforms=build_ssd_transforms(train=True))
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=torchvision_collate,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        # Workers are expensive to spawn on Windows (no fork), so keep them
        # alive between epochs - worth several seconds per epoch.
        persistent_workers=workers > 0,
    )


def _forward_losses(model, batch, device):
    images, targets = batch
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [
        {
            "boxes": t["boxes"].to(device, non_blocking=True),
            "labels": t["labels"].to(device, non_blocking=True),
        }
        for t in targets
    ]
    losses = model(images, targets)
    total = sum(losses.values())
    return total, {k: float(v.detach()) for k, v in losses.items()}, len(images)


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
    """Train SSDLite end to end and return the run summary."""
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

    print(f"[{MODEL_KEY}] device={device}  output={out_dir}")
    model = build_ssdlite(num_classes=TV_NUM_CLASSES, warm_start=warm_start).to(device)
    print(f"[{MODEL_KEY}] params={model.meta['params']:,}  warm_start={warm_start}")

    train_loader = build_train_loader(cfg["batch_size"], cfg["workers"], device)
    steps_per_epoch = limit_train_batches or len(train_loader)

    optimizer = torch.optim.SGD(
        ssdlite_param_groups(model, cfg["weight_decay"]),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        nesterov=True,
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
        detections, info = predict_ssdlite(
            eval_model, split="val", device=device,
            batch_size=cfg["batch_size"], workers=cfg["workers"],
            limit_batches=limit_val_batches,
        )
        metrics = evaluate_detections(detections, split="val", verbose=False)
        return metrics, info["wall_seconds"]

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
            "num_classes": TV_NUM_CLASSES,
            "imgsz": MODELS[MODEL_KEY]["imgsz"],
            "base_checkpoint": MODELS[MODEL_KEY]["checkpoint"],
            "warm_start": warm_start,
            "params": model.meta["params"],
        },
    )

    return run_loop(
        spec,
        out_dir,
        validate_every=validate_every,
        limit_train_batches=limit_train_batches,
        resume=resume,
        promote_weights=promote_weights,
    )
