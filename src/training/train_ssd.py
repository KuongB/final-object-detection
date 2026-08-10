"""Training loop for SSD300-VGG16.

torchvision ships the architecture but no trainer, so this is the hand-written
counterpart to the single `model.train(...)` call the ultralytics models get.
The difference in effort is itself a finding for the report: the CNN baseline
costs roughly 300 lines of loop, scheduler, AMP and evaluation plumbing that
YOLOv8 and RT-DETR provide out of the box.

Design choices that keep the three-way comparison honest:
  * identical splits, identical epoch budget, identical early-stopping rule
  * validation mAP computed with the same COCOeval used in step 5, not a
    home-grown metric
  * best checkpoint selected on val mAP@[.50:.95], the same quantity
    ultralytics uses for its own `best.pt`
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# With AMP, GradScaler skips optimizer.step() whenever it detects an inf/NaN
# gradient - which is normal during the first few iterations while it calibrates
# its scale. The LR schedule is iteration-based and must advance regardless, so
# the resulting "step() called before optimizer.step()" notice is expected here
# and would otherwise print on nearly every run.
warnings.filterwarnings(
    "ignore", message=r"Detected call of", category=UserWarning
)

from src.config import (
    CLASSES,
    NUM_CLASSES_WITH_BACKGROUND,
    PATIENCE,
    RANDOM_SEED,
    RUNS_DIR,
)
from src.data.detection_dataset import (
    CocoDetectionDataset,
    collate_fn,
    ssd_eval_transforms,
    ssd_train_transforms,
)
from src.evaluation.coco_eval import evaluate_detections
from src.models.ssd import build_ssd300
from src.training.common import (
    AverageMeter,
    EarlyStopping,
    TrainingHistory,
    count_parameters,
    format_duration,
    get_device,
    set_seed,
)

MODEL_NAME = "ssd300_vgg16"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_dataloaders(batch_size: int, size: int, num_workers: int):
    train_ds = CocoDetectionDataset("train", ssd_train_transforms(size))
    val_ds = CocoDetectionDataset("val", ssd_eval_transforms(size))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin,
        persistent_workers=num_workers > 0,
        # Each worker prefetches this many batches. Raising it from the default
        # 2 buys more slack against the jitter in per-sample augmentation cost,
        # which is what leaves the GPU idle between batches.
        prefetch_factor=4 if num_workers > 0 else None,
    )
    # Validation loads in the main process on purpose (num_workers=0).
    #
    # Worker processes here would briefly coexist with the persistent training
    # workers once per epoch, and that transient peak is exactly the kind of
    # spike that turns into an out-of-memory failure hours into a run on a
    # 16 GB machine. The cost of avoiding it is near zero: the eval transform is
    # only resize + normalise, with none of the zoom-out/IoU-crop work that
    # makes the training pipeline CPU-bound, so single-process loading keeps up
    # with the GPU anyway.
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size), shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=pin,
    )
    return train_ds, val_ds, train_loader, val_loader


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
def make_scheduler(optimizer, total_iters: int, warmup_iters: int, final_ratio: float):
    """Linear warmup, then cosine decay, stepped per iteration.

    Warmup matters here specifically because the classification head is randomly
    initialised on top of a pretrained backbone: without it the first few hundred
    steps produce large gradients that damage the pretrained VGG features.
    """
    warmup_iters = max(1, min(warmup_iters, max(1, total_iters - 1)))

    def lr_lambda(it: int) -> float:
        if it < warmup_iters:
            return 0.01 + 0.99 * it / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return final_ratio + (1.0 - final_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, optimizer, scheduler, scaler, loader, device, epoch, epochs,
    clip_grad: float, log_every: int,
) -> dict:
    model.train()
    meters = {"loss": AverageMeter(), "bbox_regression": AverageMeter(),
              "classification": AverageMeter()}
    skipped = 0
    started = time.perf_counter()
    n_batches = len(loader)

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [
            {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in t.items()}
            for t in targets
        ]

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        # Hard-negative mining in fp16 occasionally produces a non-finite loss.
        # Skipping the batch is safer than letting a NaN propagate into the
        # weights, which would silently destroy the whole run.
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        scheduler.step()

        batch_n = len(images)
        meters["loss"].update(loss.item(), batch_n)
        for key, value in loss_dict.items():
            if key in meters:
                meters[key].update(value.item(), batch_n)

        if log_every and (i % log_every == 0 or i == n_batches - 1):
            done = i + 1
            rate = done / (time.perf_counter() - started)
            eta = (n_batches - done) / rate if rate else 0
            print(
                f"  epoch {epoch}/{epochs}  [{done:>4}/{n_batches}]  "
                f"loss {meters['loss'].avg:.4f}  "
                f"(cls {meters['classification'].avg:.4f} "
                f"box {meters['bbox_regression'].avg:.4f})  "
                f"lr {optimizer.param_groups[0]['lr']:.5f}  "
                f"{rate:.2f} it/s  ETA {format_duration(eta)}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    return {
        "train_loss": round(meters["loss"].avg, 5),
        "train_loss_cls": round(meters["classification"].avg, 5),
        "train_loss_box": round(meters["bbox_regression"].avg, 5),
        "lr": optimizer.param_groups[0]["lr"],
        "skipped_batches": skipped,
        "train_time_s": round(elapsed, 2),
        "train_img_per_s": round(meters["loss"].count / elapsed, 1) if elapsed else 0,
        "peak_vram_gb": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if device.type == "cuda" else 0
        ),
    }


# ---------------------------------------------------------------------------
# inference -> COCO detections
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_coco(model, dataset, loader, device, size: int | None = None) -> list[dict]:
    """Run the model over a loader and return COCO-format detections.

    No coordinate rescaling here on purpose. The eval pipeline hands torchvision
    the image at its ORIGINAL resolution as raw float [0, 1]; the model's own
    GeneralizedRCNNTransform resizes to 300x300, normalises, and then maps the
    predicted boxes back to the original size in `postprocess_detections`. Doing
    that arithmetic a second time by hand would double-scale every box - and
    that class of error is silent, because the outputs stay plausible.

    Labels come out as 1..5, which is already the COCO `category_id` our
    annotation files use, so no class remapping either.
    """
    model.eval()

    detections: list[dict] = []
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"])
            boxes = output["boxes"].float().cpu().tolist()
            scores = output["scores"].float().cpu().tolist()
            labels = output["labels"].cpu().tolist()

            for (x0, y0, x1, y1), score, label in zip(boxes, scores, labels):
                detections.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [round(x0, 2), round(y0, 2),
                             round(x1 - x0, 2), round(y1 - y0, 2)],
                    "score": round(float(score), 5),
                })
    return detections


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def train_ssd(
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.002,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    size: int = 300,
    num_workers: int = 4,
    patience: int = PATIENCE,
    amp: bool = True,
    clip_grad: float = 5.0,
    warmup_iters: int = 500,
    final_lr_ratio: float = 0.01,
    seed: int = RANDOM_SEED,
    run_name: str = MODEL_NAME,
    log_every: int = 50,
    limit_batches: int | None = None,
) -> dict:
    """Train SSD300 and return a summary dict. Writes into runs/<run_name>/."""
    set_seed(seed)
    device = get_device()
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(
        batch_size, size, num_workers
    )

    model = build_ssd300(NUM_CLASSES_WITH_BACKGROUND).to(device)
    stats = count_parameters(model)

    # Weight decay on norm layers and biases hurts more than it helps; the
    # torchvision reference recipes exclude them the same way.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)
    optimizer = torch.optim.SGD(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=momentum, nesterov=True,
    )

    iters_per_epoch = limit_batches or len(train_loader)
    scheduler = make_scheduler(
        optimizer, epochs * iters_per_epoch, warmup_iters, final_lr_ratio
    )
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None

    config = {
        "model": MODEL_NAME, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "momentum": momentum, "weight_decay": weight_decay,
        "input_size": size, "optimizer": "SGD+nesterov",
        "schedule": "linear warmup -> cosine", "warmup_iters": warmup_iters,
        "amp": scaler is not None, "clip_grad": clip_grad, "patience": patience,
        "seed": seed, "num_workers": num_workers,
        "train_images": len(train_ds), "val_images": len(val_ds),
        **stats,
    }
    history = TrainingHistory(run_dir, MODEL_NAME, config)
    stopper = EarlyStopping(patience=patience, mode="max")

    print(f"{'=' * 78}\nTraining {MODEL_NAME}")
    print("=" * 78)
    for key, value in config.items():
        print(f"  {key:<18} {value}")
    print(f"  device             {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print("=" * 78, flush=True)

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        loader = train_loader
        if limit_batches:
            from itertools import islice

            class _Limited:
                def __iter__(self):
                    return islice(iter(train_loader), limit_batches)

                def __len__(self):
                    return limit_batches

            loader = _Limited()

        train_metrics = train_one_epoch(
            model, optimizer, scheduler, scaler, loader, device,
            epoch, epochs, clip_grad, log_every,
        )

        eval_started = time.perf_counter()
        detections = predict_coco(model, val_ds, val_loader, device, size)
        metrics = evaluate_detections(detections, split="val")
        eval_time = time.perf_counter() - eval_started

        val_map = metrics["mAP"]
        is_best = stopper.step(val_map, epoch)
        if is_best:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "val_mAP": val_map, "config": config},
                best_path,
            )
        torch.save(
            {"model": model.state_dict(), "epoch": epoch,
             "val_mAP": val_map, "config": config},
            last_path,
        )

        history.log_epoch(
            epoch=epoch,
            **train_metrics,
            val_mAP=round(val_map, 5),
            val_mAP50=round(metrics["mAP50"], 5),
            val_mAP75=round(metrics["mAP75"], 5),
            val_mAP_small=round(metrics["mAP_small"], 5),
            val_mAP_medium=round(metrics["mAP_medium"], 5),
            val_mAP_large=round(metrics["mAP_large"], 5),
            val_AR_100=round(metrics["AR_100"], 5),
            per_class_AP={k: round(v, 5) for k, v in metrics["per_class_AP"].items()},
            eval_time_s=round(eval_time, 2),
        )

        flag = "  <- best" if is_best else ""
        print(
            f"epoch {epoch:>3}/{epochs}  loss {train_metrics['train_loss']:.4f}  "
            f"val mAP {val_map:.4f}  mAP50 {metrics['mAP50']:.4f}  "
            f"AP_s {metrics['mAP_small']:.4f}  "
            f"[{format_duration(train_metrics['train_time_s'])} train, "
            f"{format_duration(eval_time)} eval, "
            f"{train_metrics['train_img_per_s']:.0f} img/s, "
            f"{train_metrics['peak_vram_gb']:.1f}GB]{flag}",
            flush=True,
        )

        if stopper.should_stop:
            print(
                f"\nEarly stopping at epoch {epoch}: no val mAP improvement for "
                f"{patience} epochs (best {stopper.best:.4f} @ epoch "
                f"{stopper.best_epoch})"
            )
            break

    summary = {
        "model": MODEL_NAME,
        "epochs_run": len(history.epochs),
        "epochs_planned": epochs,
        **stopper.summary(),
        "total_train_time_s": round(history.elapsed, 2),
        "total_train_time": format_duration(history.elapsed),
        "best_checkpoint": str(best_path),
        **stats,
    }
    history.flush(extra={"summary": summary})

    print(f"\n{'=' * 78}")
    print(f"Done in {summary['total_train_time']}  |  "
          f"best val mAP {stopper.best:.4f} @ epoch {stopper.best_epoch}")
    print(f"Checkpoint: {best_path}")
    print("=" * 78)
    return summary


__all__ = ["MODEL_NAME", "predict_coco", "train_ssd"]
