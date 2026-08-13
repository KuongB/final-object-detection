"""The training loop SSDLite and D-FINE share.

Both are hand-rolled PyTorch loops with the same shape - AMP forward, non-finite
guard, clip, step, EMA, per-epoch COCO validation, best-checkpoint selection -
and only the forward call and the loss dict genuinely differ. Keeping two
copies meant every change (resume, the checkpoint schema, promotion) had to be
written twice and stayed in sync only by luck.

So the loop lives here once, and a `TrainSpec` supplies the three things that
are actually model-specific: how to turn a batch into a loss, how to validate,
and how to export.

Ultralytics is deliberately not routed through this - it owns its own trainer,
and wrapping it would be a fiction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from src.training.artifacts import promote
from src.training.common import (
    LossTracker,
    ModelEMA,
    RunHistory,
    cuda_peak_memory_gb,
    describe_environment,
    format_epoch_line,
    shutdown_loader,
)

#: `(model, batch, device) -> (total_loss, {component: float}, batch_size)`
ForwardLosses = Callable[[torch.nn.Module, object, torch.device], tuple[torch.Tensor, dict, int]]

#: `(eval_model) -> (coco_metrics, wall_seconds)`
Validate = Callable[[torch.nn.Module], tuple[dict, float]]


@dataclass
class TrainSpec:
    """Everything `run_loop` needs, assembled by the per-model trainer."""

    model_key: str
    display_name: str
    model: torch.nn.Module
    train_loader: object
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    ema: ModelEMA | None
    cfg: dict
    device: torch.device
    forward_losses: ForwardLosses
    validate: Validate

    #: Merged into every checkpoint: `framework`, `num_classes`, `imgsz`,
    #: `base_checkpoint`. Read back by `src.models.loader`.
    checkpoint_meta: dict = field(default_factory=dict)

    #: Called once at the end with the *best* weights already loaded into
    #: `model`, for formats that are not a bare state dict (D-FINE's `hf/`).
    on_export: Callable[[torch.nn.Module, Path], None] | None = None


def _save_checkpoint(path: Path, spec: TrainSpec, state_dict: dict, epoch: int, **extra) -> None:
    torch.save(
        {
            "model_key": spec.model_key,
            "state_dict": state_dict,
            "epoch": epoch,
            "config": spec.cfg,
            **spec.checkpoint_meta,
            **extra,
        },
        path,
    )


def _restore(spec: TrainSpec, out_dir: Path) -> tuple[int, float, RunHistory | None]:
    """Reload `last.pt` and return where to pick up from."""
    last_path = out_dir / "last.pt"
    if not last_path.is_file():
        raise FileNotFoundError(f"--resume asked for, but {last_path} does not exist")

    ckpt = torch.load(last_path, map_location=spec.device, weights_only=False)
    spec.model.load_state_dict(ckpt["state_dict"])
    spec.optimizer.load_state_dict(ckpt["optimizer"])
    spec.scheduler.load_state_dict(ckpt["scheduler"])
    spec.scaler.load_state_dict(ckpt["scaler"])
    if spec.ema is not None and ckpt.get("ema") is not None:
        spec.ema.load_state_dict(ckpt["ema"])

    history_path = out_dir / "history.json"
    history = RunHistory.load(history_path) if history_path.is_file() else None
    if history is not None:
        # A run killed between the checkpoint write and the history write can
        # leave records for epochs the weights no longer reflect.
        history.epochs = [e for e in history.epochs if e["epoch"] <= ckpt["epoch"]]

    print(f"[{spec.model_key}] resumed from epoch {ckpt['epoch']}, "
          f"best val mAP={ckpt.get('best_map', -1.0):.4f}")
    return ckpt["epoch"] + 1, ckpt.get("best_map", -1.0), history


def run_loop(
    spec: TrainSpec,
    out_dir: Path,
    validate_every: int = 1,
    limit_train_batches: int | None = None,
    resume: bool = False,
    promote_weights: bool = True,
    log_every: int = 50,
) -> dict:
    """Train `spec.model` to completion and write the run's artefacts.

    Writes `best.pt` (highest val mAP@[.5:.95], EMA weights - this is what
    downstream code should load), `last.pt` (raw weights plus optimiser state,
    for `resume` only) and `history.json` after every epoch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    total_epochs = spec.cfg["epochs"]

    start_epoch, best_map, history = 1, -1.0, None
    if resume:
        start_epoch, best_map, history = _restore(spec, out_dir)

    if history is None:
        history = RunHistory(
            model_key=spec.model_key,
            display_name=spec.display_name,
            config=spec.cfg | spec.checkpoint_meta,
            environment=describe_environment(),
        )

    steps_per_epoch = limit_train_batches or len(spec.train_loader)
    started = time.perf_counter()

    for epoch in range(start_epoch, total_epochs + 1):
        if spec.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()

        tracker, lr = _train_one_epoch(spec, epoch, steps_per_epoch, log_every)
        seconds = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "lr": lr,
            "seconds": round(seconds, 2),
            "vram_peak_gb": round(cuda_peak_memory_gb(), 3),
            **{f"train_{k}": v for k, v in tracker.summary().items()},
        }

        eval_model = spec.ema.ema if spec.ema is not None else spec.model
        if validate_every and epoch % validate_every == 0:
            metrics, val_seconds = spec.validate(eval_model)
            record |= {
                "val_mAP_50_95": round(metrics["mAP_50_95"], 5),
                "val_mAP_50": round(metrics["mAP_50"], 5),
                "val_seconds": round(val_seconds, 2),
            }
            if metrics["mAP_50_95"] > best_map:
                best_map = metrics["mAP_50_95"]
                _save_checkpoint(
                    best_path, spec, eval_model.state_dict(), epoch,
                    metrics={k: v for k, v in metrics.items() if k != "summary_text"},
                )
                record["best"] = True

        history.add_epoch(record)
        print(
            format_epoch_line(
                epoch, total_epochs, tracker, lr, seconds,
                extra=f"val_mAP={record.get('val_mAP_50_95', float('nan')):.4f}",
            ),
            flush=True,
        )

        # Written every epoch, after the history record, so a kill between the
        # two loses an epoch of training rather than desynchronising them.
        _save_checkpoint(
            last_path, spec, spec.model.state_dict(), epoch,
            optimizer=spec.optimizer.state_dict(),
            scheduler=spec.scheduler.state_dict(),
            scaler=spec.scaler.state_dict(),
            ema=spec.ema.state_dict() if spec.ema is not None else None,
            best_map=best_map,
        )
        history.save(out_dir / "history.json")

    # Release the persistent worker pool before anything else starts one -
    # `--model all` runs the three stages in a single process.
    shutdown_loader(spec.train_loader)

    if spec.on_export is not None:
        # Load the best weights into the live model first, so an alternative
        # export format cannot silently ship the last epoch instead.
        if best_path.is_file():
            spec.model.load_state_dict(
                torch.load(best_path, map_location=spec.device, weights_only=False)["state_dict"]
            )
        elif spec.ema is not None:
            spec.model.load_state_dict(spec.ema.ema.state_dict())
        spec.on_export(spec.model, out_dir)

    history.best = {"val_mAP_50_95": best_map, "checkpoint": str(best_path)}
    history.totals = {
        "train_seconds": round(time.perf_counter() - started, 1),
        "epochs": total_epochs,
        "steps_per_epoch": steps_per_epoch,
    }
    history.save(out_dir / "history.json")

    if promote_weights:
        promote(
            spec.model_key,
            out_dir,
            {
                "display_name": spec.display_name,
                **spec.checkpoint_meta,
                "val_mAP_50_95": round(best_map, 5),
            },
        )

    print(f"\n[{spec.model_key}] done in {history.totals['train_seconds'] / 60:.1f} min, "
          f"best val mAP@[.5:.95]={best_map:.4f}")
    return {
        "model_key": spec.model_key,
        "best_map": best_map,
        "run_dir": str(out_dir),
        "checkpoint": str(best_path) if best_path.is_file() else None,
    }


def _train_one_epoch(
    spec: TrainSpec, epoch: int, n_batches: int, log_every: int
) -> tuple[LossTracker, float]:
    spec.model.train()
    tracker = LossTracker()

    for step, batch in enumerate(spec.train_loader):
        if step >= n_batches:
            break

        with torch.autocast(device_type=spec.device.type, enabled=spec.scaler.is_enabled()):
            total, components, batch_n = spec.forward_losses(spec.model, batch, spec.device)

        if not torch.isfinite(total):
            # A single bad batch (e.g. every box cropped away) must not poison
            # the weights; skip it and keep the run alive.
            print(f"  [warn] non-finite loss at step {step}, batch skipped")
            spec.optimizer.zero_grad(set_to_none=True)
            spec.scheduler.step()
            continue

        spec.optimizer.zero_grad(set_to_none=True)
        spec.scaler.scale(total).backward()
        if spec.cfg["clip_grad_norm"]:
            spec.scaler.unscale_(spec.optimizer)
            torch.nn.utils.clip_grad_norm_(spec.model.parameters(), spec.cfg["clip_grad_norm"])
        spec.scaler.step(spec.optimizer)
        spec.scaler.update()
        spec.scheduler.step()

        if spec.ema is not None:
            spec.ema.update(spec.model)

        tracker.update(components | {"total": float(total.detach())}, n=batch_n)

        if log_every and step % log_every == 0:
            print(
                f"    epoch {epoch} [{step:>4}/{n_batches}]  {tracker}  "
                f"lr={spec.optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

    return tracker, spec.optimizer.param_groups[0]["lr"]
