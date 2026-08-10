"""Shared training utilities: reproducibility, early stopping, history logging.

Used by the SSD loop directly. The ultralytics runs get the same seed, the same
patience and the same history schema through `scripts/11_train_ultralytics.py`,
so the three models can be compared on equal terms in step 5.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed every RNG the pipeline touches.

    `deterministic=True` also disables cuDNN autotuning, which costs roughly
    10-20% throughput. Left off by default: with 40-epoch runs the wall-clock
    cost is real, and detection metrics move by far less than the run-to-run
    noise this removes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_environment() -> dict:
    """Snapshot of what the run happened on - goes into the results file so the
    speed numbers in the report can be attributed to real hardware."""
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info |= {
            "gpu": props.name,
            "gpu_memory_gb": round(props.total_memory / 1024**3, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda_version": torch.version.cuda,
        }
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001
        info["git_commit"] = None
    return info


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": total,
        "params_trainable": trainable,
        "params_total_m": round(total / 1e6, 2),
    }


# ---------------------------------------------------------------------------
# early stopping
# ---------------------------------------------------------------------------
@dataclass
class EarlyStopping:
    """Stop when the monitored metric has not improved for `patience` epochs.

    Mirrors ultralytics' `patience` semantics (count epochs since the best, not
    since the last improvement of any size) so the stopping rule is the same for
    all three models.
    """

    patience: int = 10
    min_delta: float = 0.0
    mode: str = "max"

    best: float = field(init=False)
    best_epoch: int = field(default=-1, init=False)
    num_bad_epochs: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)

    def __post_init__(self):
        self.best = -float("inf") if self.mode == "max" else float("inf")

    def _improved(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Returns True when this epoch is a new best."""
        if self._improved(value):
            self.best = value
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        if self.patience > 0 and self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False

    def summary(self) -> dict:
        return {
            "best_metric": self.best,
            "best_epoch": self.best_epoch,
            "stopped_early": self.should_stop,
            "patience": self.patience,
            "epochs_without_improvement": self.num_bad_epochs,
        }


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------
class TrainingHistory:
    """Per-epoch log, written to JSON after every epoch.

    Flushing each epoch (rather than at the end) means a crashed or interrupted
    run still leaves usable curves for the report.
    """

    def __init__(self, run_dir: Path, model_name: str, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "history.json"
        self.model_name = model_name
        self.config = config
        self.epochs: list[dict] = []
        self.started = time.perf_counter()
        self.environment = describe_environment()

    def log_epoch(self, **metrics) -> None:
        metrics["wall_time_s"] = round(time.perf_counter() - self.started, 2)
        self.epochs.append(metrics)
        self.flush()

    def flush(self, extra: dict | None = None) -> None:
        payload = {
            "model": self.model_name,
            "config": self.config,
            "environment": self.environment,
            "total_train_time_s": round(time.perf_counter() - self.started, 2),
            "epochs": self.epochs,
        }
        if extra:
            payload |= extra
        self.path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


__all__ = [
    "AverageMeter",
    "EarlyStopping",
    "TrainingHistory",
    "count_parameters",
    "describe_environment",
    "format_duration",
    "get_device",
    "set_seed",
]
