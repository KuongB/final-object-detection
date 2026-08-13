"""Pieces shared by the SSDLite and D-FINE training loops.

Ultralytics brings its own equivalents for YOLO11s, so nothing here is imposed
on it - but the *logging schema* below is deliberately model-agnostic, so all
three runs end up writing a `history.json` with the same shape and the report
can plot them on one axis.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Reproducibility & device
# --------------------------------------------------------------------------- #


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed every RNG a run touches.

    `deterministic=False` by default on purpose: cuDNN's autotuner picks
    noticeably faster convolution algorithms, and full determinism would cost
    ~15-20% of training throughput for a reproducibility guarantee that a
    detection benchmark does not need (the split, which does matter, is pinned
    separately in `data/splits.json`).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_environment() -> dict:
    """Snapshot of the machine, stored next to every run's metrics."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info |= {
            "gpu": props.name,
            "vram_gb": round(props.total_memory / 1024**3, 2),
            "capability": f"sm_{props.major}{props.minor}",
            "cuda": torch.version.cuda,
        }
    return info


# --------------------------------------------------------------------------- #
# Metering
# --------------------------------------------------------------------------- #


class AverageMeter:
    """Running mean of a scalar - one per loss component."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class LossTracker:
    """Keeps one `AverageMeter` per named loss term, created on first sight."""

    def __init__(self) -> None:
        self.meters: dict[str, AverageMeter] = {}

    def update(self, losses: dict[str, float], n: int = 1) -> None:
        for name, value in losses.items():
            self.meters.setdefault(name, AverageMeter()).update(float(value), n)

    def summary(self) -> dict[str, float]:
        return {name: round(m.avg, 5) for name, m in self.meters.items()}

    def __str__(self) -> str:
        return "  ".join(f"{k}={v:.4f}" for k, v in self.summary().items())


def shutdown_loader(loader) -> None:
    """Tear down a DataLoader's worker processes immediately.

    `persistent_workers=True` keeps workers alive on purpose - respawning them
    every epoch is expensive on Windows, where there is no `fork`. The cost is
    that simply dropping the reference leaves those processes parked until the
    garbage collector gets round to them. When the *next* stage then starts its
    own worker pool (ultralytics does), the two pools can deadlock over the
    CUDA context. Calling this between stages avoids that entirely.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        try:
            iterator._shutdown_workers()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass
        loader._iterator = None


def cuda_peak_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #


def build_lr_lambda(
    total_epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int,
    schedule: str = "cosine",
    final_factor: float = 0.01,
):
    """Per-*step* LR multiplier: linear warmup, then cosine or linear decay.

    Stepping per batch rather than per epoch matters for the warmup: with ~180
    batches an epoch, a per-epoch warmup would jump the LR in three big steps
    and can blow up a freshly-attached detection head in the first few
    iterations.
    """
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Start at 1/warmup_steps rather than 0 so the first step still moves.
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        if schedule == "cosine":
            return final_factor + (1 - final_factor) * 0.5 * (1 + math.cos(math.pi * progress))
        return final_factor + (1 - final_factor) * (1 - progress)

    return lr_lambda


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #


class ModelEMA:
    """Exponential moving average of the weights, evaluated instead of the raw model.

    Standard in every modern detector (YOLO does it internally). On a short
    fine-tune it is worth roughly half a mAP point and, more usefully, it makes
    the epoch-to-epoch validation curve smooth enough to read.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, warmup: int = 2000):
        from copy import deepcopy

        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.updates = 0

        # Pair up the EMA and live tensors ONCE.
        #
        # The obvious implementation calls `model.state_dict()` inside `update`
        # and walks it entry by entry. That rebuilds a ~300-entry OrderedDict in
        # Python and issues two CUDA kernels per entry - roughly 600 launches -
        # on every single optimiser step. Under Windows' WDDM driver, kernel
        # launch overhead is high enough that this cost more than the forward
        # and backward passes combined: measured 1.018 s/step against 0.162
        # s/step for the identical loop without EMA.
        #
        # Caching the tensors is safe because both the optimiser and BatchNorm
        # write in place, so these references stay valid for the whole run.
        ema_state = self.ema.state_dict()
        float_pairs, other_pairs = [], []
        for key, value in model.state_dict().items():
            target = ema_state[key]
            if target.dtype.is_floating_point:
                float_pairs.append((target, value))
            else:
                other_pairs.append((target, value))  # num_batches_tracked, ...

        self._ema_floats = [t for t, _ in float_pairs]
        self._model_floats = [v for _, v in float_pairs]
        self._ema_others = [t for t, _ in other_pairs]
        self._model_others = [v for _, v in other_pairs]

    @torch.no_grad()
    def update(self, model: torch.nn.Module | None = None) -> None:
        self.updates += 1
        # Ramp the decay in: early on, the EMA should track the model closely,
        # otherwise it stays near the initialisation for thousands of steps.
        d = self.decay * (1 - math.exp(-self.updates / self.warmup))

        # `_foreach_*` fuses the whole parameter list into a handful of kernels
        # instead of one pair per tensor.
        torch._foreach_mul_(self._ema_floats, d)
        torch._foreach_add_(self._ema_floats, self._model_floats, alpha=1 - d)

        if self._ema_others:
            torch._foreach_copy_(self._ema_others, self._model_others)

    def state_dict(self) -> dict:
        return {"ema": self.ema.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        # `nn.Module.load_state_dict` copies in place, so the tensor references
        # cached in `__init__` stay valid.
        self.ema.load_state_dict(state["ema"])
        self.updates = state["updates"]


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class RunHistory:
    """The per-run record every model writes, in one schema.

    `epochs` holds one dict per epoch with at least `epoch`, `train_loss`,
    `lr`, `seconds`; validation metrics are merged in when they are computed.
    """

    model_key: str
    display_name: str
    config: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    epochs: list[dict] = field(default_factory=list)
    best: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)

    def add_epoch(self, record: dict) -> None:
        self.epochs.append(record)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "RunHistory":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def format_epoch_line(
    epoch: int, total: int, losses: LossTracker, lr: float, seconds: float, extra: str = ""
) -> str:
    return (
        f"epoch {epoch:>3}/{total}  {losses}  lr={lr:.2e}  "
        f"{seconds:6.1f}s  vram={cuda_peak_memory_gb():.2f}GB  {extra}"
    )
