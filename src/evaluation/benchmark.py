"""Speed and size measurements - the other half of the model comparison.

The report has to argue about "practical applicability", and mAP alone cannot
support that argument. This module measures the things that decide whether a
model can actually sit behind the web app: parameter count, model file size,
and single-image latency on both GPU and CPU.

Two details that make the difference between a real measurement and a
misleading one:

* **Warmup.** The first forward pass through a CUDA model pays for kernel
  autotuning and lazy module init. Timing it would flatter whichever model ran
  second. Every measurement here discards warmup iterations.
* **Synchronisation.** CUDA calls are asynchronous, so `time.perf_counter()`
  around a forward pass measures how long it took to *queue* the work. Each
  timed iteration calls `torch.cuda.synchronize()`.

Latency is reported at batch size 1, because that is what an upload-one-image
web request actually does.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch


def count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": total,
        "params_trainable": trainable,
        "params_millions": round(total / 1e6, 3),
    }


def checkpoint_size_mb(path: str | Path) -> float:
    path = Path(path)
    return round(path.stat().st_size / 1024**2, 2) if path.is_file() else 0.0


@torch.no_grad()
def measure_latency(
    forward,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, float]:
    """Time a zero-argument `forward` callable and return latency statistics.

    Returns the median as well as the mean: on a laptop GPU, thermal throttling
    and background compositor work produce occasional multi-millisecond
    outliers, and the median is the honest summary of typical latency.
    """
    for _ in range(warmup):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        forward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    return {
        "latency_ms_mean": round(statistics.mean(samples), 3),
        "latency_ms_median": round(statistics.median(samples), 3),
        "latency_ms_p90": round(samples[int(0.9 * len(samples)) - 1], 3),
        "latency_ms_min": round(samples[0], 3),
        "fps": round(1000.0 / statistics.median(samples), 2),
        "iterations": iterations,
    }


def benchmark_torch_model(
    model,
    imgsz: int,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
) -> dict:
    """Latency for a plain `nn.Module` that takes a single image tensor."""
    model = model.to(device).eval()
    sample = torch.rand(3, imgsz, imgsz, device=device)

    def forward():
        model([sample])

    return measure_latency(forward, device, warmup, iterations)


def benchmark_hf_model(
    model,
    imgsz: int,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
) -> dict:
    """Latency for a HuggingFace detection model taking `pixel_values`."""
    model = model.to(device).eval()
    sample = torch.rand(1, 3, imgsz, imgsz, device=device)

    def forward():
        model(pixel_values=sample)

    return measure_latency(forward, device, warmup, iterations)


def benchmark_ultralytics_model(
    model,
    imgsz: int,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
) -> dict:
    """Latency for an ultralytics `YOLO`, measured on its raw torch module.

    Calling `model.predict()` would fold image loading, letterboxing and NMS
    into the number, which is a different quantity from the other two models'
    pure forward pass. Timing `model.model` keeps all three comparable; the
    end-to-end cost is captured separately by the evaluation wall clock.
    """
    net = model.model.to(device).eval()
    sample = torch.rand(1, 3, imgsz, imgsz, device=device)

    def forward():
        net(sample)

    return measure_latency(forward, device, warmup, iterations)


def estimate_flops(model, imgsz: int, device: torch.device, hf: bool = False) -> float | None:
    """GFLOPs for one forward pass, or `None` if the profiler cannot trace it.

    Best-effort: DETR-style models with dynamic control flow sometimes defeat
    the FLOP counter, and a missing number is better than a wrong one.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    model = model.to(device).eval()
    try:
        with FlopCounterMode(display=False) as counter:
            with torch.no_grad():
                if hf:
                    model(pixel_values=torch.rand(1, 3, imgsz, imgsz, device=device))
                else:
                    model([torch.rand(3, imgsz, imgsz, device=device)])
        return round(counter.get_total_flops() / 1e9, 2)
    except Exception:  # noqa: BLE001 - profiling must never break a run
        return None
