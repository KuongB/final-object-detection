"""Run all three trainings back to back, unattended.

Sequential, not parallel: one 8 GB GPU and 16 GB of host RAM cannot hold two of
these runs at once, and sharing the GPU would corrupt the throughput numbers the
report compares.

Order is cheapest first, so partial results exist early if the machine has to be
stopped:

    SSD300-VGG16   ~1.3 h   (24.3M params, 300x300, batch 16)
    YOLOv8m        ~3.6 h   (25.9M params, 640x640, batch 16)
    RT-DETR-l      ~7.1 h   (32.8M params, 640x640, batch 4)

A failure in one model does not stop the others - each is a separate process and
its log is kept for diagnosis.

    python scripts/12_train_all.py
    python scripts/12_train_all.py --only yolov8m rtdetr-l
    python scripts/12_train_all.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EPOCHS, MODEL_NAMES, PATIENCE, RESULTS_DIR, RUNS_DIR  # noqa: E402
from src.training.common import format_duration  # noqa: E402

# Rough wall-clock per model at 40 epochs, measured on the RTX 4060 Laptop.
# Used only to print an ETA; early stopping usually finishes sooner.
ETA_HOURS_AT_40 = {"ssd300_vgg16": 1.3, "yolov8m": 3.6, "rtdetr-l": 7.1}
ETA_REFERENCE_EPOCHS = 40

# Measured on this machine: a spawned DataLoader worker on Windows is a fresh
# Python process that re-imports torch, costing roughly this much resident
# memory. Linux forks instead and is far cheaper - do not reuse this number
# there. Rounded up from the ~0.75 GB observed, because being wrong in the
# other direction means an OOM several hours into a run.
WORKER_RAM_GB = 0.9
# The training process itself: CUDA context, model, optimiser state, and the
# COCO annotations held in memory.
MAIN_PROC_RAM_GB = 3.0
# Never consume the last of physical RAM: Windows starts paging aggressively,
# and a run that pages is slower than the same run with fewer workers. Also
# absorbs whatever the user opens while training runs for hours unattended.
RAM_HEADROOM_GB = 2.0
# Past this the GPU is saturated anyway, and each extra process is pure risk.
MAX_WORKERS = 6


def auto_workers(cpu_cap: int = MAX_WORKERS) -> int:
    """Pick the DataLoader worker count that free RAM can actually support.

    Profiling this project showed the GPU sitting at 58% utilisation with 2
    workers - the bottleneck is decoding and augmenting images on the CPU, not
    the GPU and not VRAM (1.9 GB of 8 GB in use). More workers is the fix, but
    on a 16 GB Windows machine RAM runs out long before the 20 CPU cores do, and
    an earlier attempt at a larger batch died with a numpy allocation failure
    rather than a CUDA OOM.

    So: derive the count from RAM measured at launch, and cap it by cores.
    """
    import os

    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1024**3
    except Exception:  # noqa: BLE001
        print("  (psutil unavailable - falling back to 2 workers)")
        return 2

    budget = free_gb - MAIN_PROC_RAM_GB - RAM_HEADROOM_GB
    by_ram = int(max(0, budget) // WORKER_RAM_GB)
    by_cpu = min(cpu_cap, max(1, (os.cpu_count() or 4) - 2))
    workers = max(2, min(by_ram, by_cpu))

    print(f"  free RAM {free_gb:.1f} GB -> budget {budget:.1f} GB "
          f"-> {by_ram} workers by RAM, {by_cpu} by CPU -> using {workers}")
    return workers


def command_for(model: str, epochs: int, patience: int, workers: int) -> list[str]:
    python = sys.executable
    if model == "ssd300_vgg16":
        return [
            python, str(ROOT / "scripts" / "10_train_ssd.py"),
            "--epochs", str(epochs),
            "--batch-size", "16",
            "--patience", str(patience),
            "--num-workers", str(workers),
        ]
    return [
        python, str(ROOT / "scripts" / "11_train_ultralytics.py"),
        "--model", model,
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--workers", str(workers),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="*", default=list(MODEL_NAMES),
                        choices=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--workers", default="auto",
        help="DataLoader workers, or 'auto' to size them from free RAM",
    )
    args = parser.parse_args()

    models = [m for m in MODEL_NAMES if m in args.only]
    log_dir = RUNS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Choosing DataLoader worker count:")
    workers = auto_workers() if args.workers == "auto" else int(args.workers)

    scale = args.epochs / ETA_REFERENCE_EPOCHS
    total_eta = sum(ETA_HOURS_AT_40.get(m, 3.0) for m in models) * scale

    print("=" * 78)
    print(f"TRAINING {len(models)} MODEL(S) SEQUENTIALLY")
    print("=" * 78)
    for m in models:
        print(f"  {m:<16} ~{ETA_HOURS_AT_40.get(m, 3.0) * scale:.1f} h")
    print(f"  {'TOTAL':<16} ~{total_eta:.1f} h "
          f"(finishes around "
          f"{(datetime.now() + timedelta(hours=total_eta)):%Y-%m-%d %H:%M})")
    print(f"  epochs={args.epochs}  patience={args.patience}  workers={workers}")
    print("  (ETA assumes 2 workers; more workers should beat it)")
    print("=" * 78, flush=True)

    results = {}
    overall_start = time.perf_counter()

    for i, model in enumerate(models, 1):
        log_path = log_dir / f"{model}.log"
        print(f"\n[{i}/{len(models)}] {model}  ->  {log_path}", flush=True)
        started = time.perf_counter()

        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(
                command_for(model, args.epochs, args.patience, workers),
                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, text=True,
            )

        elapsed = time.perf_counter() - started
        ok = proc.returncode == 0
        results[model] = {
            "returncode": proc.returncode,
            "ok": ok,
            "elapsed_s": round(elapsed, 1),
            "elapsed": format_duration(elapsed),
            "log": str(log_path),
        }
        status = "OK" if ok else f"FAILED (exit {proc.returncode})"
        print(f"[{i}/{len(models)}] {model}: {status} in {format_duration(elapsed)}",
              flush=True)
        if not ok:
            print(f"    last lines of {log_path}:")
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in tail[-15:]:
                print(f"    | {line}")

    total = time.perf_counter() - overall_start
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "training_runs.json").write_text(
        json.dumps({"total_s": round(total, 1),
                    "total": format_duration(total),
                    "runs": results}, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 78}")
    print(f"ALL DONE in {format_duration(total)}")
    for model, info in results.items():
        print(f"  {model:<16} {'OK    ' if info['ok'] else 'FAILED'} {info['elapsed']}")
    print("=" * 78)
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
