"""Score every trained model on the held-out test split, one way for all three.

Training already reports a validation mAP per epoch, but that number chose the
checkpoint, so it cannot also be the result - the model was selected on it.
This module runs the saved checkpoints over `test`, which nothing has ever been
selected on, and records what comes out.

Everything model-specific is already solved elsewhere and reused here:
`load_trained` rebuilds each architecture from `weights/index.json`, the
`predict_*` adapters turn three different output formats into one COCO
detections list, and `evaluate_detections` scores all of them with the same
`pycocotools`. What this module adds is the loop, the speed measurements, and
one JSON that holds the whole picture.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.config import (
    CLASS_TO_COCO_ID,
    CLASSES,
    HF_COCO80_INDEX,
    MODELS,
    NUM_CLASSES,
    RESULTS_DIR,
    RUNS_DIR,
    TV_NUM_CLASSES,
)
from src.evaluation.benchmark import (
    benchmark_hf_model,
    benchmark_torch_model,
    benchmark_ultralytics_model,
    checkpoint_size_mb,
    count_parameters,
)
from src.evaluation.coco_eval import evaluate_detections, save_detections
from src.models.loader import load_trained
from src.training.common import describe_environment, get_device


def _detect(loaded, split: str, device: torch.device, batch_size: int, workers: int,
            class_map: dict[int, int] | None = None):
    """Run the right predictor for whichever framework produced the checkpoint."""
    if loaded.framework == "torchvision":
        from src.evaluation.predict import predict_ssdlite

        return predict_ssdlite(
            loaded.model, split=split, device=device,
            batch_size=batch_size, workers=workers,
        )

    if loaded.framework == "ultralytics":
        from src.evaluation.predict import predict_yolo

        # Ultralytics takes a device index, not a torch.device.
        return predict_yolo(
            loaded.model, split=split, imgsz=loaded.imgsz,
            device=0 if device.type == "cuda" else "cpu", batch_size=batch_size,
            class_map=class_map,
        )

    from src.evaluation.predict import predict_dfine

    return predict_dfine(
        loaded.model, loaded.processor, split=split, device=device,
        batch_size=batch_size, workers=workers,
    )


def load_pretrained(model_key: str, device_str: str):
    """The COCO checkpoint before any fine-tuning - the "epoch 0" reference.

    All five classes are COCO classes, so each pretrained checkpoint already
    detects them. Scoring that directly answers the question a training curve
    cannot: how much did fine-tuning on this dataset actually add, over simply
    taking the public checkpoint as-is?

    Returns the loaded model plus a class map for the frameworks whose head is
    still 80 classes wide (`None` once the head has been narrowed to five).
    """
    from src.models.loader import LoadedModel
    from src.training.common import get_device

    spec = MODELS[model_key]
    framework, imgsz = spec["framework"], spec["imgsz"]
    device = get_device(device_str)

    if framework == "ultralytics":
        from ultralytics import YOLO

        # Left at 80 classes on purpose - narrowing the head here would mean
        # re-running the warm start, which is a different thing to measure.
        model = YOLO(spec["checkpoint"])
        class_map = {
            HF_COCO80_INDEX[name]: CLASS_TO_COCO_ID[name] for name in CLASSES
        }
        return LoadedModel(model_key, framework, model, imgsz, 80, meta={}), class_map

    if framework == "torchvision":
        from src.models.ssdlite import build_ssdlite

        # `warm_start=True` copies the five pretrained rows into a 5-class head,
        # which is numerically the same predictor restricted to our classes.
        model = build_ssdlite(num_classes=TV_NUM_CLASSES, warm_start=True)
        model.to(device).eval()
        return LoadedModel(model_key, framework, model, imgsz, TV_NUM_CLASSES, meta={}), None

    from src.models.dfine import build_dfine, build_dfine_processor

    processor = build_dfine_processor(spec["checkpoint"], size=imgsz)
    model = build_dfine(spec["checkpoint"], num_classes=NUM_CLASSES, warm_start=True)
    model.to(device).eval()
    return (
        LoadedModel(model_key, framework, model, imgsz, NUM_CLASSES,
                    processor=processor, meta={}),
        None,
    )


def _speed(loaded, device: torch.device, iterations: int) -> dict:
    """Single-image latency - what one web request actually costs.

    Only models measured in the same invocation are comparable to each other:
    this laptop's GPU idles down between runs, and the absolute numbers move
    with whatever the card happened to be doing beforehand.
    """
    if loaded.framework == "torchvision":
        return benchmark_torch_model(loaded.model, loaded.imgsz, device, iterations=iterations)
    if loaded.framework == "ultralytics":
        return benchmark_ultralytics_model(loaded.model, loaded.imgsz, device, iterations=iterations)
    return benchmark_hf_model(loaded.model, loaded.imgsz, device, iterations=iterations)


def _training_facts(model_key: str) -> dict:
    """What the run that produced this checkpoint actually did.

    Read from `runs/<key>/history.json` rather than from the registry, because
    the registry holds the *defaults* and a run may have overridden them.
    """
    path = RUNS_DIR / model_key / "history.json"
    if not path.is_file():
        return {}

    history = json.loads(path.read_text(encoding="utf-8"))
    epochs = history["epochs"]

    # Highest validation score wins. The per-epoch `best` flag marks *every*
    # new record, so picking one of those would land on whichever epoch
    # happened to improve first, not on the epoch that was kept. Ultralytics
    # writes its own column name and no flag at all, so score is the only key
    # both schemas share.
    scored = [
        (e.get("val_mAP_50_95", e.get("metrics/mAP50-95(B)", -1)), e["epoch"])
        for e in epochs
    ]
    best_epoch = max(scored)[1] if scored else None

    per_epoch = [e["seconds"] for e in epochs if e.get("seconds")]
    if not per_epoch:
        # Ultralytics records cumulative wall time; per-epoch cost is the delta.
        cumulative = [e["time"] for e in epochs if e.get("time")]
        per_epoch = [b - a for a, b in zip(cumulative, cumulative[1:])]
    return {
        "train_seconds": history.get("totals", {}).get("train_seconds"),
        "epochs_run": history.get("totals", {}).get("epochs", len(history["epochs"])),
        "steps_per_epoch": history.get("totals", {}).get("steps_per_epoch"),
        "best_epoch": best_epoch,
        "val_mAP_50_95": history.get("best", {}).get("val_mAP_50_95"),
        "median_epoch_seconds": round(sorted(per_epoch)[len(per_epoch) // 2], 1) if per_epoch else None,
        "peak_vram_gb": max(
            (e.get("vram_peak_gb", 0) for e in history["epochs"]), default=None
        ) or None,
        "config": history.get("config", {}),
    }


def resolve_checkpoint(model_key: str, which: str) -> Path | None:
    """Path to the run's final checkpoint, or `None` to use `weights/index.json`.

    `promote` only copies the *best* checkpoint into `weights/`, so scoring the
    end-of-training weights means reaching back into the run directory - and
    ultralytics keeps its own under `weights/` inside that directory.

    The run directory comes from the index rather than from the model key: a
    run started with `--tag deep` lives in `runs/<key>_deep`, and assuming
    `runs/<key>` would silently score an older run instead.
    """
    if which == "best":
        return None

    from src.config import PROJECT_ROOT
    from src.training.artifacts import read_index

    entry = read_index().get(model_key, {})
    run_dir = PROJECT_ROOT / entry["run_dir"] if entry.get("run_dir") else RUNS_DIR / model_key

    for path in (run_dir / "last.pt", run_dir / "weights" / "last.pt"):
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{model_key}: no last.pt under {run_dir} - the run directory may have "
        f"been cleaned up. Use --checkpoint best."
    )


def evaluate_model(
    model_key: str,
    split: str = "test",
    device_str: str = "auto",
    batch_size: int = 16,
    workers: int = 2,
    benchmark_iterations: int = 50,
    save_raw: bool = True,
    checkpoint: str = "best",
) -> dict:
    """Load one trained model, score it on `split`, and measure its speed."""
    device = get_device(device_str)
    class_map = None
    if checkpoint == "pretrained":
        print(f"\n[{model_key}] loading the un-fine-tuned COCO checkpoint "
              f"({MODELS[model_key]['checkpoint']}) ...", flush=True)
        loaded, class_map = load_pretrained(model_key, device_str)
    else:
        weights = resolve_checkpoint(model_key, checkpoint)
        source = str(weights) if weights else "weights/index.json"
        print(f"\n[{model_key}] loading {checkpoint} from {source} ...", flush=True)
        loaded = load_trained(model_key, weights=weights, device=device_str)

    started = time.perf_counter()
    detections, info = _detect(loaded, split, device, batch_size, workers, class_map)
    metrics = evaluate_detections(detections, split=split, verbose=False)
    print(f"[{model_key}] {split}: mAP@[.5:.95]={metrics['mAP_50_95']:.4f}  "
          f"mAP@.5={metrics['mAP_50']:.4f}  ({len(detections)} detections)", flush=True)

    if save_raw:
        save_detections(
            detections,
            RESULTS_DIR / f"detections_{model_key}_{split}_{checkpoint}.json",
        )

    speed = _speed(loaded, device, benchmark_iterations)
    print(f"[{model_key}] latency {speed['latency_ms_median']:.2f} ms/image "
          f"({speed['fps']:.1f} FPS, batch 1)", flush=True)

    # `last.pt` also carries optimiser state, so its file size is not the size
    # of the model - report the promoted `best.pt`, which is what would ship.
    from src.training.artifacts import read_index

    weights_path = read_index().get(model_key, {}).get("weights")
    result = {
        "model_key": model_key,
        "display_name": MODELS[model_key]["display_name"],
        "family": MODELS[model_key]["family"],
        "framework": loaded.framework,
        "imgsz": loaded.imgsz,
        **count_parameters(
            loaded.model.model if loaded.framework == "ultralytics" else loaded.model
        ),
        "checkpoint": checkpoint,
        "checkpoint_mb": checkpoint_size_mb(weights_path) if weights_path else None,
        # A pretrained run has no training of its own to describe.
        "training": {} if checkpoint == "pretrained" else _training_facts(model_key),
        "speed": speed,
        "test": {k: v for k, v in metrics.items() if k != "summary_text"},
        "eval_wall_seconds": round(time.perf_counter() - started, 2),
        "n_images": info["n_images"],
    }

    # Free the GPU before the next model builds its own copy.
    del loaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_evaluation(
    model_keys: list[str],
    split: str = "test",
    device_str: str = "auto",
    batch_size: int = 16,
    workers: int = 2,
    benchmark_iterations: int = 50,
    save_raw: bool = True,
    checkpoint: str = "last",
) -> dict:
    """Evaluate several models and write one `evaluation_<split>.json`."""
    results: dict[str, dict] = {}
    for key in model_keys:
        results[key] = evaluate_model(
            key, split=split, device_str=device_str, batch_size=batch_size,
            workers=workers, benchmark_iterations=benchmark_iterations,
            save_raw=save_raw, checkpoint=checkpoint,
        )

    out = RESULTS_DIR / f"evaluation_{split}_{checkpoint}.json"

    # Merge rather than replace: evaluating one model must not delete the
    # results of the others already recorded for this split and checkpoint.
    merged = dict(results)
    if out.is_file():
        previous = json.loads(out.read_text(encoding="utf-8")).get("models", {})
        merged = {**previous, **results}

    payload = {
        "split": split,
        "checkpoint": checkpoint,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": describe_environment(),
        "models": merged,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")

    # Summaries and figures describe only what this invocation measured.
    return {**payload, "models": results}


def format_summary(payload: dict) -> str:
    """One fixed-width table of the numbers the report needs."""
    from src.config import CLASSES

    rows = payload["models"]
    lines = [
        f"test split - {payload['split']}",
        "",
        f"{'model':<26}{'params(M)':>10}{'mAP50-95':>10}{'mAP50':>8}{'mAP75':>8}"
        f"{'ms/img':>9}{'FPS':>7}{'MB':>7}",
    ]
    lines.append("-" * len(lines[-1]))
    for r in rows.values():
        t = r["test"]
        lines.append(
            f"{r['display_name']:<26}{r['params_millions']:>10.2f}"
            f"{t['mAP_50_95']:>10.4f}{t['mAP_50']:>8.4f}{t['mAP_75']:>8.4f}"
            f"{r['speed']['latency_ms_median']:>9.2f}{r['speed']['fps']:>7.1f}"
            f"{r['checkpoint_mb'] or 0:>7.1f}"
        )

    lines += ["", f"{'per-class AP@[.5:.95]':<26}" + "".join(f"{c:>11}" for c in CLASSES)]
    lines.append("-" * (26 + 11 * len(CLASSES)))
    for r in rows.values():
        row = f"{r['display_name']:<26}"
        for cls in CLASSES:
            row += f"{r['test']['per_class'][cls]['AP_50_95']:>11.4f}"
        lines.append(row)
    return "\n".join(lines)
