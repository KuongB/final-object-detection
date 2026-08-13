"""Training wrapper for YOLO11s.

Ultralytics owns its own trainer, and fighting it would be a mistake - its
mosaic/mixup schedule, task-aligned assigner, EMA and auto-batch logic are a
large part of why the YOLO family performs as well as it does. So this module
is thin: it configures the run from `src.config.MODELS`, launches
`YOLO.train()`, then re-scores the result through *our* COCO evaluator so the
number that reaches the report was produced the same way as SSDLite's and
D-FINE's.

That last step is the important one. Ultralytics reports its own mAP, computed
with its own matching and interpolation; it is usually a point or two above
`pycocotools` on the same weights. Quoting it next to two pycocotools numbers
would hand YOLO an advantage it did not earn.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import torch

from src.config import DATA_YAML, MODELS, NUM_CLASSES, RANDOM_SEED, RUNS_DIR, run_dir
from src.evaluation.coco_eval import evaluate_detections
from src.evaluation.predict import predict_yolo
from src.training.artifacts import promote
from src.training.common import RunHistory, describe_environment, seed_everything

MODEL_KEY = "yolo11s"


def run(
    epochs: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    imgsz: int | None = None,
    tag: str = "",
    device_str: str = "auto",
    resume: bool = False,
    promote_weights: bool = True,
    extra: dict | None = None,
) -> dict:
    """Fine-tune YOLO11s and score the best weights with pycocotools."""
    from ultralytics import YOLO

    cfg = dict(MODELS[MODEL_KEY]["train_kwargs"])
    if epochs is not None:
        cfg["epochs"] = epochs
    if batch is not None:
        cfg["batch"] = batch
    if workers is not None:
        cfg["workers"] = workers
    if extra:
        cfg |= extra

    size = imgsz or MODELS[MODEL_KEY]["imgsz"]
    seed_everything(RANDOM_SEED)

    device = 0 if (device_str == "auto" and torch.cuda.is_available()) else (
        device_str if device_str != "auto" else "cpu"
    )
    out_dir = run_dir(MODEL_KEY, tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{MODEL_KEY}] device={device}  output={out_dir}")
    model = YOLO(MODELS[MODEL_KEY]["checkpoint"])

    started = time.perf_counter()
    model.train(
        data=str(DATA_YAML),
        imgsz=size,
        seed=RANDOM_SEED,
        project=str(RUNS_DIR),
        name=out_dir.name,
        exist_ok=True,
        resume=resume,
        device=device,
        # `val` uses our val split via data.yaml; the definitive score is the
        # pycocotools pass below, this just drives checkpoint selection.
        val=True,
        plots=True,
        **cfg,
    )
    train_seconds = time.perf_counter() - started

    best_weights = out_dir / "weights" / "best.pt"
    if not best_weights.is_file():  # ultralytics may suffix the dir on collision
        candidates = sorted(RUNS_DIR.glob(f"{out_dir.name}*/weights/best.pt"))
        if candidates:
            best_weights = candidates[-1]
    print(f"[{MODEL_KEY}] best weights: {best_weights}")

    history = RunHistory(
        model_key=MODEL_KEY,
        display_name=MODELS[MODEL_KEY]["display_name"],
        config=cfg | {"imgsz": size},
        environment=describe_environment(),
    )
    history.epochs = _read_ultralytics_csv(best_weights.parent.parent / "results.csv")

    best_map = -1.0
    if best_weights.is_file():
        scored = YOLO(str(best_weights))
        detections, info = predict_yolo(scored, split="val", imgsz=size, device=device)
        metrics = evaluate_detections(detections, split="val", verbose=False)
        best_map = metrics["mAP_50_95"]
        history.best = {
            "val_mAP_50_95": best_map,
            "val_mAP_50": metrics["mAP_50"],
            "checkpoint": str(best_weights),
            "val_seconds": round(info["wall_seconds"], 2),
        }
        print(f"[{MODEL_KEY}] pycocotools val mAP@[.5:.95]={best_map:.4f}")

        # Mirror to the same `best.pt` name the other two models use, so
        # `promote` and every downstream reader see one layout.
        shutil.copy2(best_weights, out_dir / "best.pt")

    history.totals = {"train_seconds": round(train_seconds, 1), "epochs": cfg["epochs"]}
    history.save(out_dir / "history.json")

    if promote_weights and best_weights.is_file():
        promote(
            MODEL_KEY,
            out_dir,
            {
                "display_name": MODELS[MODEL_KEY]["display_name"],
                "framework": MODELS[MODEL_KEY]["framework"],
                "num_classes": NUM_CLASSES,
                "imgsz": size,
                "base_checkpoint": MODELS[MODEL_KEY]["checkpoint"],
                "warm_start": True,
                "params": sum(p.numel() for p in scored.model.parameters()),
                "val_mAP_50_95": round(best_map, 5),
            },
        )

    print(f"\n[{MODEL_KEY}] done in {train_seconds / 60:.1f} min")
    return {"model_key": MODEL_KEY, "best_map": best_map, "run_dir": str(out_dir)}


def _read_ultralytics_csv(path: Path) -> list[dict]:
    """Parse `results.csv` into the same per-epoch schema the other models use."""
    if not path.is_file():
        return []
    import csv

    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            clean = {k.strip(): v for k, v in raw.items() if k}
            record = {"epoch": int(float(clean.get("epoch", 0)))}
            for key, value in clean.items():
                if key == "epoch":
                    continue
                try:
                    record[key] = float(value)
                except (TypeError, ValueError):
                    record[key] = value
            rows.append(record)
    return rows
