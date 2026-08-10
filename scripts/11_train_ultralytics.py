"""Train YOLOv8m (YOLO family) or RT-DETR-l (Transformer family).

Both use the ultralytics API, so one script covers them. The contrast with
`10_train_ssd.py` - ~300 lines of hand-written loop for the same job - is itself
a result worth reporting under "training complexity".

    python scripts/11_train_ultralytics.py --model yolov8m
    python scripts/11_train_ultralytics.py --model rtdetr-l
    python scripts/11_train_ultralytics.py --model yolov8m --smoke

Settings held identical to the SSD run so the comparison is like-for-like:
epochs, patience (early stopping), seed, and the train/val/test splits. What
cannot be held identical - input resolution and batch size - is recorded in the
run config and must be stated whenever the speed numbers are compared.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    BATCH_SIZES,
    CLOSE_MOSAIC,
    DATA_YAML,
    EPOCHS,
    INPUT_SIZES,
    PATIENCE,
    RANDOM_SEED,
    RESULTS_DIR,
    RUNS_DIR,
    WARMUP_EPOCHS,
    WEIGHTS_DIR,
)
from src.training.common import (  # noqa: E402
    describe_environment,
    format_duration,
)

# The two ultralytics models of the three-model comparison. Batch sizes and
# input sizes come from src/config.py.
SUPPORTED = ("yolov8m", "rtdetr-l")


def build_model(name: str):
    from ultralytics import RTDETR, YOLO

    if name.startswith("rtdetr"):
        return RTDETR(f"{name}.pt")
    return YOLO(f"{name}.pt")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="yolov8m", choices=list(SUPPORTED))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--no-warmstart", action="store_true",
        help=(
            "skip copying the pretrained class logits into the 5-class head. "
            "Ultralytics discards that head when nc changes 80->5, so without "
            "warm-starting the model relearns classification from scratch"
        ),
    )
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on a tiny fraction - verifies the setup")
    args = parser.parse_args()

    batch = args.batch or BATCH_SIZES[args.model]
    imgsz = args.imgsz or INPUT_SIZES[args.model]
    run_name = args.run_name or args.model
    epochs = args.epochs
    fraction = 1.0

    if args.smoke:
        epochs, fraction, run_name = 2, 0.02, f"smoke_{args.model}"
        print(f"SMOKE MODE: {epochs} epochs on {fraction:.0%} of train\n")

    if not DATA_YAML.is_file():
        print(f"ERROR: {DATA_YAML} not found. Run scripts/02_build_dataset.py first.")
        return 1

    run_dir = RUNS_DIR / run_name
    if run_dir.exists():
        print(f"Removing previous run at {run_dir}")
        shutil.rmtree(run_dir)

    model = build_model(args.model)

    # Ultralytics rebuilds the network at nc=5 inside the trainer and keeps only
    # shape-compatible tensors, which drops the classification head (6 of 475
    # tensors for YOLOv8m, 15 of 941 for RT-DETR-l). Since all five classes are
    # already COCO classes, the pretrained rows for them can be copied into the
    # new head instead of being relearned. Registered as a callback because the
    # trainer only creates the model once training starts.
    if not args.no_warmstart:
        from src.models.head_transfer import make_ultralytics_warmstart_callback

        model.add_callback(
            "on_pretrain_routine_end",
            make_ultralytics_warmstart_callback(f"{args.model}.pt"),
        )

    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=args.patience,      # same early-stopping rule as the SSD run
        seed=RANDOM_SEED,
        deterministic=False,         # matches set_seed(deterministic=False)
        workers=args.workers,
        device=args.device,
        amp=not args.no_amp,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        val=True,
        plots=True,
        fraction=fraction,
        pretrained=True,             # COCO weights; head warm-started below
        # Both of these are counted in epochs, so they were rescaled when the
        # budget dropped from 40 to 15 - see src/config.py for why.
        close_mosaic=CLOSE_MOSAIC,
        warmup_epochs=WARMUP_EPOCHS,
    )

    print(f"{'=' * 78}\nTraining {args.model}")
    print("=" * 78)
    for key, value in train_kwargs.items():
        print(f"  {key:<16} {value}")
    print("=" * 78, flush=True)

    started = time.perf_counter()
    results = model.train(**train_kwargs)
    elapsed = time.perf_counter() - started

    # ------------------------------------------------------------------
    # Normalise ultralytics' output into the same shape the SSD run writes,
    # so step 5 can read all three runs with one loader.
    # ------------------------------------------------------------------
    best_weights = run_dir / "weights" / "best.pt"
    summary = {
        "model": args.model,
        "epochs_planned": epochs,
        "total_train_time_s": round(elapsed, 2),
        "total_train_time": format_duration(elapsed),
        "best_checkpoint": str(best_weights),
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in train_kwargs.items()},
        "warmstart_head": not args.no_warmstart,
        "environment": describe_environment(),
    }

    # Ultralytics' own validation numbers. Kept for reference only - the
    # headline metrics in step 5 come from COCOeval over all three models,
    # because ultralytics' mAP implementation does not match pycocotools.
    try:
        box = results.box
        summary["ultralytics_val"] = {
            "mAP50_95": round(float(box.map), 5),
            "mAP50": round(float(box.map50), 5),
            "mAP75": round(float(box.map75), 5),
            "precision": round(float(box.mp), 5),
            "recall": round(float(box.mr), 5),
            "per_class_mAP50_95": {
                name: round(float(v), 5)
                for name, v in zip(results.names.values(), box.maps)
            },
            "note": "ultralytics internal metric, NOT comparable to COCOeval",
        }
    except Exception as exc:  # noqa: BLE001
        summary["ultralytics_val_error"] = str(exc)

    # Per-epoch history from results.csv, converted to the SSD history schema.
    csv_path = run_dir / "results.csv"
    if csv_path.is_file():
        import pandas as pd

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        summary["epochs_run"] = len(df)
        history = {
            "model": args.model,
            "config": summary["config"],
            "environment": summary["environment"],
            "total_train_time_s": summary["total_train_time_s"],
            "epochs": df.to_dict(orient="records"),
            "summary": summary,
        }
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=1), encoding="utf-8"
        )

        map_col = next((c for c in df.columns if "mAP50-95" in c), None)
        if map_col:
            best_idx = int(df[map_col].idxmax())
            summary["best_epoch"] = int(df.loc[best_idx, "epoch"])
            summary["best_metric"] = round(float(df.loc[best_idx, map_col]), 5)
            summary["stopped_early"] = len(df) < epochs

    if best_weights.is_file() and not args.smoke:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        target = WEIGHTS_DIR / f"{args.model}_best.pt"
        shutil.copy2(best_weights, target)
        summary["exported_weights"] = str(target)
        print(f"\nCopied best weights to {target}")

    if not args.smoke:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"train_summary_{args.model}.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {path}")

    print(f"\n{'=' * 78}")
    print(f"Done in {summary['total_train_time']}  |  "
          f"best mAP50-95 (ultralytics) "
          f"{summary.get('best_metric', float('nan'))}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
