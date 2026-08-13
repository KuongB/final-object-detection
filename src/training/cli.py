"""Command-line front end for training - the whole of it.

`scripts/10_train.py` is only a launcher. Everything that decides what a run
does lives here, next to the trainers it drives, so there is one place to look
when a flag or a guard behaves unexpectedly.

    python scripts/10_train.py --model ssdlite
    python scripts/10_train.py --model all --epochs 15
    python scripts/10_train.py --model dfine --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import MODELS, run_dir
from src.training import TRAINERS, run_training

#: Fraction of the training set ultralytics uses for `--smoke` - the
#: equivalent of `--limit-train-batches` for the one model that has no such
#: option. 5% of 5,803 images is ~290, a handful of batches.
SMOKE_YOLO_FRACTION = 0.05


class Tee:
    """Mirror a stream to a file, so a background run stays followable."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.file = path.open("a", encoding="utf-8", errors="replace")

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        # `__dict__` rather than `self.stream`, which would recurse. tqdm and
        # ultralytics probe `isatty`/`encoding` before writing.
        return getattr(self.__dict__["stream"], name)


def check_run_dir(out_dir: Path, resume: bool, overwrite: bool) -> None:
    """Refuse to silently write over a previous run's results."""
    if (out_dir / "history.json").is_file() and not (resume or overwrite):
        raise SystemExit(
            f"{out_dir} already holds a finished run.\n"
            f"  --resume     continue it from last.pt\n"
            f"  --overwrite  discard it and start again\n"
            f"  --tag NAME   write to a separate directory instead"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/10_train.py",
        description=(
            "Fine-tune any of the three detectors, or all of them in sequence.\n"
            "Defaults come from the MODELS registry in src/config.py; every flag\n"
            "below overrides them. Flags a given model cannot honour are reported\n"
            "and skipped, so --model all never dies halfway over one stray option."
        ),
        epilog=(
            "Each run streams to runs/<name>/train.log as well as the console, so a\n"
            "background launch can be followed with:\n"
            "    Get-Content runs/ssdlite/train.log -Tail 20 -Wait"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, choices=[*TRAINERS, "all"],
        help="which architecture to train; 'all' runs the three in sequence",
    )
    parser.add_argument("--epochs", type=int, help="override the registry's epoch count")
    parser.add_argument("--batch", type=int, help="batch size")
    parser.add_argument("--workers", type=int, help="dataloader workers (2 is tuned for Windows)")
    parser.add_argument("--imgsz", type=int, help="input size; YOLO only, the others are fixed")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cpu', or an index")
    parser.add_argument("--tag", default="", help="suffix for the run directory")
    parser.add_argument("--validate-every", type=int, help="validate every N epochs; 0 disables")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="ablation: random detection head instead of the COCO one")
    parser.add_argument("--limit-train-batches", type=int, help="truncate each training epoch")
    parser.add_argument("--limit-val-batches", type=int, help="truncate each validation pass")
    parser.add_argument("--resume", action="store_true", help="continue from the run's last.pt")
    parser.add_argument("--overwrite", action="store_true", help="discard an existing run")
    parser.add_argument("--no-promote", action="store_true",
                        help="skip copying the best checkpoint into weights/")
    parser.add_argument("--smoke", action="store_true",
                        help="1 short epoch, tagged 'smoke', not promoted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    tag = args.tag
    if args.smoke:
        args.epochs = args.epochs or 1
        args.limit_train_batches = args.limit_train_batches or 5
        args.limit_val_batches = args.limit_val_batches or 5
        tag = tag or "smoke"

    common = {
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "imgsz": args.imgsz,
        "device_str": args.device,
        "tag": tag,
        "validate_every": args.validate_every,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        # Sent only when they change something, otherwise every run prints an
        # "ignoring ..." line for the models that cannot take them.
        "warm_start": False if args.no_warm_start else None,
        "resume": True if args.resume else None,
        "promote_weights": False if (args.no_promote or args.smoke) else None,
    }

    keys = list(TRAINERS) if args.model == "all" else [args.model]

    # Check every target before training anything, so `--model all` fails on
    # the guard immediately rather than an hour in.
    for key in keys:
        check_run_dir(run_dir(key, tag), args.resume, args.overwrite)

    results = []
    for key in keys:
        kwargs = dict(common)
        if args.smoke and key == "yolo11s":
            kwargs["extra"] = {"fraction": SMOKE_YOLO_FRACTION}

        out_dir = run_dir(key, tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "train.log"
        print(f"\n=== {MODELS[key]['display_name']} -> {out_dir}  (log: {log_path})", flush=True)

        tee = Tee(sys.stdout, log_path)
        sys.stdout = tee
        try:
            results.append(run_training(key, **kwargs))
        finally:
            sys.stdout = tee.stream
            tee.file.close()

    if len(results) > 1:
        print(f"\n{'model':<12}{'best val mAP@[.5:.95]':>24}   run")
        for r in results:
            print(f"{r['model_key']:<12}{r['best_map']:>24.4f}   {r['run_dir']}")
    return 0
