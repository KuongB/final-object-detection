"""Command-line front end for evaluation - the whole of it.

`scripts/20_evaluate.py` is only a launcher; the behaviour lives here, next to
the runner it drives, matching how training is laid out.

    python scripts/20_evaluate.py
    python scripts/20_evaluate.py --model yolo11s --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import RESULTS_DIR, SPLITS
from src.evaluation.runner import format_summary, run_evaluation
from src.training.artifacts import read_index


class Tee:
    """Mirror a stream to a file, so the run leaves a readable record."""

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
        return getattr(self.__dict__["stream"], name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/20_evaluate.py",
        description=(
            "Score the trained checkpoints on a held-out split and measure their\n"
            "single-image latency. Models are read from weights/index.json, so only\n"
            "models that finished training and were promoted can be evaluated."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # No `choices=`: what can be *evaluated* is whatever weights/index.json
    # holds, which is not the same set as what can be *trained* (`TRAINERS`).
    # A model trained elsewhere and promoted here belongs to the first set only.
    # `main` validates against the index and lists what is available on a miss.
    parser.add_argument(
        "--model", action="append", default=None, metavar="KEY",
        help="repeatable; defaults to every model present in weights/index.json",
    )
    parser.add_argument("--split", default="test", choices=SPLITS,
                        help="which split to score on (default: test)")
    parser.add_argument("--checkpoint", default="last",
                        choices=("last", "best", "pretrained"),
                        help="which weights to score: the end of training (default), "
                             "the epoch that scored highest on val, or the public COCO "
                             "checkpoint before any fine-tuning (the 'epoch 0' baseline)")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'")
    parser.add_argument("--batch", type=int, default=16, help="inference batch size")
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers")
    parser.add_argument("--benchmark-iterations", type=int, default=50,
                        help="timed forward passes per model; 0 still runs a minimum of 1")
    parser.add_argument("--no-save-detections", action="store_true",
                        help="skip writing the raw per-model COCO detections files")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip rendering the report figures into reports/figures/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    index = read_index()
    if not index:
        raise SystemExit(
            "weights/index.json is empty or missing - nothing has been trained and "
            "promoted yet. Run: python scripts/10_train.py --model all --epochs 15"
        )

    requested = args.model or ["all"]
    keys = list(index) if "all" in requested else requested

    missing = [k for k in keys if k not in index]
    if missing:
        raise SystemExit(
            f"not in weights/index.json: {', '.join(missing)}\n"
            f"available: {', '.join(index)}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"evaluate_{args.split}.log"
    print(f"evaluating {', '.join(keys)} on '{args.split}' "
          f"using the '{args.checkpoint}' checkpoint  (log: {log_path})", flush=True)

    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    try:
        payload = run_evaluation(
            keys,
            split=args.split,
            device_str=args.device,
            batch_size=args.batch,
            workers=args.workers,
            benchmark_iterations=max(1, args.benchmark_iterations),
            save_raw=not args.no_save_detections,
            checkpoint=args.checkpoint,
        )
        # The tee writes to the real stdout as well, so printing once here
        # reaches both the console and the log file.
        print()
        print(format_summary(payload))

        if not args.no_figures:
            # Imported here so a run with --no-figures never pays for matplotlib.
            from src.evaluation.figures import build_all

            print()
            build_all(payload)
    finally:
        sys.stdout = tee.stream
        tee.file.close()

    return 0
