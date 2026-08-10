"""Train SSD300-VGG16 (the CNN-based baseline).

    python scripts/10_train_ssd.py                       # full 40-epoch run
    python scripts/10_train_ssd.py --smoke               # 2 epochs, few batches
    python scripts/10_train_ssd.py --batch-size 16 --lr 0.001

Batch size and learning rate move together: the defaults (32 / 0.002) follow the
torchvision reference recipe for ssd300_vgg16. If VRAM forces a smaller batch,
scale the lr linearly or the run will underfit - `--auto-lr` does this for you.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EPOCHS, PATIENCE, RESULTS_DIR  # noqa: E402

REFERENCE_BATCH = 32
REFERENCE_LR = 0.002


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=REFERENCE_LR)
    parser.add_argument(
        "--auto-lr", action="store_true",
        help="scale lr linearly from the reference 0.002 @ batch 32",
    )
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--run-name", type=str, default="ssd300_vgg16")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--smoke", action="store_true",
        help="2 epochs x 20 batches - verifies the loop end to end in ~2 min",
    )
    args = parser.parse_args()

    if args.auto_lr:
        args.lr = REFERENCE_LR * args.batch_size / REFERENCE_BATCH
        print(f"--auto-lr: lr = {args.lr:.5f} for batch {args.batch_size}")

    limit_batches = None
    if args.smoke:
        args.epochs = 2
        args.warmup_iters = 10
        args.log_every = 5
        args.run_name = "smoke_ssd300_vgg16"
        limit_batches = 20
        print("SMOKE MODE: 2 epochs x 20 batches\n")

    from src.training.train_ssd import train_ssd

    summary = train_ssd(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        size=args.size,
        num_workers=args.num_workers,
        patience=args.patience,
        amp=not args.no_amp,
        clip_grad=args.clip_grad,
        warmup_iters=args.warmup_iters,
        run_name=args.run_name,
        log_every=args.log_every,
        limit_batches=limit_batches,
    )

    if not args.smoke:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "train_summary_ssd300_vgg16.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
