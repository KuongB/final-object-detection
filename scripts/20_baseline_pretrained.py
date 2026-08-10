"""Zero-shot baseline: score the OFF-THE-SHELF COCO checkpoints on our data,
before any fine-tuning.

All five classes (apple, banana, broccoli, carrot, orange) are already COCO
classes, so every pretrained checkpoint can already detect them. That makes a
zero-shot baseline available - and it is the number that gives fine-tuning its
meaning. "YOLOv8m reaches mAP 0.55" says little on its own; "YOLOv8m reaches
0.55, up from 0.44 off the shelf" quantifies what the training actually bought.

Also runs an A/B on SSD input normalisation, which documents a real bug found in
this project: torchvision's SSD normalises internally, so a pipeline that also
applies ImageNet normalisation feeds the network values ~4.4x too large. The two
numbers below show the size of that mistake.

    python scripts/20_baseline_pretrained.py
    python scripts/20_baseline_pretrained.py --split val --models yolov8m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASSES, INPUT_SIZES, RESULTS_DIR  # noqa: E402
from src.evaluation.coco_eval import evaluate_detections  # noqa: E402
from src.evaluation.predict import (  # noqa: E402
    predict_ssd,
    predict_ultralytics,
    torchvision_pretrained_class_map,
    ultralytics_class_map,
)

ULTRALYTICS_MODELS = {"yolov8m": "yolov8m.pt", "rtdetr-l": "rtdetr-l.pt"}


def report(name: str, metrics: dict, elapsed: float) -> None:
    print(f"\n{'-' * 74}")
    print(f"{name}   ({elapsed:.1f}s, {metrics.get('n_detections', 0):,} detections)")
    print("-" * 74)
    print(f"  mAP@[.50:.95] {metrics['mAP']:.4f}   "
          f"mAP@.50 {metrics['mAP50']:.4f}   mAP@.75 {metrics['mAP75']:.4f}")
    print(f"  AP small {metrics['mAP_small']:.4f}   "
          f"medium {metrics['mAP_medium']:.4f}   large {metrics['mAP_large']:.4f}")
    print(f"  AR@100   {metrics['AR_100']:.4f}")
    per_class = "  ".join(
        f"{c}={metrics['per_class_AP'][c]:.3f}" for c in CLASSES
    )
    print(f"  per-class AP: {per_class}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--models", nargs="*",
        default=["ssd300_vgg16", "yolov8m", "rtdetr-l"],
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    print("=" * 74)
    print(f"ZERO-SHOT BASELINE on split '{args.split}'")
    print("  pretrained COCO checkpoints, no fine-tuning")
    print("=" * 74)

    results: dict[str, dict] = {}

    # ---------------------------------------------------------------- SSD
    if "ssd300_vgg16" in args.models:
        import torch
        from torchvision.models.detection import (
            SSD300_VGG16_Weights, ssd300_vgg16,
        )

        weights = SSD300_VGG16_Weights.COCO_V1
        model = ssd300_vgg16(weights=weights, score_thresh=0.001,
                             detections_per_img=100)
        class_map = torchvision_pretrained_class_map(weights)
        print(f"\nSSD 91-class -> our ids: {class_map}")

        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

        # correct: raw [0,1], model normalises internally
        started = time.perf_counter()
        dets = predict_ssd(model, args.split, device, args.batch, class_map,
                           normalize_imagenet=False)
        metrics = evaluate_detections(dets, args.split)
        report("ssd300_vgg16 (pretrained, correct input)", metrics,
               time.perf_counter() - started)
        results["ssd300_vgg16"] = metrics

        # wrong: the double-normalisation bug, kept to quantify it
        started = time.perf_counter()
        dets_bad = predict_ssd(model, args.split, device, args.batch, class_map,
                               normalize_imagenet=True)
        metrics_bad = evaluate_detections(dets_bad, args.split)
        report("ssd300_vgg16 (pretrained, DOUBLE-NORMALISED - the bug)",
               metrics_bad, time.perf_counter() - started)
        results["ssd300_vgg16_double_normalised"] = metrics_bad

        drop = metrics["mAP"] - metrics_bad["mAP"]
        print(f"\n  >> double normalisation costs {drop:.4f} mAP "
              f"({100 * drop / max(metrics['mAP'], 1e-9):.0f}% relative)")

        del model
        torch.cuda.empty_cache()

    # -------------------------------------------------------- ultralytics
    for name in ("yolov8m", "rtdetr-l"):
        if name not in args.models:
            continue
        from ultralytics import RTDETR, YOLO

        loader = RTDETR if name.startswith("rtdetr") else YOLO
        model = loader(ULTRALYTICS_MODELS[name])
        class_map = ultralytics_class_map(model.names)
        print(f"\n{name} 80-class -> our ids: {class_map}")

        started = time.perf_counter()
        dets = predict_ultralytics(
            model, args.split, imgsz=INPUT_SIZES[name], device=args.device,
            batch=args.batch, class_map=class_map,
        )
        metrics = evaluate_detections(dets, args.split)
        report(f"{name} (pretrained)", metrics, time.perf_counter() - started)
        results[name] = metrics

        del model
        import torch

        torch.cuda.empty_cache()

    # ------------------------------------------------------------- summary
    print(f"\n{'=' * 74}")
    print(f"{'model':<38}{'mAP':>8}{'mAP50':>8}{'AP_s':>8}{'AP_l':>8}")
    print("-" * 74)
    for name, m in results.items():
        print(f"{name:<38}{m['mAP']:>8.4f}{m['mAP50']:>8.4f}"
              f"{m['mAP_small']:>8.4f}{m['mAP_large']:>8.4f}")
    print("=" * 74)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"baseline_pretrained_{args.split}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
