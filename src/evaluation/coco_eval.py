"""One scorer for all three models: `pycocotools` on the same ground-truth file.

Every framework ships its own validator, and they do not agree - ultralytics
computes mAP with its own 101-point interpolation and its own IoU matching,
torchvision ships none, and transformers defers to whatever you hand it.
Reporting those numbers side by side would compare implementations, not
models.

So the pipeline is the same for everyone: run the model, write detections in
COCO's `[{image_id, category_id, bbox, score}]` format, and score them with the
reference `COCOeval` against `data/annotations/instances_<split>.json`. The
only thing that differs between models is the code that produces the boxes.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from src.config import CLASSES, CLASS_TO_COCO_ID, ann_path

#: Index of each COCOeval summary statistic, in the order pycocotools emits them.
_STAT_NAMES = [
    "mAP_50_95",
    "mAP_50",
    "mAP_75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
    "AR_max1",
    "AR_max10",
    "AR_max100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def evaluate_detections(
    detections: list[dict],
    split: str = "test",
    verbose: bool = True,
) -> dict:
    """Score a COCO detections list and return every metric the report needs.

    Returns `mAP_50_95` / `mAP_50` / size-bucketed AP / recall, plus per-class
    AP@[.5:.95] and AP@0.5 - the per-class numbers are what expose *which*
    class an architecture struggles with, which the headline mAP hides.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_path = ann_path(split)

    # pycocotools prints an unconditional banner on load; silence it so the
    # training logs stay readable.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))

    if not detections:
        # A model that predicts nothing is a legitimate (bad) outcome - report
        # zeros rather than crashing inside pycocotools.
        return {
            "split": split,
            "n_detections": 0,
            "n_images": len(coco_gt.imgs),
            **{name: 0.0 for name in _STAT_NAMES},
            "per_class": {name: {"AP_50_95": 0.0, "AP_50": 0.0} for name in CLASSES},
        }

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(list(detections))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        evaluator.summarize()
    summary_text = buffer.getvalue()
    if verbose:
        print(summary_text)

    metrics = {
        "split": split,
        "n_detections": len(detections),
        "n_images": len(coco_gt.imgs),
    }
    metrics |= {name: float(evaluator.stats[i]) for i, name in enumerate(_STAT_NAMES)}
    metrics["per_class"] = _per_class_ap(evaluator, coco_gt)
    metrics["summary_text"] = summary_text
    return metrics


def _per_class_ap(evaluator, coco_gt) -> dict[str, dict[str, float]]:
    """Pull per-category AP out of the accumulated precision tensor.

    `evaluator.eval["precision"]` has shape
    `[iou_thresholds(10), recall_points(101), categories, area_ranges(4), max_dets(3)]`.
    Area range 0 is "all", max-dets index 2 is 100 - the standard COCO setting.
    Entries of -1 mark recall levels the model never reached and must be
    excluded from the mean, not treated as zero.
    """
    precision = evaluator.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    id_to_position = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    per_class: dict[str, dict[str, float]] = {}
    for name in CLASSES:
        position = id_to_position.get(CLASS_TO_COCO_ID[name])
        if position is None:
            per_class[name] = {"AP_50_95": 0.0, "AP_50": 0.0}
            continue

        all_iou = precision[:, :, position, 0, 2]
        iou_50 = precision[0, :, position, 0, 2]
        per_class[name] = {
            "AP_50_95": float(np.mean(all_iou[all_iou > -1])) if (all_iou > -1).any() else 0.0,
            "AP_50": float(np.mean(iou_50[iou_50 > -1])) if (iou_50 > -1).any() else 0.0,
        }
    return per_class


def save_detections(detections: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detections), encoding="utf-8")
    return path


def format_metrics_table(results: dict[str, dict]) -> str:
    """Render the final model-vs-model comparison as fixed-width text."""
    header = (
        f"{'model':<28}{'mAP50-95':>10}{'mAP50':>9}{'mAP75':>9}"
        f"{'APs':>8}{'APm':>8}{'APl':>8}{'AR100':>8}"
    )
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(
            f"{name:<28}{m['mAP_50_95']:>10.4f}{m['mAP_50']:>9.4f}{m['mAP_75']:>9.4f}"
            f"{m['mAP_small']:>8.3f}{m['mAP_medium']:>8.3f}{m['mAP_large']:>8.3f}"
            f"{m['AR_max100']:>8.3f}"
        )

    lines += ["", f"{'per-class AP@[.5:.95]':<28}" + "".join(f"{c:>11}" for c in CLASSES)]
    lines.append("-" * (28 + 11 * len(CLASSES)))
    for name, m in results.items():
        row = f"{name:<28}"
        for cls in CLASSES:
            row += f"{m['per_class'][cls]['AP_50_95']:>11.4f}"
        lines.append(row)
    return "\n".join(lines)
