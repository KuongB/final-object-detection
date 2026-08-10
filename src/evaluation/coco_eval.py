"""One evaluation path for all three models.

Every model - SSD, YOLOv8, RT-DETR - is scored by writing its predictions in
COCO detection format and running the same `pycocotools` COCOeval against the
same ground-truth file. This matters: ultralytics computes mAP with its own
matching code, which does not agree with COCOeval to the decimal. Comparing an
ultralytics number against a COCOeval number would measure the difference
between two metric implementations, not between two models.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from src.config import CLASSES, CLASS_TO_COCO_ID, COCO_ID_TO_CLASS, ann_path


def _summary_keys() -> list[str]:
    """Names for COCOeval.stats, in the order pycocotools fills them."""
    return [
        "mAP",  # IoU=0.50:0.95, all areas, maxDets=100
        "mAP50",
        "mAP75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]


def evaluate_detections(
    detections: list[dict],
    split: str = "test",
    gt_path: Path | None = None,
    verbose: bool = False,
) -> dict:
    """Score COCO-format detections.

    `detections` items: {image_id, category_id, bbox: [x, y, w, h], score}
    with absolute pixel coordinates in the ORIGINAL image resolution.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_path = Path(gt_path) if gt_path else ann_path(split)

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO(str(gt_path))

    if not detections:
        return {
            "error": "no detections produced",
            **{k: 0.0 for k in _summary_keys()},
            "per_class_AP": {c: 0.0 for c in CLASSES},
        }

    with contextlib.redirect_stdout(sink):
        coco_dt = coco_gt.loadRes(list(detections))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    if verbose:
        print(sink.getvalue())

    results = {k: float(v) for k, v in zip(_summary_keys(), coco_eval.stats)}

    # Per-class AP. precision has shape [T, R, K, A, M]:
    #   T=10 IoU thresholds, R=101 recall points, K=classes,
    #   A=4 area ranges (0 = all), M=3 maxDets (2 = 100).
    # -1 marks (class, IoU) combinations with no ground truth, and must be
    # excluded rather than averaged in as zero.
    precision = coco_eval.eval["precision"]
    cat_ids = coco_eval.params.catIds
    per_class: dict[str, float] = {}
    for k, cat_id in enumerate(cat_ids):
        p = precision[:, :, k, 0, 2]
        p = p[p > -1]
        name = COCO_ID_TO_CLASS.get(cat_id, str(cat_id))
        per_class[name] = float(p.mean()) if p.size else float("nan")

    results["per_class_AP"] = {c: per_class.get(c, float("nan")) for c in CLASSES}

    # AP50 per class, easier to interpret for a counting application.
    per_class_50: dict[str, float] = {}
    for k, cat_id in enumerate(cat_ids):
        p = precision[0, :, k, 0, 2]  # index 0 = IoU 0.50
        p = p[p > -1]
        name = COCO_ID_TO_CLASS.get(cat_id, str(cat_id))
        per_class_50[name] = float(p.mean()) if p.size else float("nan")
    results["per_class_AP50"] = {c: per_class_50.get(c, float("nan")) for c in CLASSES}

    results["n_detections"] = len(detections)
    results["n_gt_images"] = len(coco_gt.imgs)
    results["n_gt_annotations"] = len(coco_gt.anns)
    return results


# ---------------------------------------------------------------------------
# precision / recall at an operating point
# ---------------------------------------------------------------------------
def precision_recall_at(
    detections: list[dict],
    split: str = "test",
    score_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    gt_path: Path | None = None,
) -> dict:
    """Precision, recall and F1 at one confidence threshold.

    AP summarises the whole precision-recall curve, which is the right way to
    compare architectures but says nothing about how a deployed model behaves at
    the single threshold the web application actually runs at. This reports that
    operating point.

    Matching follows COCO's rule: within an image, detections are taken in
    descending score order and each is greedily matched to the highest-IoU
    unmatched ground-truth box of the same class. Crowd boxes are excluded, so a
    detection landing on a crowd region counts as neither right nor wrong.
    """
    from pycocotools.coco import COCO

    gt_path = Path(gt_path) if gt_path else ann_path(split)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))

    gt_by_image_class: dict[tuple[int, int], list[list[float]]] = {}
    n_gt_per_class: dict[int, int] = {cid: 0 for cid in CLASS_TO_COCO_ID.values()}
    for ann in coco_gt.anns.values():
        if ann["iscrowd"]:
            continue
        key = (ann["image_id"], ann["category_id"])
        gt_by_image_class.setdefault(key, []).append(ann["bbox"])
        n_gt_per_class[ann["category_id"]] += 1

    kept = [d for d in detections if d["score"] >= score_thresh]
    kept.sort(key=lambda d: -d["score"])

    matched: dict[tuple[int, int], set[int]] = {}
    tp_per_class: dict[int, int] = {cid: 0 for cid in CLASS_TO_COCO_ID.values()}
    fp_per_class: dict[int, int] = {cid: 0 for cid in CLASS_TO_COCO_ID.values()}

    for det in kept:
        key = (det["image_id"], det["category_id"])
        candidates = gt_by_image_class.get(key, [])
        used = matched.setdefault(key, set())

        best_i, best_iou = -1, 0.0
        for i, gt_box in enumerate(candidates):
            if i in used:
                continue
            score = _iou_xywh(det["bbox"], gt_box)
            if score > best_iou:
                best_i, best_iou = i, score

        if best_i >= 0 and best_iou >= iou_thresh:
            used.add(best_i)
            tp_per_class[det["category_id"]] += 1
        else:
            fp_per_class[det["category_id"]] += 1

    per_class = {}
    for name, cid in CLASS_TO_COCO_ID.items():
        tp, fp, n_gt = tp_per_class[cid], fp_per_class[cid], n_gt_per_class[cid]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": n_gt - tp, "n_gt": n_gt,
        }

    tp = sum(tp_per_class.values())
    fp = sum(fp_per_class.values())
    n_gt = sum(n_gt_per_class.values())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "score_thresh": score_thresh,
        "iou_thresh": iou_thresh,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": n_gt - tp, "n_gt": n_gt,
        "per_class": per_class,
    }


def best_f1_threshold(
    detections: list[dict],
    split: str = "test",
    thresholds: np.ndarray | None = None,
    iou_thresh: float = 0.5,
) -> dict:
    """Sweep the confidence threshold and return the best-F1 operating point.

    The web application has to pick one threshold; this is how that choice gets
    made from data instead of by guessing 0.5.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    sweep = []
    for thr in thresholds:
        res = precision_recall_at(detections, split, float(thr), iou_thresh)
        sweep.append({
            "score_thresh": round(float(thr), 3),
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
        })
    best = max(sweep, key=lambda r: r["f1"])
    return {"best": best, "sweep": sweep}


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def save_detections(detections: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detections), encoding="utf-8")
    return path


def load_detections(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "best_f1_threshold",
    "evaluate_detections",
    "load_detections",
    "precision_recall_at",
    "save_detections",
]
