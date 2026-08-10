"""Prove the 5-class head transplant is correct, before trusting it.

A warm-started model is built with nc=5 and has the pretrained class logits
copied into it. If the copy is right, that model - WITHOUT ANY TRAINING - must
score the same as the original 80/91-class model does on our five classes.

If instead the rows were mis-indexed, the shapes would still all match and
nothing would raise: the model would simply predict "banana" where it means
"apple", and mAP would collapse. So this comparison is the only real check.

    python scripts/21_verify_warmstart.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.config import (  # noqa: E402
    CLASSES,
    INPUT_SIZES,
    NUM_CLASSES_WITH_BACKGROUND,
    RESULTS_DIR,
)
from src.evaluation.coco_eval import evaluate_detections  # noqa: E402
from src.evaluation.predict import (  # noqa: E402
    identity_class_map,
    predict_ssd,
    predict_ultralytics,
    torchvision_pretrained_class_map,
    ultralytics_class_map,
)
from src.models.head_transfer import (  # noqa: E402
    warmstart_rtdetr_head,
    warmstart_ssd_head,
    warmstart_yolo_head,
)


def line(name, m):
    print(f"  {name:<34} mAP {m['mAP']:.4f}  mAP50 {m['mAP50']:.4f}  "
          f"AP_s {m['mAP_small']:.4f}  AP_l {m['mAP_large']:.4f}")


def verdict(before: dict, after: dict, tol: float = 0.02) -> bool:
    delta = after["mAP"] - before["mAP"]
    ok = abs(delta) <= tol
    print(f"  delta mAP {delta:+.4f}   "
          f"{'MATCH - transplant correct' if ok else 'MISMATCH - investigate'}")
    return ok


def check_ssd(split: str, batch: int, device: str) -> dict:
    from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16

    from src.models.ssd import build_ssd300

    print(f"\n{'=' * 74}\nSSD300-VGG16\n{'=' * 74}")
    weights = SSD300_VGG16_Weights.COCO_V1
    categories = weights.meta["categories"]

    original = ssd300_vgg16(weights=weights, score_thresh=0.001,
                            detections_per_img=100)
    dets = predict_ssd(original, split, device, batch,
                       torchvision_pretrained_class_map(weights))
    before = evaluate_detections(dets, split)
    line("pretrained, 91 classes", before)

    fresh = build_ssd300(NUM_CLASSES_WITH_BACKGROUND, pretrained=True,
                         score_thresh=0.001, detections_per_img=100,
                         warmstart_head=False)
    info = warmstart_ssd_head(
        fresh.head.classification_head, original.head.classification_head,
        categories,
    )
    print(f"  transplant: {info['levels_copied']} levels, rows {info['source_rows']}")

    dets2 = predict_ssd(fresh, split, device, batch, identity_class_map())
    after = evaluate_detections(dets2, split)
    line("warm-started, 6 classes", after)
    ok = verdict(before, after)

    del original, fresh
    torch.cuda.empty_cache()
    return {"before": before, "after": after, "match": ok}


def check_ultralytics(name: str, split: str, batch: int, device: str) -> dict:
    from ultralytics import RTDETR, YOLO
    from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
    from ultralytics.utils.torch_utils import intersect_dicts

    from src.models.head_transfer import load_pretrained_ultralytics

    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
    is_rtdetr = name.startswith("rtdetr")
    wrapper = (RTDETR if is_rtdetr else YOLO)(f"{name}.pt")

    dets = predict_ultralytics(
        wrapper, split, imgsz=INPUT_SIZES[name], device=device, batch=batch,
        class_map=ultralytics_class_map(wrapper.names),
    )
    before = evaluate_detections(dets, split)
    line("pretrained, 80 classes", before)

    # Rebuild at nc=5 exactly the way the trainer does, then transplant.
    pretrained = load_pretrained_ultralytics(f"{name}.pt")
    builder = RTDETRDetectionModel if is_rtdetr else DetectionModel
    fresh = builder(pretrained.yaml, nc=len(CLASSES), verbose=False)
    csd = intersect_dicts(pretrained.float().state_dict(), fresh.state_dict())
    fresh.load_state_dict(csd, strict=False)
    print(f"  rebuilt at nc=5: {len(csd)} tensors transferred by shape match")

    info = (warmstart_rtdetr_head if is_rtdetr else warmstart_yolo_head)(
        fresh, pretrained
    )
    print(f"  transplant: source rows {info['source_rows']} for {CLASSES}")

    fresh.names = {i: c for i, c in enumerate(CLASSES)}
    fresh.args = pretrained.args
    fresh.task = getattr(pretrained, "task", "detect")
    wrapper.model = fresh.eval()
    # Ultralytics builds a Predictor on the first predict() call and reuses it
    # afterwards WITHOUT re-reading Model.model. Without clearing it, the second
    # evaluation silently runs the original 80-class network and then applies
    # the 5-class label map to it - which is how this check first reported
    # mAP 0.0000 (person->apple, bicycle->banana, ...).
    wrapper.predictor = None

    dets2 = predict_ultralytics(
        wrapper, split, imgsz=INPUT_SIZES[name], device=device, batch=batch,
        class_map=identity_class_map_for_ultralytics(),
    )
    after = evaluate_detections(dets2, split)
    line("warm-started, 5 classes", after)
    ok = verdict(before, after, tol=0.03 if is_rtdetr else 0.02)
    if is_rtdetr and not ok:
        print("  (RT-DETR selects its top-k queries by max class score, so "
              "restricting to 5 classes legitimately changes which queries "
              "survive - a moderate difference here is expected, a collapse "
              "is not)")

    del wrapper, pretrained, fresh
    torch.cuda.empty_cache()
    return {"before": before, "after": after, "match": ok}


def identity_class_map_for_ultralytics() -> dict[int, int]:
    """After the transplant the model's class i is our category_id i+1."""
    return {i: i + 1 for i in range(len(CLASSES))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--models", nargs="*",
                        default=["ssd300_vgg16", "yolov8m", "rtdetr-l"])
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    results = {}

    if "ssd300_vgg16" in args.models:
        results["ssd300_vgg16"] = check_ssd(args.split, args.batch, device)
    for name in ("yolov8m", "rtdetr-l"):
        if name in args.models:
            results[name] = check_ultralytics(name, args.split, args.batch,
                                              args.device)

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    for name, r in results.items():
        status = "OK" if r["match"] else "CHECK"
        print(f"  {name:<16} {r['before']['mAP']:.4f} -> {r['after']['mAP']:.4f}"
              f"   {status}")
    print("=" * 74)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"warmstart_verification_{args.split}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Written to {out}")
    return 0 if all(r["match"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
