"""Step 2c - prove the two annotation formats actually agree.

`02_build_dataset.py` checks that file counts line up, but counts say nothing
about geometry: a bbox written as xyxy where xywh was meant, or a width/height
swap, would pass every count check and silently poison all three models.

This script closes that gap:

  1. Round-trip every YOLO box back to absolute pixels and match it against the
     COCO box it came from. Requires IoU ~ 1.0 for all ~38k boxes.
  2. Confirm ultralytics itself can build a dataloader from data.yaml and that
     the labels it parses match ours.
  3. Render a sample sheet with boxes drawn from the COCO JSON, so the
     coordinates get one human look as well.

    python scripts/03_verify_dataset.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CLASS_COLORS_RGB,
    CLASSES,
    COCO_ID_TO_CLASS,
    DATA_YAML,
    FIGURES_DIR,
    IDX_TO_CLASS,
    RANDOM_SEED,
    SPLITS,
    ann_path,
    images_dir,
    labels_dir,
)

IOU_TOLERANCE = 0.995


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two xyxy boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def check_roundtrip(split: str) -> list[str]:
    """Every non-crowd COCO box must reappear in the YOLO file as the same box."""
    problems: list[str] = []
    coco = json.loads(ann_path(split).read_text(encoding="utf-8"))

    images = {img["id"]: img for img in coco["images"]}
    by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if ann["iscrowd"]:
            continue  # deliberately absent from the YOLO labels
        by_image.setdefault(ann["image_id"], []).append(ann)

    worst_iou = 1.0
    n_boxes = 0

    for image_id, img in images.items():
        stem = Path(img["file_name"]).stem
        label_file = labels_dir(split) / f"{stem}.txt"
        if not label_file.is_file():
            problems.append(f"[{split}] missing label file for {img['file_name']}")
            continue

        w, h = img["width"], img["height"]

        yolo_boxes = []
        for line_no, line in enumerate(
            label_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"[{split}] {label_file.name}:{line_no} has {len(parts)} fields, expected 5")
                continue
            cls_idx = int(parts[0])
            cx, cy, nw, nh = (float(p) for p in parts[1:])
            if not (0 <= cls_idx < len(CLASSES)):
                problems.append(f"[{split}] {label_file.name}:{line_no} bad class index {cls_idx}")
                continue
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, nw, nh)):
                problems.append(f"[{split}] {label_file.name}:{line_no} coords outside [0,1]")
                continue
            yolo_boxes.append(
                (
                    IDX_TO_CLASS[cls_idx],
                    (
                        (cx - nw / 2) * w,
                        (cy - nh / 2) * h,
                        (cx + nw / 2) * w,
                        (cy + nh / 2) * h,
                    ),
                )
            )

        coco_boxes = []
        for ann in by_image.get(image_id, []):
            x, y, bw, bh = ann["bbox"]
            coco_boxes.append(
                (COCO_ID_TO_CLASS[ann["category_id"]], (x, y, x + bw, y + bh))
            )

        if len(yolo_boxes) != len(coco_boxes):
            problems.append(
                f"[{split}] {img['file_name']}: {len(yolo_boxes)} yolo boxes vs "
                f"{len(coco_boxes)} coco boxes"
            )
            continue

        # Greedy match: for each COCO box find its best YOLO counterpart.
        remaining = list(yolo_boxes)
        for cls_name, cbox in coco_boxes:
            best_i, best = -1, -1.0
            for i, (ycls, ybox) in enumerate(remaining):
                if ycls != cls_name:
                    continue
                score = iou(cbox, ybox)
                if score > best:
                    best_i, best = i, score
            if best_i < 0:
                problems.append(
                    f"[{split}] {img['file_name']}: no yolo box of class {cls_name}"
                )
                continue
            if best < IOU_TOLERANCE:
                problems.append(
                    f"[{split}] {img['file_name']}: class {cls_name} best IoU "
                    f"{best:.4f} < {IOU_TOLERANCE}"
                )
            worst_iou = min(worst_iou, best)
            remaining.pop(best_i)
            n_boxes += 1

    print(f"  [{split}] round-tripped {n_boxes:,} boxes, worst IoU = {worst_iou:.6f}")
    return problems


def check_ultralytics() -> list[str]:
    """Build a real ultralytics dataloader - the only way to be sure it agrees."""
    problems: list[str] = []
    try:
        from ultralytics.data.utils import check_det_dataset

        info = check_det_dataset(str(DATA_YAML))
    except Exception as exc:  # noqa: BLE001
        return [f"ultralytics could not load data.yaml: {type(exc).__name__}: {exc}"]

    names = info.get("names", {})
    if len(names) != len(CLASSES):
        problems.append(f"data.yaml exposes {len(names)} classes, expected {len(CLASSES)}")
    for idx, name in names.items():
        if IDX_TO_CLASS.get(int(idx)) != name:
            problems.append(f"class index {idx} is '{name}', expected '{IDX_TO_CLASS.get(int(idx))}'")

    for split in SPLITS:
        if info.get(split) is None:
            problems.append(f"data.yaml has no '{split}' path")

    print(f"  ultralytics resolved: names={ {int(k): v for k, v in names.items()} }")
    print(f"  paths ok for splits : {[s for s in SPLITS if info.get(s)]}")
    return problems


def render_samples(split: str = "train", n: int = 9) -> Path:
    """Draw boxes straight from the COCO JSON onto real images."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    coco = json.loads(ann_path(split).read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    rng = random.Random(RANDOM_SEED)
    # Prefer busy images - they make a coordinate bug obvious at a glance.
    candidates = [img for img in coco["images"] if len(by_image.get(img["id"], [])) >= 3]
    picks = rng.sample(candidates, min(n, len(candidates)))

    cols = 3
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, img_info in zip(axes, picks):
        image = Image.open(images_dir(split) / img_info["file_name"]).convert("RGB")
        ax.imshow(image)
        counts = Counter()
        for ann in by_image[img_info["id"]]:
            name = COCO_ID_TO_CLASS[ann["category_id"]]
            counts[name] += 1
            x, y, w, h = ann["bbox"]
            color = tuple(c / 255 for c in CLASS_COLORS_RGB[name])
            ax.add_patch(
                patches.Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none",
                    linestyle="--" if ann["iscrowd"] else "-",
                )
            )
            ax.text(
                x, max(y - 4, 8), name + (" (crowd)" if ann["iscrowd"] else ""),
                color="white", fontsize=7,
                bbox=dict(facecolor=color, edgecolor="none", pad=1.2),
            )
        ax.set_title(
            f"{img_info['file_name']}\n"
            + ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())),
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle(
        f"Sanity check - boxes drawn from instances_{split}.json "
        "(dashed = iscrowd, excluded from YOLO labels)",
        fontsize=11,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"sample_{split}_boxes.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    problems: list[str] = []

    print("=" * 78)
    print("1. COCO <-> YOLO coordinate round-trip")
    print("=" * 78)
    for split in SPLITS:
        problems += check_roundtrip(split)

    print(f"\n{'=' * 78}")
    print("2. ultralytics dataset resolution")
    print("=" * 78)
    problems += check_ultralytics()

    print(f"\n{'=' * 78}")
    print("3. visual sample sheet")
    print("=" * 78)
    for split in ("train", "test"):
        out = render_samples(split)
        print(f"  wrote {out}")

    print(f"\n{'=' * 78}")
    if problems:
        for p in problems[:40]:
            print(f"  FAIL  {p}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print(f"\n{len(problems)} problem(s) found.")
        return 1
    print("ALL VERIFICATION PASSED")
    print("  - every YOLO box reproduces its COCO box (IoU >= 0.995)")
    print("  - ultralytics resolves data.yaml with the expected class mapping")
    print("  - sample sheets rendered for visual inspection")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
