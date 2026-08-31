"""Turn `data/manifest.json` into the on-disk dataset the three models read.

The manifest is the raw harvest from FiftyOne: one record per image, with its
source path and every annotation, still in absolute COCO `xywh` pixels. This
module is what converts that into the two layouts training actually needs:

    data/images/<split>/<id>.jpg          shared image tree (hard-linked)
    data/labels/<split>/<id>.txt          YOLO - normalised cxcywh, 0-indexed
    data/annotations/instances_<s>.json   COCO - absolute xywh, 1-indexed
    data/data.yaml                        ultralytics entry point
    data/splits.json                      the split itself, as plain ids

Both label formats are written from the *same* in-memory records in the same
pass, which is what makes the two mathematically consistent - `scripts/
03_verify_dataset.py` then proves it by round-tripping every box.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    CLASS_TO_COCO_ID,
    CLASS_TO_IDX,
    CLASSES,
    DATA_YAML,
    IMAGES_DIR,
    RANDOM_SEED,
    SPLITS,
    SPLITS_INDEX_PATH,
    VAL_FRACTION,
    ann_path,
    images_dir,
    labels_dir,
)

# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def _class_signature(img: dict) -> tuple[str, ...]:
    """The set of classes present in an image, as a hashable key.

    Stratifying on this - rather than on a single 'primary' class - keeps
    multi-class images (a fruit bowl with apples *and* oranges) balanced
    across train and val instead of piling into whichever split got lucky.
    """
    return tuple(sorted({ann["class"] for ann in img["annotations"]}))


def _stratified_val_split(
    images: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split `images` into (train, val), preserving the class mix."""
    buckets: dict[tuple[str, ...], list[dict]] = {}
    for img in images:
        buckets.setdefault(_class_signature(img), []).append(img)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []

    for signature in sorted(buckets):  # sorted -> deterministic bucket order
        group = sorted(buckets[signature], key=lambda i: i["coco_id"])
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        # A signature seen only once or twice must still contribute to train.
        n_val = min(n_val, len(group) - 1) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    train.sort(key=lambda i: i["coco_id"])
    val.sort(key=lambda i: i["coco_id"])
    return train, val


def build_splits(manifest: dict) -> dict[str, list[dict]]:
    """Assign every manifest image to train / val / test.

    `test` is fixed by provenance, never by a random draw: it is exactly the
    COCO *val2017* portion of the harvest, so it stays an untouched held-out
    set from a distribution the model never trained on. Only the COCO
    *train2017* portion is split into train and val.

    If `data/splits.json` already exists its assignment is reused verbatim.
    That is what makes a rebuild reproduce the *same* dataset rather than
    merely a statistically similar one - re-running this module can never
    silently leak a previously-validated image into training.
    """
    images = manifest["images"]
    by_id = {img["coco_id"]: img for img in images}

    if SPLITS_INDEX_PATH.is_file():
        recorded = json.loads(SPLITS_INDEX_PATH.read_text(encoding="utf-8"))
        assignment = recorded.get("assignment", {})
        if all(split in assignment for split in SPLITS):
            known = {cid for ids in assignment.values() for cid in ids}
            # Only honour the recorded split if it still describes this
            # manifest; otherwise it is stale and we re-derive from scratch.
            if known == set(by_id):
                print(f"  reusing recorded split from {SPLITS_INDEX_PATH.name}")
                return {
                    split: [by_id[cid] for cid in sorted(assignment[split])]
                    for split in SPLITS
                }
            print(
                f"  {SPLITS_INDEX_PATH.name} does not match the manifest "
                f"({len(known):,} ids recorded vs {len(by_id):,} present) "
                "- deriving a fresh split"
            )

    pool = [img for img in images if img["source_split"] == "coco_train"]
    test = [img for img in images if img["source_split"] == "coco_val"]
    train, val = _stratified_val_split(pool, VAL_FRACTION, RANDOM_SEED)
    test.sort(key=lambda i: i["coco_id"])
    return {"train": train, "val": val, "test": test}


# --------------------------------------------------------------------------- #
# Image tree
# --------------------------------------------------------------------------- #


def link_images(split: str, images: list[dict]) -> str:
    """Materialise `data/images/<split>/`.

    Hard links keep a second copy of ~3 GB of JPEGs off the disk; when the
    filesystem refuses (different volume, or a FAT/exFAT target) we fall back
    to a real copy so the dataset is still self-contained.
    """
    out = images_dir(split)
    out.mkdir(parents=True, exist_ok=True)

    linked = copied = skipped = missing = 0
    for img in images:
        source = Path(img["source_path"])
        dest = out / img["file_name"]
        if dest.exists():
            skipped += 1
            continue
        if not source.is_file():
            missing += 1
            continue
        try:
            os.link(source, dest)
            linked += 1
        except OSError:
            shutil.copy2(source, dest)
            copied += 1

    parts = [f"{linked:,} linked", f"{copied:,} copied", f"{skipped:,} already present"]
    if missing:
        parts.append(f"{missing:,} MISSING AT SOURCE")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Label writers
# --------------------------------------------------------------------------- #


def write_yolo_labels(split: str, images: list[dict]) -> str:
    """Write one `.txt` per image: `cls cx cy w h`, all normalised to [0, 1].

    `iscrowd` regions are skipped. A crowd box marks "a pile of oranges
    somewhere in here" rather than one object; feeding it to a detector that
    has no crowd concept teaches it a box it can never reproduce.
    """
    out = labels_dir(split)
    out.mkdir(parents=True, exist_ok=True)

    n_boxes = n_crowd = n_clamped = 0
    for img in images:
        width, height = img["width"], img["height"]
        lines: list[str] = []
        for ann in img["annotations"]:
            if ann["iscrowd"]:
                n_crowd += 1
                continue
            x, y, w, h = ann["bbox"]

            # Clamp to the image before normalising: a handful of COCO boxes
            # overhang the border by a pixel or two, which would otherwise
            # emit a coordinate > 1.0 and trip every loader's validation.
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(width), x + w), min(float(height), y + h)
            if (x0, y0, x1, y1) != (x, y, x + w, y + h):
                n_clamped += 1
            if x1 <= x0 or y1 <= y0:
                continue

            cx = ((x0 + x1) / 2) / width
            cy = ((y0 + y1) / 2) / height
            bw = (x1 - x0) / width
            bh = (y1 - y0) / height
            lines.append(
                f"{CLASS_TO_IDX[ann['class']]} "
                f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )
            n_boxes += 1

        (out / f"{Path(img['file_name']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    return (
        f"{len(images):,} files, {n_boxes:,} boxes "
        f"({n_crowd:,} iscrowd skipped, {n_clamped:,} clamped to border)"
    )


def write_coco_json(split: str, images: list[dict]) -> Path:
    """Write `instances_<split>.json` with our compact 1..5 category ids.

    Crowd annotations are *kept* here - `COCOeval` knows to exclude them from
    the recall denominator, which is precisely the behaviour we want when
    scoring. That is why the COCO file legitimately holds more boxes than the
    YOLO labels do.
    """
    out = ann_path(split)
    out.parent.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict] = []
    coco_anns: list[dict] = []
    ann_id = 1

    for img in images:
        coco_images.append(
            {
                "id": img["coco_id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }
        )
        for ann in img["annotations"]:
            x, y, w, h = ann["bbox"]
            coco_anns.append(
                {
                    "id": ann_id,
                    "image_id": img["coco_id"],
                    "category_id": CLASS_TO_COCO_ID[ann["class"]],
                    "bbox": [round(v, 2) for v in (x, y, w, h)],
                    "area": round(ann.get("area", w * h), 2),
                    "iscrowd": int(ann["iscrowd"]),
                    "segmentation": [],
                }
            )
            ann_id += 1

    payload = {
        "info": {
            "description": "Fruit & vegetable detection - COCO 2017 subset",
            "classes": list(CLASSES),
            "split": split,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "licenses": [],
        "images": coco_images,
        "categories": [
            {"id": CLASS_TO_COCO_ID[name], "name": name, "supercategory": "food"}
            for name in CLASSES
        ],
        "annotations": coco_anns,
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def write_data_yaml(data_dir: str | Path | None = None) -> Path:
    """Write the ultralytics dataset descriptor.

    `path` is absolute because ultralytics resolves the split paths relative
    to it, and a relative root breaks the moment training is launched from a
    different working directory (which notebooks always do).
    """
    root = Path(data_dir).resolve() if data_dir else IMAGES_DIR.parent
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(
        "# Ultralytics dataset config - generated by src/data/build.py\n"
        "# Regenerate for another machine with:\n"
        '#   python -c "from src.data.build import write_data_yaml; '
        "write_data_yaml('/path/to/data')\"\n"
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    return DATA_YAML


def write_splits_index(splits: dict[str, list[dict]]) -> Path:
    """Record the split as bare COCO ids - the file `build_splits` reads back."""
    payload = {
        "seed": RANDOM_SEED,
        "val_fraction": VAL_FRACTION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assignment": {
            split: [img["coco_id"] for img in images] for split, images in splits.items()
        },
    }
    SPLITS_INDEX_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return SPLITS_INDEX_PATH


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def split_stats(images: list[dict]) -> dict:
    """Per-split counts used by the report, the EDA notebook and the CLI table."""
    per_class = Counter()
    n_instances = n_crowd = 0

    for img in images:
        for ann in img["annotations"]:
            if ann["iscrowd"]:
                n_crowd += 1
                continue
            per_class[ann["class"]] += 1
            n_instances += 1

    return {
        "n_images": len(images),
        "n_instances": n_instances,
        "n_iscrowd": n_crowd,
        "objects_per_image": n_instances / len(images) if images else 0.0,
        "instances_per_class": {name: per_class[name] for name in CLASSES},
    }
