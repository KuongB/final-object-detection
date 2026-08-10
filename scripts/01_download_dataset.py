"""Step 2a - download the COCO-2017 subset containing our 5 classes via FiftyOne
and dump it into a canonical, framework-agnostic manifest.

Why a manifest instead of exporting directly?
  * FiftyOne downloads are slow and network-bound; the export step is fast and
    gets re-run often (different splits, formats, fixes). Separating them means
    a mistake in the export costs seconds, not another 30-minute download.
  * The manifest is a plain JSON file we can inspect, diff and verify - which
    matters here because some FiftyOne versions are known to mis-apply the
    `classes=` filter. This script verifies the filter actually worked instead
    of trusting it.

Usage
-----
    python scripts/01_download_dataset.py                 # full download
    python scripts/01_download_dataset.py --max-samples 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASSES, DATA_DIR, MANIFEST_PATH, RESULTS_DIR  # noqa: E402

# FiftyOne source split -> the role it plays in our project
SOURCE_SPLITS = {
    "train": "coco_train",  # -> later split into our train + val
    "validation": "coco_val",  # -> our held-out test set
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def find_detections_field(dataset) -> str:
    """Return the name of the sample field holding `fo.Detections`.

    The zoo dataset calls it `ground_truth`, but that has changed between
    FiftyOne releases, so we discover it rather than hard-coding it.
    """
    import fiftyone as fo

    for name, field in dataset.get_field_schema().items():
        doc_type = getattr(field, "document_type", None)
        if doc_type is not None and issubclass(doc_type, fo.Detections):
            return name
    raise RuntimeError(
        f"No fo.Detections field found. Schema: {list(dataset.get_field_schema())}"
    )


def detection_attr(det, name, default=None):
    """Read an extra COCO attribute (`iscrowd`, `area`) off a Detection.

    FiftyOne has moved these between dynamic fields and the legacy `attributes`
    dict across versions, so try both.
    """
    try:
        if det.has_field(name):
            value = det.get_field(name)
            if value is not None:
                return value
    except Exception:  # noqa: BLE001
        pass
    try:
        attrs = det.get_field("attributes") or {}
        if name in attrs:
            return attrs[name].value
    except Exception:  # noqa: BLE001
        pass
    return default


def reset_zoo_bookkeeping() -> None:
    """Drop FiftyOne's record of *what* has been downloaded, keeping the images.

    FiftyOne decides a split is complete from `<zoo>/coco-2017/info.json`
    (`downloaded_splits[split].num_samples`) plus the per-split `labels.json`,
    not from the files on disk. So a previous `--max-samples 30` run - or images
    fetched out-of-band by `01b_fetch_images.py` - leaves the bookkeeping and
    the data directory disagreeing, and FiftyOne reports "existing download is
    sufficient" while loading only the 30 samples it knows about.

    Deleting these two files makes FiftyOne rebuild its index from the raw
    annotations and the images actually present. The images themselves live in
    `<split>/data/` and are never touched, so nothing is re-downloaded.
    """
    import fiftyone as fo

    zoo_root = Path(fo.config.dataset_zoo_dir) / "coco-2017"
    removed = []
    for target in (
        zoo_root / "info.json",
        *(zoo_root / split / "labels.json" for split in SOURCE_SPLITS),
    ):
        if target.is_file():
            backup = target.with_suffix(".json.bak")
            backup.unlink(missing_ok=True)
            target.rename(backup)
            removed.append(str(target.relative_to(zoo_root)))

    print(f"Reset zoo bookkeeping in {zoo_root}")
    print(f"  moved aside: {removed or 'nothing found'}")
    for split in SOURCE_SPLITS:
        data_dir = zoo_root / split / "data"
        n = len(list(data_dir.glob("*.jpg"))) if data_dir.is_dir() else 0
        print(f"  {split:<12} {n:>7,} image files kept on disk")


def coco_id_from_filename(path: Path, fallback: int) -> int:
    """COCO files are named `000000123456.jpg`; keep the original id so results
    stay traceable back to upstream COCO."""
    try:
        return int(path.stem)
    except ValueError:
        return fallback


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def load_split(
    fo_split: str,
    max_samples: int | None,
    attempts: int = 12,
    num_workers: int = 4,
):
    """Download one COCO split, retrying on transient network failures.

    FiftyOne fetches images with a multiprocessing pool that has no retry: a
    single aborted connection (WinError 10053 is common on Windows over a few
    thousand requests) propagates out of `imap_unordered` and kills the whole
    download. Already-downloaded images are kept on disk though, and FiftyOne
    only re-requests the missing ids, so re-calling it simply resumes.
    """
    import fiftyone as fo
    import fiftyone.zoo as foz

    print(f"\n{'=' * 78}")
    print(f"Downloading coco-2017 / {fo_split}  (classes={CLASSES})")
    print("=" * 78)

    dataset_name = f"coco2017-fruitveg-{fo_split}" + (
        f"-{max_samples}" if max_samples else ""
    )

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            # A partially-created dataset from a failed attempt would make
            # FiftyOne refuse the name on the retry.
            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)

            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split=fo_split,
                label_types=["detections"],
                classes=list(CLASSES),
                only_matching=True,  # drop labels of the other 75 COCO classes
                max_samples=max_samples,
                num_workers=num_workers,
                dataset_name=dataset_name,
            )
            print(
                f"  loaded {len(dataset)} samples into FiftyOne dataset "
                f"'{dataset.name}' (attempt {attempt})"
            )
            return dataset
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = min(30, 5 * attempt)
            print(
                f"\n  !! attempt {attempt}/{attempts} failed: "
                f"{type(exc).__name__}: {exc}"
                f"\n  !! images already fetched are kept; retrying in {wait}s "
                "(download resumes where it stopped)\n",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Download of coco-2017/{fo_split} failed after {attempts} attempts"
    ) from last_error


def extract_images(dataset, source_split: str) -> tuple[list[dict], dict]:
    """Turn a FiftyOne dataset into plain dicts, keeping only our 5 classes."""
    det_field = find_detections_field(dataset)
    print(f"  detections field: '{det_field}'")

    print("  computing image metadata (width/height)...")
    dataset.compute_metadata()

    allowed = set(CLASSES)
    images: list[dict] = []
    stats = {
        "samples_seen": len(dataset),
        "labels_total": 0,
        "labels_kept": 0,
        "labels_dropped_other_class": 0,
        "labels_dropped_invalid_box": 0,
        "images_dropped_no_label": 0,
        "images_dropped_no_metadata": 0,
        "images_dropped_missing_file": 0,
        "dropped_class_names": Counter(),
    }

    for i, sample in enumerate(dataset.iter_samples(progress=True)):
        src_path = Path(sample.filepath)
        meta = sample.metadata
        if meta is None or not meta.width or not meta.height:
            stats["images_dropped_no_metadata"] += 1
            continue
        if not src_path.is_file():
            stats["images_dropped_missing_file"] += 1
            continue

        width, height = int(meta.width), int(meta.height)
        detections = getattr(sample, det_field)
        det_list = detections.detections if detections is not None else []
        stats["labels_total"] += len(det_list)

        annotations = []
        for det in det_list:
            if det.label not in allowed:
                stats["labels_dropped_other_class"] += 1
                stats["dropped_class_names"][det.label] += 1
                continue

            rx, ry, rw, rh = det.bounding_box  # relative [x, y, w, h]
            x, y = rx * width, ry * height
            w, h = rw * width, rh * height

            # Clip to the image and reject degenerate boxes: a handful of COCO
            # boxes extend past the image border or have zero extent, and they
            # break both COCOeval and the YOLO loss.
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(width), x + w), min(float(height), y + h)
            if x1 - x0 < 1.0 or y1 - y0 < 1.0:
                stats["labels_dropped_invalid_box"] += 1
                continue

            iscrowd = int(float(detection_attr(det, "iscrowd", 0) or 0))
            area = detection_attr(det, "area", None)
            area = float(area) if area is not None else (x1 - x0) * (y1 - y0)

            annotations.append(
                {
                    "class": det.label,
                    "bbox": [
                        round(x0, 2),
                        round(y0, 2),
                        round(x1 - x0, 2),
                        round(y1 - y0, 2),
                    ],
                    "area": round(area, 2),
                    "iscrowd": iscrowd,
                }
            )

        if not annotations:
            stats["images_dropped_no_label"] += 1
            continue

        stats["labels_kept"] += len(annotations)
        images.append(
            {
                "coco_id": coco_id_from_filename(src_path, fallback=10_000_000 + i),
                "file_name": src_path.name,
                "source_path": str(src_path),
                "source_split": source_split,
                "width": width,
                "height": height,
                "annotations": annotations,
            }
        )

    stats["dropped_class_names"] = dict(stats["dropped_class_names"])
    stats["images_kept"] = len(images)
    return images, stats


def print_stats(source_split: str, images: list[dict], stats: dict) -> None:
    print(f"\n--- {source_split} ---")
    for key in (
        "samples_seen",
        "images_kept",
        "images_dropped_no_label",
        "images_dropped_no_metadata",
        "images_dropped_missing_file",
        "labels_total",
        "labels_kept",
        "labels_dropped_other_class",
        "labels_dropped_invalid_box",
    ):
        print(f"  {key:<32} {stats[key]:>8,}")

    if stats["dropped_class_names"]:
        print("\n  !! FiftyOne returned labels outside our class list "
              "(only_matching filter leaked) - they were removed here:")
        for name, count in sorted(
            stats["dropped_class_names"].items(), key=lambda kv: -kv[1]
        )[:15]:
            print(f"     {name:<24} {count:>7,}")

    # per-class verification: this is the check that catches the FiftyOne bug
    img_count = Counter()
    inst_count = Counter()
    crowd_count = Counter()
    for img in images:
        present = set()
        for ann in img["annotations"]:
            inst_count[ann["class"]] += 1
            crowd_count[ann["class"]] += ann["iscrowd"]
            present.add(ann["class"])
        for name in present:
            img_count[name] += 1

    print(f"\n  {'class':<12} {'images':>9} {'instances':>11} {'iscrowd':>9}")
    print(f"  {'-' * 12} {'-' * 9} {'-' * 11} {'-' * 9}")
    for name in CLASSES:
        print(
            f"  {name:<12} {img_count[name]:>9,} {inst_count[name]:>11,} "
            f"{crowd_count[name]:>9,}"
        )
    print(
        f"  {'TOTAL':<12} {len(images):>9,} {sum(inst_count.values()):>11,} "
        f"{sum(crowd_count.values()):>9,}"
    )

    missing = [c for c in CLASSES if inst_count[c] == 0]
    if missing:
        print(f"\n  !! WARNING: no instances found for {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="limit samples per split (smoke test); omit to download everything",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=12,
        help="retries per split on network failure (each retry resumes)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="parallel download workers; lower is slower but more stable",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help=(
            "discard FiftyOne's record of what has been downloaded (images are "
            "kept) and rebuild it from the files on disk - needed after a "
            "--max-samples run or after 01b_fetch_images.py"
        ),
    )
    args = parser.parse_args()

    if args.reset_cache:
        reset_zoo_bookkeeping()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "classes": list(CLASSES),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "coco-2017 via fiftyone",
        "max_samples": args.max_samples,
        "stats": {},
        "images": [],
    }

    all_stats = {}
    for fo_split, source_split in SOURCE_SPLITS.items():
        dataset = load_split(
            fo_split,
            args.max_samples,
            attempts=args.attempts,
            num_workers=args.num_workers,
        )
        images, stats = extract_images(dataset, source_split)
        print_stats(source_split, images, stats)
        manifest["images"].extend(images)
        manifest["stats"][source_split] = stats
        all_stats[source_split] = (images, stats)

    # Guard against the two subsets overlapping - our test set must be clean.
    ids_by_source = defaultdict(set)
    for img in manifest["images"]:
        ids_by_source[img["source_split"]].add(img["coco_id"])
    overlap = ids_by_source["coco_train"] & ids_by_source["coco_val"]
    print(f"\n{'=' * 78}")
    print(f"Overlap between coco_train and coco_val image ids: {len(overlap)}")
    if overlap:
        print(f"  !! WARNING: {len(overlap)} shared ids, test set is contaminated")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    size_mb = MANIFEST_PATH.stat().st_size / 1024**2
    print(f"Manifest written: {MANIFEST_PATH}  ({size_mb:.1f} MB)")
    print(f"Total images: {len(manifest['images']):,}")
    print(
        "Total instances: "
        f"{sum(len(i['annotations']) for i in manifest['images']):,}"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
