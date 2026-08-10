"""Step 2b - build the final dataset from the manifest produced by
`01_download_dataset.py`.

  manifest.json ->  train / val / test split
                ->  data/images/<split>/            (shared image tree)
                ->  data/labels/<split>/*.txt       (YOLO)
                ->  data/annotations/*.json         (COCO)
                ->  data/data.yaml                  (ultralytics)

Everything here is deterministic (seed 42) and cheap to re-run.

Usage
-----
    python scripts/02_build_dataset.py
    python scripts/02_build_dataset.py --clean      # rebuild from scratch
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CLASSES,
    DATA_DIR,
    DATA_YAML,
    IMAGES_DIR,
    LABELS_DIR,
    MANIFEST_PATH,
    RESULTS_DIR,
    SPLITS,
    ann_path,
    images_dir,
    labels_dir,
)
from src.data.build import (  # noqa: E402
    build_splits,
    link_images,
    split_stats,
    write_coco_json,
    write_data_yaml,
    write_splits_index,
    write_yolo_labels,
)


def drop_crowd_only_images(images: list[dict]) -> tuple[list[dict], int]:
    """Remove images whose every annotation is `iscrowd`.

    Such an image would carry no YOLO label at all, so the model would be
    trained to see a pile of oranges as background - actively harmful. Dropping
    them from BOTH formats also keeps the two image sets identical.
    """
    kept, dropped = [], 0
    for img in images:
        if all(ann["iscrowd"] for ann in img["annotations"]):
            dropped += 1
        else:
            kept.append(img)
    return kept, dropped


def verify(splits: dict[str, list[dict]]) -> list[str]:
    """Cross-check what actually landed on disk. Returns a list of problems."""
    problems: list[str] = []

    for split, images in splits.items():
        expected = {img["file_name"] for img in images}
        on_disk = {p.name for p in images_dir(split).glob("*.jpg")}
        if missing := expected - on_disk:
            problems.append(f"[{split}] {len(missing)} image files missing on disk")
        if extra := on_disk - expected:
            problems.append(f"[{split}] {len(extra)} stray image files on disk")

        expected_lbl = {Path(f).stem + ".txt" for f in expected}
        lbl_on_disk = {p.name for p in labels_dir(split).glob("*.txt")}
        if missing := expected_lbl - lbl_on_disk:
            problems.append(f"[{split}] {len(missing)} label files missing")
        if extra := lbl_on_disk - expected_lbl:
            problems.append(f"[{split}] {len(extra)} stray label files")

        # COCO JSON must be loadable by the exact library used for evaluation.
        try:
            from pycocotools.coco import COCO

            coco = COCO(str(ann_path(split)))
            if len(coco.imgs) != len(images):
                problems.append(
                    f"[{split}] COCO json has {len(coco.imgs)} images, "
                    f"expected {len(images)}"
                )
            if set(coco.getCatIds()) != set(range(1, len(CLASSES) + 1)):
                problems.append(f"[{split}] unexpected category ids {coco.getCatIds()}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"[{split}] pycocotools failed to load: {exc}")

    # No image may appear in more than one split.
    seen: dict[int, str] = {}
    for split, images in splits.items():
        for img in images:
            if (other := seen.get(img["coco_id"])) is not None:
                problems.append(
                    f"image {img['coco_id']} appears in both {other} and {split}"
                )
            seen[img["coco_id"]] = split

    return problems


def print_distribution(stats: dict[str, dict]) -> None:
    print(f"\n{'=' * 88}")
    print("PER-SPLIT SUMMARY")
    print("=" * 88)
    print(f"{'split':<8}{'images':>10}{'instances':>12}{'iscrowd':>10}{'obj/img':>10}{'% images':>10}")
    print("-" * 88)
    total_images = sum(s["n_images"] for s in stats.values())
    for split in SPLITS:
        s = stats[split]
        pct = 100 * s["n_images"] / total_images if total_images else 0
        print(
            f"{split:<8}{s['n_images']:>10,}{s['n_instances']:>12,}"
            f"{s['n_iscrowd']:>10,}{s['objects_per_image']:>10.2f}{pct:>9.1f}%"
        )
    print("-" * 88)
    print(
        f"{'TOTAL':<8}{total_images:>10,}"
        f"{sum(s['n_instances'] for s in stats.values()):>12,}"
    )

    print(f"\n{'=' * 88}")
    print("INSTANCES PER CLASS  (share within split - these should stay close)")
    print("=" * 88)
    header = f"{'class':<12}"
    for split in SPLITS:
        header += f"{split + ' n':>10}{split + ' %':>9}"
    print(header)
    print("-" * 88)
    for name in CLASSES:
        row = f"{name:<12}"
        for split in SPLITS:
            n = stats[split]["instances_per_class"][name]
            total = stats[split]["n_instances"]
            row += f"{n:>10,}{100 * n / total if total else 0:>8.1f}%"
        print(row)

    # Largest train-vs-val drift tells us whether stratification worked.
    drift = max(
        abs(
            100 * stats["train"]["instances_per_class"][c] / stats["train"]["n_instances"]
            - 100 * stats["val"]["instances_per_class"][c] / stats["val"]["n_instances"]
        )
        for c in CLASSES
    )
    print(f"\nmax train-vs-val class share drift: {drift:.2f} percentage points")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete existing images/ and labels/ before rebuilding",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.is_file():
        print(f"ERROR: {MANIFEST_PATH} not found. Run 01_download_dataset.py first.")
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    images = manifest["images"]
    print(f"Manifest: {len(images):,} images, "
          f"{sum(len(i['annotations']) for i in images):,} instances")

    images, n_crowd_only = drop_crowd_only_images(images)
    print(f"Dropped {n_crowd_only} images whose annotations are all iscrowd")
    manifest["images"] = images

    if args.clean:
        for d in (IMAGES_DIR, LABELS_DIR):
            if d.exists():
                print(f"Removing {d}")
                shutil.rmtree(d)

    splits = build_splits(manifest)
    print(
        "\nSplit sizes: "
        + ", ".join(f"{s}={len(splits[s]):,}" for s in SPLITS)
    )

    stats: dict[str, dict] = {}
    for split in SPLITS:
        imgs = splits[split]
        link = link_images(split, imgs)
        yolo = write_yolo_labels(split, imgs)
        coco_path = write_coco_json(split, imgs)
        stats[split] = split_stats(imgs)
        print(
            f"\n[{split}] images: {link}"
            f"\n[{split}] yolo  : {yolo}"
            f"\n[{split}] coco  : {coco_path.name} "
            f"({coco_path.stat().st_size / 1024**2:.1f} MB)"
        )

    yaml_path = write_data_yaml()
    splits_path = write_splits_index(splits)
    print(f"\nWrote {yaml_path}")
    print(f"Wrote {splits_path}")

    print_distribution(stats)

    # Persist the numbers for the report / EDA notebook.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = RESULTS_DIR / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"\nWrote {stats_path}")

    print(f"\n{'=' * 88}")
    print("VERIFICATION")
    print("=" * 88)
    problems = verify(splits)
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        print(f"\n{len(problems)} problem(s) found.")
        return 1
    print("  All checks passed: image/label counts match, COCO json loads in")
    print("  pycocotools, category ids are 1..5, no image shared between splits.")

    size_gb = sum(
        f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file()
    ) / 1024**3
    print(f"\ndata/ on disk: {size_gb:.2f} GB")
    print(f"Ultralytics config: {DATA_YAML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
