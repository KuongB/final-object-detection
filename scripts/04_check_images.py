"""Step 2d - find and repair truncated or corrupt image files.

Why this exists: `01b_fetch_images.py` streams each image to disk and renames it
into place, but it never compared the number of bytes written against the
response's Content-Length. An HTTP response cut short mid-transfer therefore
lands as a complete-looking JPEG that PIL refuses to decode:

    OSError: image file is truncated (N bytes not processed)

PIL only raises this on full decode, so nothing before the first training epoch
notices. This script forces a full decode of every image, then re-downloads the
bad ones - verifying the length this time.

Both the project copy (`data/images/<split>/`) and the FiftyOne cache copy are
repaired, because the two are hard-linked to the same bytes.

    python scripts/04_check_images.py            # scan only
    python scripts/04_check_images.py --repair   # scan, then re-download bad files
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, SPLITS, images_dir  # noqa: E402

FIFTYONE_COCO = Path.home() / "fiftyone" / "coco-2017"
SOURCE_DIRS = {
    "train": FIFTYONE_COCO / "train" / "data",
    "val": FIFTYONE_COCO / "train" / "data",  # our val also comes from coco train
    "test": FIFTYONE_COCO / "validation" / "data",
}
URL_PREFIX = {"train": "train2017", "val": "train2017", "test": "val2017"}


def check_image(path: Path) -> str | None:
    """Fully decode an image. Returns an error string, or None when healthy.

    `Image.verify()` is not enough - it only checks the header. Truncation is
    only caught by decoding the pixel data, which is what `load()` does.
    """
    from PIL import Image

    try:
        with Image.open(path) as img:
            img.load()
            if img.size[0] < 2 or img.size[1] < 2:
                return f"degenerate size {img.size}"
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def scan(splits=SPLITS) -> dict[str, list[tuple[Path, str]]]:
    broken: dict[str, list[tuple[Path, str]]] = {}
    for split in splits:
        files = sorted(images_dir(split).glob("*.jpg"))
        bad: list[tuple[Path, str]] = []
        print(f"\n[{split}] checking {len(files):,} images...", flush=True)
        for i, path in enumerate(files, 1):
            error = check_image(path)
            if error:
                bad.append((path, error))
                print(f"  BAD  {path.name}  {error}")
            if i % 1000 == 0:
                print(f"  ...{i:,}/{len(files):,}  ({len(bad)} bad so far)", flush=True)
        print(f"[{split}] {len(bad)} bad / {len(files):,}")
        broken[split] = bad
    return broken


def redownload(split: str, file_name: str, attempts: int = 6) -> bool:
    """Re-fetch one image, verifying the byte count this time."""
    import requests

    url = f"http://images.cocodataset.org/{URL_PREFIX[split]}/{file_name}"
    source = SOURCE_DIRS[split] / file_name
    dest = images_dir(split) / file_name
    tmp = source.with_suffix(".jpg.part")

    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(10, 60)) as resp:
                resp.raise_for_status()
                expected = resp.headers.get("Content-Length")
                expected = int(expected) if expected else None
                written = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
            # The check that was missing the first time round.
            if expected is not None and written != expected:
                raise OSError(f"short read: {written} of {expected} bytes")

            tmp.replace(source)
            if error := check_image(source):
                raise OSError(f"still undecodable after download: {error}")

            # data/images/<split>/x.jpg is a hard link to the cache copy, and
            # replace() above broke that link, so relink.
            dest.unlink(missing_ok=True)
            try:
                os.link(source, dest)
            except OSError:
                shutil.copy2(source, dest)
            return True
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            if attempt == attempts:
                print(f"  FAILED {file_name}: {exc}")
                return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = parser.parse_args()

    broken = scan(args.splits)
    total = sum(len(v) for v in broken.values())

    print(f"\n{'=' * 78}")
    print(f"SCAN COMPLETE: {total} bad image(s)")
    print("=" * 78)

    report = {
        split: [{"file": str(p.name), "error": e} for p, e in items]
        for split, items in broken.items()
    }
    (DATA_DIR / "image_integrity.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )

    if total == 0:
        print("Every image decodes cleanly.")
        return 0

    if not args.repair:
        print("Re-run with --repair to re-download the bad files.")
        return 1

    print(f"\nRepairing {total} file(s)...")
    fixed, failed = 0, []
    for split, items in broken.items():
        for path, _ in items:
            print(f"  re-downloading [{split}] {path.name}")
            if redownload(split, path.name):
                fixed += 1
            else:
                failed.append((split, path.name))

    print(f"\n{'=' * 78}")
    print(f"repaired {fixed}/{total}")
    if failed:
        print(f"still broken: {failed}")
        return 1
    print("All images decode cleanly now.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
