"""Step 2a-bis - resilient image fetcher, used when FiftyOne's own downloader
keeps dying on a flaky connection.

FiftyOne downloads images with a multiprocessing pool that has no per-image
retry: one aborted socket propagates out of `imap_unordered` and takes the whole
batch down. On a connection that drops every few hundred requests (WinError
10053 - "aborted by the software in your host machine", typically a local
firewall/antivirus/VPN rather than the server) that turns into minutes of
progress per attempt.

This script does the same job with retries at the level of a single image:
a dead socket costs one image, not the batch.

It writes into FiftyOne's own cache layout, so afterwards
`01_download_dataset.py` finds every file already present, skips downloading
entirely, and just builds the manifest.

    C:/Users/<you>/fiftyone/coco-2017/<split>/data/000000123456.jpg

Usage
-----
    python scripts/01b_fetch_images.py                  # both splits
    python scripts/01b_fetch_images.py --split train
    python scripts/01b_fetch_images.py --workers 2      # gentler on the network
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASSES  # noqa: E402

# FiftyOne's on-disk layout for the coco-2017 zoo dataset.
FIFTYONE_COCO = Path.home() / "fiftyone" / "coco-2017"
RAW_DIR = FIFTYONE_COCO / "raw"

# our split name -> (annotation json, image directory, coco url prefix)
SPLIT_SPECS = {
    "train": ("instances_train2017.json", "train", "train2017"),
    "validation": ("instances_val2017.json", "validation", "val2017"),
}

_print_lock = threading.Lock()


def build_session(pool_size: int):
    """A session that retries connection-level failures inside urllib3 itself,
    before the exception ever reaches us."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = "fruitveg-detection-coursework/1.0"
    return session


def required_images(ann_file: Path) -> dict[str, str]:
    """file_name -> coco_url for every image holding >=1 label of our classes.

    Mirrors what FiftyOne selects with `classes=[...]`, computed straight from
    the annotation file that FiftyOne already downloaded.
    """
    print(f"  reading {ann_file.name} ({ann_file.stat().st_size / 1024**2:.0f} MB)...")
    data = json.loads(ann_file.read_text(encoding="utf-8"))

    wanted = {c["id"] for c in data["categories"] if c["name"] in set(CLASSES)}
    names = sorted(c["name"] for c in data["categories"] if c["id"] in wanted)
    print(f"  matched categories: {names}")
    if len(wanted) != len(CLASSES):
        missing = set(CLASSES) - set(names)
        raise SystemExit(f"ERROR: categories not found in annotations: {missing}")

    image_ids = {
        ann["image_id"] for ann in data["annotations"] if ann["category_id"] in wanted
    }
    return {
        img["file_name"]: img.get("coco_url", "")
        for img in data["images"]
        if img["id"] in image_ids
    }


def fetch_one(session, url: str, dest: Path, attempts: int = 6) -> str:
    """Download a single image. Returns 'ok', 'skip' or 'fail'."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with session.get(url, stream=True, timeout=(10, 60)) as resp:
                resp.raise_for_status()
                expected = resp.headers.get("Content-Length")
                expected = int(expected) if expected else None
                written = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
            if written == 0:
                raise OSError("empty file")
            # A connection dropped mid-body still yields a syntactically valid
            # JPEG that PIL only rejects on full decode, at training time. The
            # byte count is the cheap way to catch it here instead.
            if expected is not None and written != expected:
                raise OSError(f"short read: {written} of {expected} bytes")
            tmp.replace(dest)  # atomic: a partial file is never seen as done
            return "ok"
        except Exception:  # noqa: BLE001 - any network/IO error is retryable
            tmp.unlink(missing_ok=True)
            if attempt == attempts:
                return "fail"
            time.sleep(min(10.0, 0.5 * 2 ** (attempt - 1)))
    return "fail"


def fetch_split(split: str, workers: int, attempts: int) -> int:
    ann_name, dir_name, url_prefix = SPLIT_SPECS[split]
    ann_file = RAW_DIR / ann_name
    if not ann_file.is_file():
        raise SystemExit(
            f"ERROR: {ann_file} not found.\n"
            "Run 01_download_dataset.py once first so FiftyOne fetches the "
            "COCO annotation files."
        )

    dest_dir = FIFTYONE_COCO / dir_name / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 78}")
    print(f"SPLIT: {split}")
    print("=" * 78)

    targets = required_images(ann_file)
    missing = {
        name: (url or f"http://images.cocodataset.org/{url_prefix}/{name}")
        for name, url in targets.items()
        if not (dest_dir / name).exists() or (dest_dir / name).stat().st_size == 0
    }
    print(f"  required : {len(targets):,}")
    print(f"  on disk  : {len(targets) - len(missing):,}")
    print(f"  to fetch : {len(missing):,}")
    if not missing:
        print("  nothing to do")
        return 0

    session = build_session(pool_size=max(workers * 2, 8))
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed: list[str] = []
    started = time.time()
    done = 0
    total = len(missing)

    def worker(item):
        nonlocal done
        name, url = item
        status = fetch_one(session, url, dest_dir / name, attempts=attempts)
        with _print_lock:
            counts[status] += 1
            if status == "fail":
                failed.append(name)
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                print(
                    f"  {done:>6,}/{total:,}  ok={counts['ok']:,} "
                    f"fail={counts['fail']:,}  {rate:5.1f} img/s  "
                    f"ETA {eta / 60:5.1f} min",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, missing.items()))

    print(f"\n  ok={counts['ok']:,}  skipped={counts['skip']:,}  failed={counts['fail']:,}")
    if failed:
        print(f"  first failures: {failed[:10]}")
        print("  re-run this script to retry only the ones still missing")
    return counts["fail"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=[*SPLIT_SPECS, "all"], default="all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="concurrent downloads; lower this if the connection keeps dropping",
    )
    parser.add_argument(
        "--attempts", type=int, default=6, help="retries per image"
    )
    args = parser.parse_args()

    splits = list(SPLIT_SPECS) if args.split == "all" else [args.split]
    total_failed = sum(fetch_split(s, args.workers, args.attempts) for s in splits)

    print(f"\n{'=' * 78}")
    if total_failed:
        print(f"{total_failed:,} image(s) still missing - re-run to retry them.")
        return 1
    print("All required images are present.")
    print("Next: python scripts/01_download_dataset.py   (will skip downloading)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
