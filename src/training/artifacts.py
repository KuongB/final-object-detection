"""Promote a finished run's best checkpoint into `weights/`.

`runs/` is a training log - one directory per attempt, tagged, disposable. But
evaluation and the web app need a single answer to "where is the trained
SSDLite?", and they should not have to know which run tag won or how each
framework lays out its files.

So every completed run copies its best checkpoint to `weights/<model_key>/` and
records one entry in `weights/index.json`. That index is the only file a
consumer has to read.

Paths in the index are relative to the project root, so the whole folder stays
valid when it is zipped up for submission or moved to another machine.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT, WEIGHTS_DIR

INDEX_PATH = WEIGHTS_DIR / "index.json"


def _relative(path: Path) -> str:
    """Project-relative where possible, absolute otherwise.

    `OBJDET_RUNS_DIR` can point anywhere, and this runs at the very end of a
    multi-hour training run - it must not be the thing that raises.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_index() -> dict:
    if not INDEX_PATH.is_file():
        return {}
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def promote(model_key: str, run_dir: Path, entry: dict) -> Path | None:
    """Copy `run_dir/best.pt` to `weights/<model_key>/` and index it.

    `entry` carries what the index cannot derive: `framework`, `imgsz`,
    `num_classes`, `val_mAP_50_95`, `params`. Returns the promoted path, or
    `None` when the run produced no best checkpoint (a run with validation
    turned off, or one that died in its first epoch).
    """
    source = run_dir / "best.pt"
    if not source.is_file():
        print(f"[promote] {model_key}: no best.pt in {run_dir}, nothing to promote")
        return None

    target_dir = WEIGHTS_DIR / model_key
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "best.pt"
    shutil.copy2(source, target)

    record = {
        **entry,
        "weights": _relative(target),
        "run_dir": _relative(run_dir),
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    hf_source = run_dir / "hf"
    if hf_source.is_dir():
        hf_target = target_dir / "hf"
        shutil.copytree(hf_source, hf_target, dirs_exist_ok=True)
        record["hf_dir"] = _relative(hf_target)

    index = read_index()
    index[model_key] = record
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=1), encoding="utf-8")

    print(f"[promote] {model_key} -> {record['weights']}  (index: {_relative(INDEX_PATH)})")
    return target
