"""Step 5 - score the trained models. Launcher only; see `src/evaluation/cli.py`.

    python scripts/20_evaluate.py
    python scripts/20_evaluate.py --model yolo11s --split val
    python scripts/20_evaluate.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
