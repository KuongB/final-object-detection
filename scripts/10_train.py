"""Step 4 - fine-tune the detectors. Launcher only; see `src/training/cli.py`.

    python scripts/10_train.py --model ssdlite
    python scripts/10_train.py --model all --epochs 15
    python scripts/10_train.py --model dfine --resume
    python scripts/10_train.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
