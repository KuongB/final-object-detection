"""Step 6 - serve the web app. Launcher only; see `src/webapp/cli.py`.

    python scripts/30_serve.py
    python scripts/30_serve.py --port 8080 --device cpu
    python scripts/30_serve.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.webapp.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
