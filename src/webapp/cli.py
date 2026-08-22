"""Command-line front end for the web application - the whole of it.

`scripts/30_serve.py` is only a launcher; the behaviour lives here, next to the
app it starts, matching how training and evaluation are laid out.

    python scripts/30_serve.py
    python scripts/30_serve.py --port 8080 --device cpu
"""

from __future__ import annotations

import argparse
import os

from src.config import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/30_serve.py",
        description=(
            "Serve the detection web app: upload an image, or stream the webcam.\n"
            "Open the printed URL on this machine - browsers only grant camera\n"
            "access over localhost or HTTPS, so a LAN address will not work for\n"
            "the live tab."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'")
    parser.add_argument("--model", default="yolo26m",
                        help="key in src/config.py MODELS; only ultralytics "
                             "checkpoints are servable (default: yolo26m)")
    parser.add_argument("--reload", action="store_true",
                        help="restart on source changes - development only, it "
                             "reloads the model too")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # `load_pretrained` asks ultralytics for "yolo26m.pt" by relative path, and
    # ultralytics resolves that against the working directory - downloading a
    # fresh copy wherever it happens to be if the file is not there. Anchoring
    # to the repo root makes the checkpoint at the root the one that is served,
    # from whatever directory the command was typed in.
    os.chdir(PROJECT_ROOT)

    # The app is imported by name so `--reload` can re-import it; settings
    # therefore travel through the environment rather than a function argument.
    os.environ["OBJDET_WEBAPP_DEVICE"] = args.device
    os.environ["OBJDET_WEBAPP_MODEL"] = args.model

    import uvicorn

    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"\n  http://{shown}:{args.port}\n", flush=True)

    uvicorn.run(
        "src.webapp.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(PROJECT_ROOT / "src")] if args.reload else None,
        log_level="info",
    )
    return 0
