"""Verify that every dependency of the project is importable and that the GPU
is usable. Run this after setting up the conda environment, and again on
Colab/Kaggle before training, to catch a broken install early.

    python scripts/00_check_env.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS: list[tuple[str, str, str]] = []


def check(name, fn, verbose_errors=True):
    try:
        RESULTS.append((name, "OK", str(fn())))
    except Exception as exc:  # noqa: BLE001 - we want to report, not crash
        RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        if verbose_errors:
            traceback.print_exc()


def _torch():
    import torch

    if not torch.cuda.is_available():
        return f"{torch.__version__} | cuda=False (CPU only - training will be slow)"
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cap = torch.cuda.get_device_capability(0)
    return f"{torch.__version__} | {name} | {total_gb:.1f} GB | sm_{cap[0]}{cap[1]}"


def _torchvision():
    import torchvision
    from torchvision.models.detection import (  # noqa: F401
        ssdlite320_mobilenet_v3_large,
    )

    return f"{torchvision.__version__} | ssdlite320_mobilenet_v3_large available"


def _cv2():
    import cv2
    import numpy as np

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (20, 20), (0, 255, 0), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode failed"
    return f"{cv2.__version__} | draw + encode OK"


def _fiftyone():
    import fiftyone as fo

    return fo.__version__


def _ultralytics():
    import ultralytics

    return ultralytics.__version__


def _transformers():
    import transformers
    from transformers import DFineForObjectDetection  # noqa: F401

    return f"{transformers.__version__} | DFineForObjectDetection available"


def _pycocotools():
    from pycocotools.coco import COCO  # noqa: F401
    from pycocotools.cocoeval import COCOeval  # noqa: F401

    return "COCO + COCOeval available"


def _webstack():
    import fastapi
    import multipart  # noqa: F401  (python-multipart, needed for file upload)
    import uvicorn

    return f"fastapi={fastapi.__version__} uvicorn={uvicorn.__version__}"


def _sci():
    import matplotlib
    import numpy as np
    import pandas as pd
    import seaborn as sns

    return (
        f"numpy={np.__version__} pandas={pd.__version__} "
        f"matplotlib={matplotlib.__version__} seaborn={sns.__version__}"
    )


def _gpu_compute():
    import torch

    if not torch.cuda.is_available():
        return "skipped (no CUDA)"
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    (a @ b).sum().item()
    torch.cuda.synchronize()
    return "1024x1024 matmul on GPU OK"


def _project_config():
    from src.config import CLASSES, PROJECT_ROOT, ensure_dirs

    ensure_dirs()
    return f"{len(CLASSES)} classes {CLASSES} | root={PROJECT_ROOT}"


def main() -> int:
    print(f"python     : {sys.version.split()[0]}")
    print(f"executable : {sys.executable}\n")

    check("torch", _torch)
    check("torchvision + SSD", _torchvision)
    check("opencv (cv2)", _cv2)
    check("fiftyone", _fiftyone)
    check("ultralytics", _ultralytics)
    check("transformers", _transformers)
    check("pycocotools", _pycocotools)
    check("fastapi stack", _webstack)
    check("scientific stack", _sci)
    check("GPU compute", _gpu_compute)
    check("project config", _project_config)

    width = 100
    print("\n" + "=" * width)
    print(f"{'COMPONENT':<20} {'STATUS':<7} DETAIL")
    print("-" * width)
    for name, status, detail in RESULTS:
        print(f"{name:<20} {status:<7} {detail}")
    print("=" * width)

    failed = [n for n, s, _ in RESULTS if s == "FAIL"]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
