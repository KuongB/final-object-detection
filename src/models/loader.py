"""Load a trained model back from `weights/`, whichever framework produced it.

Training writes three different things to disk - a torchvision state dict, an
ultralytics `.pt`, a HuggingFace directory - and both the evaluation scripts
and the web app need all three. Without this module each of them would grow its
own copy of "which builder do I call, and what do I feed it".

    from src.models.loader import load_trained
    from src.evaluation.predict import predict_dfine

    loaded = load_trained("dfine")
    detections, _ = predict_dfine(loaded.model, loaded.processor, split="test")

`weights/index.json` (written by `src.training.artifacts`) is the source of
truth for where each checkpoint lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import MODELS, PROJECT_ROOT
from src.training.artifacts import read_index


@dataclass
class LoadedModel:
    """A trained model plus what a caller needs to run it."""

    model_key: str
    framework: str
    model: object
    imgsz: int
    num_classes: int
    #: Only D-FINE has one; torchvision and ultralytics preprocess internally.
    processor: object | None = None
    #: The index entry it came from - carries `val_mAP_50_95`, `params`, etc.
    meta: dict | None = None


def _state_dict(ckpt: dict) -> dict:
    """The weights to evaluate: the EMA copy whenever the checkpoint has one.

    `best.pt` stores the EMA directly under `state_dict`, while `last.pt` stores
    the raw model there (the optimiser needs it to resume) and keeps the EMA
    beside it. Training validated the EMA in both cases, so preferring it here
    is what makes a `last.pt` score comparable to the training curve.
    """
    ema = ckpt.get("ema")
    return ema["ema"] if ema else ckpt["state_dict"]


def _resolve(model_key: str, weights: str | Path | None) -> tuple[Path, dict]:
    if weights is not None:
        return Path(weights).resolve(), {}

    index = read_index()
    if model_key not in index:
        raise FileNotFoundError(
            f"'{model_key}' is not in weights/index.json - train it first "
            f"(python scripts/10_train.py --model {model_key}), or pass an "
            f"explicit weights path."
        )
    entry = index[model_key]
    return (PROJECT_ROOT / entry["weights"]).resolve(), entry


def load_trained(
    model_key: str,
    weights: str | Path | None = None,
    device: str = "auto",
) -> LoadedModel:
    """Rebuild `model_key`'s architecture and load its trained weights.

    Frameworks are imported inside the branches on purpose: loading SSDLite
    should not drag in `transformers` and `ultralytics`, which cost several
    seconds of import time each.
    """
    if model_key not in MODELS:
        raise KeyError(f"unknown model '{model_key}', expected one of {list(MODELS)}")

    path, entry = _resolve(model_key, weights)
    if not path.exists():
        raise FileNotFoundError(f"{model_key}: checkpoint {path} does not exist")

    framework = MODELS[model_key]["framework"]
    if framework == "torchvision":
        return _load_torchvision(model_key, path, entry, device)
    if framework == "ultralytics":
        return _load_ultralytics(model_key, path, entry, device)
    return _load_transformers(model_key, path, entry, device)


def _load_torchvision(model_key: str, path: Path, entry: dict, device: str) -> LoadedModel:
    import torch

    from src.models.ssdlite import build_ssdlite
    from src.training.common import get_device

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    resolved = get_device(device)

    # `warm_start=False`: the trained weights overwrite the head anyway, and
    # warm-starting would only cost a pointless pass over the COCO checkpoint.
    model = build_ssdlite(num_classes=ckpt["num_classes"], warm_start=False)
    model.load_state_dict(_state_dict(ckpt))
    model.to(resolved).eval()

    return LoadedModel(
        model_key=model_key,
        framework="torchvision",
        model=model,
        imgsz=ckpt["imgsz"],
        num_classes=ckpt["num_classes"],
        meta=entry or ckpt.get("metrics"),
    )


def _load_ultralytics(model_key: str, path: Path, entry: dict, device: str) -> LoadedModel:
    from ultralytics import YOLO

    from src.config import NUM_CLASSES

    # Ultralytics checkpoints carry their own architecture, so there is nothing
    # to rebuild - and nothing to move to a device either; `predict` takes the
    # device per call.
    model = YOLO(str(path))
    return LoadedModel(
        model_key=model_key,
        framework="ultralytics",
        model=model,
        imgsz=entry.get("imgsz", MODELS[model_key]["imgsz"]),
        num_classes=entry.get("num_classes", NUM_CLASSES),
        meta=entry,
    )


def _load_transformers(model_key: str, path: Path, entry: dict, device: str) -> LoadedModel:
    import torch

    from src.models.dfine import build_dfine, build_dfine_processor
    from src.training.common import get_device

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    resolved = get_device(device)

    model = build_dfine(
        ckpt["base_checkpoint"], num_classes=ckpt["num_classes"], warm_start=False
    )
    model.load_state_dict(_state_dict(ckpt))
    model.to(resolved).eval()

    processor = build_dfine_processor(ckpt["base_checkpoint"], size=ckpt["imgsz"])

    return LoadedModel(
        model_key=model_key,
        framework="transformers",
        model=model,
        imgsz=ckpt["imgsz"],
        num_classes=ckpt["num_classes"],
        processor=processor,
        meta=entry or ckpt.get("metrics"),
    )


__all__ = ["LoadedModel", "load_trained"]
