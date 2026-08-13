"""Training entrypoints, dispatched by model key.

Callers speak one vocabulary - `batch`, `imgsz`, `warm_start` - and this module
translates it into what each trainer actually accepts. The three signatures
cannot simply be unified: ultralytics owns its own trainer and has no notion of
`warm_start` or of truncating an epoch, while SSDLite's and D-FINE's input
sizes are fixed by the model and processor rather than passed per run.

The import of each trainer is deferred until it is selected: pulling in
`train_dfine` costs a `transformers` + `timm` import, and `train_yolo` an
`ultralytics` one. Training SSDLite should pay for neither.
"""

from __future__ import annotations

#: Model key -> the `src.training.<module>` that owns it.
TRAINERS: dict[str, str] = {
    "ssdlite": "train_ssdlite",
    "yolo11s": "train_yolo",
    "dfine": "train_dfine",
}

#: Common argument name -> that trainer's own parameter name.
_RENAMES: dict[str, dict[str, str]] = {
    "ssdlite": {"batch": "batch_size"},
    "dfine": {"batch": "batch_size"},
}

#: Arguments a trainer cannot honour, dropped with a warning rather than a
#: crash - so `--model all` never dies halfway because of one stray flag.
_UNSUPPORTED: dict[str, tuple[str, ...]] = {
    "ssdlite": ("imgsz",),
    "dfine": ("imgsz",),
    "yolo11s": ("warm_start", "validate_every", "limit_train_batches", "limit_val_batches"),
}


def run_training(model_key: str, **overrides):
    """Train one model. `overrides` use the common argument names above."""
    if model_key not in TRAINERS:
        raise KeyError(f"unknown model '{model_key}', expected one of {list(TRAINERS)}")

    kwargs = {k: v for k, v in overrides.items() if v is not None}

    for name in _UNSUPPORTED[model_key]:
        if name in kwargs:
            print(f"[{model_key}] ignoring --{name.replace('_', '-')}: not supported by this model")
            kwargs.pop(name)

    for common, native in _RENAMES.get(model_key, {}).items():
        if common in kwargs:
            kwargs[native] = kwargs.pop(common)

    from importlib import import_module

    trainer = import_module(f"src.training.{TRAINERS[model_key]}")
    return trainer.run(**kwargs)


__all__ = ["TRAINERS", "run_training"]
