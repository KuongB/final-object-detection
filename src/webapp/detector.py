"""One image in, a list of objects and a per-class tally out.

This is the whole inference side of the web application. Both entry points -
the upload form and the live camera socket - come through `Detector.detect`,
so there is exactly one place where a frame becomes detections, and exactly one
class-id mapping to get wrong.

The model is the public COCO checkpoint `yolo26m.pt`, *not* a fine-tuned one.
That is a measured decision rather than an oversight: it scores 0.3063 mAP on
the test split against 0.2642 for the best fine-tuned checkpoint (see
`reports/evaluation.md`). Its head still carries all 80 COCO classes, so the
five this project cares about are selected with `classes=` at predict time and
their indices are translated back through the same map the evaluation used.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.config import CLASSES, COCO_ID_TO_CLASS, DISPLAY_SCORE_THRESHOLD

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image


@dataclass(frozen=True)
class Detection:
    """One object, in the coordinates of the image that was submitted."""

    label: str
    confidence: float
    #: xyxy, absolute pixels of the original image - not the 640x640 the model saw.
    box: tuple[float, float, float, float]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 1) for v in self.box],
        }


@dataclass
class DetectionResult:
    """Everything one frame produced, ready to be serialised as-is."""

    detections: list[Detection] = field(default_factory=list)
    #: All five classes are always present, absent ones at 0 - the front end
    #: renders the full table, so it stays visible that the app covers five.
    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    inference_ms: float = 0.0
    width: int = 0
    height: int = 0

    def as_dict(self) -> dict:
        return {
            "detections": [d.as_dict() for d in self.detections],
            "counts": self.counts,
            "total": self.total,
            "inference_ms": round(self.inference_ms, 1),
            "width": self.width,
            "height": self.height,
        }


class Detector:
    """A loaded YOLO26m, plus the bookkeeping that turns its output into names.

    Building this reads a checkpoint and nothing else - no dataset, no
    annotations - so it is cheap enough to construct once at server startup and
    hold for the lifetime of the process.
    """

    def __init__(self, model_key: str = "yolo26m", device: str = "auto") -> None:
        from src.evaluation.runner import load_pretrained
        from src.training.common import get_device

        loaded, class_map = load_pretrained(model_key, device)

        if loaded.framework != "ultralytics" or class_map is None:
            # SSDLite and D-FINE need entirely different pre- and
            # post-processing; accepting them here would silently run the
            # wrong adapter rather than fail.
            raise ValueError(
                f"the web app only serves ultralytics checkpoints, but "
                f"{model_key!r} is {loaded.framework}"
            )

        self.model_key = model_key
        self.imgsz = loaded.imgsz
        self._model = loaded.model

        # Ultralytics wants a device *index*, not a torch.device - the same
        # conversion `src/evaluation/runner.py::_detect` makes.
        resolved = get_device(device)
        self.device = str(resolved)
        self._predict_device = 0 if resolved.type == "cuda" else "cpu"

        # `class_map` is {COCO-80 index -> our 1..5 category id}, built from
        # HF_COCO80_INDEX in src/config.py. Deriving both the filter and the
        # names from it means those five indices are never spelled out twice.
        self._keep = sorted(class_map)
        self._name_of = {
            idx: COCO_ID_TO_CLASS[cat_id] for idx, cat_id in class_map.items()
        }

        # Ultralytics caches per-call state on the predictor, so two requests
        # arriving together would interleave inside it. At ~48 frames per
        # second of capacity, serialising them costs nothing worth having.
        self._lock = threading.Lock()

        self._warmup()

    def _warmup(self) -> None:
        """Pay the first-call costs at startup instead of on a user's request.

        Two of them, and both are visible without this. CUDA autotunes its
        kernels on the first forward pass, which turned the first upload into
        149 ms against a steady-state 21 ms. And ultralytics folds each
        BatchNorm into the convolution ahead of it the first time it predicts,
        so `describe()` reports 21.9 M parameters before that happens and the
        20.41 M the report quotes afterwards - the same model, counted at two
        different moments.
        """
        from PIL import Image

        self.detect(Image.new("RGB", (self.imgsz, self.imgsz)), conf=0.9)

    # ------------------------------------------------------------------ #

    def detect(
        self, image: "Image", conf: float = DISPLAY_SCORE_THRESHOLD
    ) -> DetectionResult:
        """Run the model over one PIL image.

        Takes a **PIL image**, deliberately. Handing ultralytics a numpy array
        instead makes it read the channels as BGR, and an RGB array passed that
        way loses detections without any error - measured on
        `data/images/test/000000002149.jpg`, three boxes become one. Passing
        the PIL object lets ultralytics do its own conversion, which is what
        the evaluation run got by passing file paths.
        """
        conf = min(max(float(conf), 0.01), 0.99)

        with self._lock:
            result = self._model.predict(
                image,
                imgsz=self.imgsz,
                conf=conf,
                classes=self._keep,
                device=self._predict_device,
                # Square 640x640 letterbox, which is *not* what ultralytics does
                # for a single image by default. Left alone it pads only to the
                # next stride multiple (640x480 for a landscape photo), and the
                # model then sees a different input than it did during
                # evaluation - where images arrived in mixed-shape batches of 16
                # and were padded to a full square. The two disagree on 62 of
                # the 310 test images, by up to 3 objects: 000000061658.jpg
                # gives 7 broccoli rectangular and 10 square. The report's
                # 0.3063 mAP was measured the square way, so that is what this
                # app has to reproduce.
                rect=False,
                verbose=False,
            )[0]

        detections: list[Detection] = []
        counts = {name: 0 for name in CLASSES}

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            height, width = result.orig_shape
            for cls, score, xyxy in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
            ):
                name = self._name_of.get(int(cls))
                if name is None:  # `classes=` already filtered; belt and braces
                    continue
                x0, y0, x1, y1 = xyxy
                detections.append(
                    Detection(
                        label=name,
                        confidence=float(score),
                        # Clipped because a box may overhang the border and the
                        # front end draws onto a canvas exactly the image's size.
                        box=(
                            max(0.0, min(x0, width)),
                            max(0.0, min(y0, height)),
                            max(0.0, min(x1, width)),
                            max(0.0, min(y1, height)),
                        ),
                    )
                )
                counts[name] += 1

        detections.sort(key=lambda d: d.confidence, reverse=True)

        # `result.speed` is what the caller actually waited for: the resize and
        # the box decoding are as much a part of the response time as the
        # forward pass, and on this model they are the larger half of it.
        speed = result.speed or {}
        elapsed = sum(
            float(speed.get(k, 0.0) or 0.0)
            for k in ("preprocess", "inference", "postprocess")
        )

        return DetectionResult(
            detections=detections,
            counts=counts,
            total=len(detections),
            inference_ms=elapsed,
            width=image.width,
            height=image.height,
        )

    def describe(self) -> dict:
        """The facts the front end prints in its header."""
        params = sum(p.numel() for p in self._model.model.parameters())
        return {
            "key": self.model_key,
            "name": "YOLO26m - public COCO checkpoint, not fine-tuned",
            "params_millions": round(params / 1e6, 2),
            "imgsz": self.imgsz,
            "device": self.device,
            "test_mAP_50_95": 0.3063,
        }


__all__ = ["Detection", "DetectionResult", "Detector"]
