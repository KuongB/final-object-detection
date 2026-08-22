"""The FastAPI application - four routes over one `Detector`.

Two ways in, one inference path:

    POST /api/detect    an uploaded file; answers with a drawn picture
    WS   /ws/detect     a stream of camera frames; answers with coordinates only

The split is on purpose. A still image is looked at once, so sending back a
rendered JPEG is the simplest thing that works. A camera at 30 frames per
second is not: re-encoding every frame server-side would double the work and
the bandwidth for a picture that is replaced 33 milliseconds later. The socket
therefore sends numbers, and the browser paints them onto a canvas laid over
the live `<video>` - which is also why `/api/meta` publishes the palette.

The model is built once in `lifespan` and shared. Inference blocks for ~21 ms,
which is far too long to sit on the event loop, so every call goes through
`run_in_threadpool`.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

from src.config import CLASSES, DISPLAY_SCORE_THRESHOLD, PROJECT_ROOT
from src.webapp.detector import Detector
from src.webapp.drawing import annotate, hex_colors, to_data_url

FRONTEND_DIR = PROJECT_ROOT / "webapp" / "frontend"
SAMPLES_DIR = PROJECT_ROOT / "webapp" / "samples"

#: Refuse anything larger before it reaches the decoder. Generous enough for a
#: 50-megapixel phone photo, small enough that a stray upload cannot exhaust
#: memory.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: What the socket answers with when a frame cannot be decoded - same shape as
#: a real result, so the front end needs no special case to keep going.
_EMPTY_RESULT = {
    "detections": [],
    "counts": {name: 0 for name in CLASSES},
    "total": 0,
    "inference_ms": 0.0,
    "width": 0,
    "height": 0,
}


def _open_image(data: bytes):
    """Bytes off the wire to an RGB PIL image, oriented the way it was shot.

    `exif_transpose` matters more than it looks: phone cameras record portrait
    shots as landscape pixels plus a rotation flag, and without honouring it
    every box would be drawn against a sideways image.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"not a readable image: {exc}") from exc

    return ImageOps.exif_transpose(image).convert("RGB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = os.environ.get("OBJDET_WEBAPP_DEVICE", "auto")
    model_key = os.environ.get("OBJDET_WEBAPP_MODEL", "yolo26m")

    print(f"loading {model_key} on device '{device}' ...", flush=True)
    detector = Detector(model_key=model_key, device=device)
    facts = detector.describe()
    print(
        f"ready: {facts['name']}  {facts['params_millions']} M params  "
        f"device={facts['device']}  imgsz={facts['imgsz']}",
        flush=True,
    )

    app.state.detector = detector
    yield
    app.state.detector = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Fruit & Vegetable Detection",
        description="Requirement 2 - upload an image or stream the webcam.",
        version="1.0.0",
        lifespan=lifespan,
    )

    static_dir = FRONTEND_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

    # ------------------------------------------------------------------ #
    # Pages and metadata
    # ------------------------------------------------------------------ #

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        detector = getattr(app.state, "detector", None)
        return {"status": "ok" if detector else "loading",
                "device": detector.device if detector else None}

    @app.get("/api/meta")
    async def meta() -> dict:
        """Everything the front end needs to render without hard-coding it.

        The palette in particular: the browser draws the live overlay itself,
        and shipping the colours from here keeps it matching the server-drawn
        upload result and the figures in the report.
        """
        detector: Detector = app.state.detector
        samples = sorted(
            p.name for p in SAMPLES_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        return {
            "classes": list(CLASSES),
            "colors": hex_colors(),
            "default_threshold": DISPLAY_SCORE_THRESHOLD,
            "model": detector.describe(),
            "samples": samples,
        }

    # ------------------------------------------------------------------ #
    # Upload
    # ------------------------------------------------------------------ #

    @app.post("/api/detect")
    async def detect(
        file: UploadFile = File(...),
        conf: float = Form(DISPLAY_SCORE_THRESHOLD),
    ) -> JSONResponse:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"image is {len(data) / 1e6:.1f} MB, limit is "
                       f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB",
            )

        detector: Detector = app.state.detector

        def work() -> dict:
            image = _open_image(data)
            result = detector.detect(image, conf=conf)
            return {
                **result.as_dict(),
                "image_data_url": to_data_url(annotate(image, result)),
                "filename": file.filename,
            }

        return JSONResponse(await run_in_threadpool(work))

    # ------------------------------------------------------------------ #
    # Live camera
    # ------------------------------------------------------------------ #

    @app.websocket("/ws/detect")
    async def stream(websocket: WebSocket) -> None:
        """One frame in, one result out, strictly in step.

        The client only sends the next frame once it has the previous answer,
        so at most one is ever in flight. That is what keeps the overlay from
        drifting behind the video when the machine is busy: frames are dropped
        at the source rather than queuing up here.

        A text message is a control message - currently just the confidence
        slider, which has to be changeable without tearing the socket down.
        """
        await websocket.accept()
        detector: Detector = app.state.detector
        conf = DISPLAY_SCORE_THRESHOLD
        seq = 0

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break

                text = message.get("text")
                if text is not None:
                    try:
                        conf = float(json.loads(text).get("conf", conf))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        pass  # a malformed control message keeps the old value
                    continue

                frame = message.get("bytes")
                if not frame:
                    continue

                seq += 1
                current = conf

                def work() -> dict:
                    image = _open_image(frame)
                    return detector.detect(image, conf=current).as_dict()

                try:
                    payload = await run_in_threadpool(work)
                except HTTPException as exc:
                    # Every frame gets exactly one reply, even a broken one. The
                    # client sends the next frame only after the previous answer
                    # arrives, so staying silent here would not skip a frame - it
                    # would stall the stream for good.
                    payload = {**_EMPTY_RESULT, "error": str(exc.detail)}

                await websocket.send_json({"seq": seq, **payload})
        except WebSocketDisconnect:
            pass

    return app


app = create_app()

__all__ = ["app", "create_app"]
