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
from starlette.responses import Response
from starlette.websockets import WebSocketDisconnect

from src.config import CLASSES, DISPLAY_SCORE_THRESHOLD, PROJECT_ROOT
from src.webapp.detector import Detector
from src.webapp.drawing import annotate, hex_colors, to_data_url
from src.webapp.session import SESSIONS_DIR, Session, list_sessions

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


class RevalidatingStaticFiles(StaticFiles):
    """Static assets the browser must check with the server before reusing.

    Without this, editing `style.css` or `app.js` and reloading can leave the
    old file in place: the browser has no reason to ask again, so the page
    quietly runs yesterday's code. That is a bad failure because nothing looks
    broken - a feature simply does not appear, and the obvious conclusion is
    that it was never built.

    `no-cache` does not mean "do not store": the file stays in the cache, and
    the browser revalidates it with its ETag. An unchanged file comes back as a
    304 with no body, so the cost over localhost is a fraction of a millisecond
    and correctness stops depending on anyone remembering Ctrl+Shift+R.
    """

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


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

    # A second copy of the model, for the camera only. Ultralytics keeps the
    # tracker on `model.predictor.trackers`, so a tracking model cannot be
    # shared with the upload route without the two interfering. Two instances
    # cost about 0.1 GB extra on an 8.6 GB card, which is the cheap way out.
    print("loading a second copy for camera tracking ...", flush=True)
    tracker = Detector(model_key=model_key, device=device, track=True)
    print("ready: camera tracking enabled", flush=True)

    app.state.detector = detector
    app.state.tracker = tracker
    # Only one camera session may run at a time - see the socket handler.
    app.state.live_session = None
    yield

    # A session still open when the server stops should still be recorded.
    if app.state.live_session is not None:
        app.state.live_session.close()
    app.state.detector = None
    app.state.tracker = None


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
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", RevalidatingStaticFiles(directory=static_dir), name="static")
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
    # Served directly so the JSON and the CSV are one click away, no route.
    # Revalidated too: the session list changes every time the camera stops.
    app.mount("/sessions", RevalidatingStaticFiles(directory=SESSIONS_DIR), name="sessions")

    # ------------------------------------------------------------------ #
    # Pages and metadata
    # ------------------------------------------------------------------ #

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html",
                            headers={"Cache-Control": "no-cache"})

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
    # Sessions
    # ------------------------------------------------------------------ #

    @app.get("/api/sessions")
    async def sessions(limit: int = 20) -> dict:
        """Saved sessions, newest first, plus where the files live."""
        return {
            "sessions": list_sessions(limit=limit),
            "csv_url": "/sessions/sessions.csv",
            "csv_exists": (SESSIONS_DIR / "sessions.csv").is_file(),
        }

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

        A text message is a control message: `{"conf": 0.4}` moves the slider
        without tearing the socket down, `{"action": "end"}` finishes the
        session and asks for its summary before the socket closes.

        The socket *is* the session. It opens when the user presses Start
        camera and closes when they press Stop, so there is nothing extra to
        keep in step - and a session ended by closing the laptop lid still gets
        written, because the disconnect runs the same close path.
        """
        await websocket.accept()

        # One session at a time. The tracker lives on the model instance, so a
        # second camera in another tab would feed its frames into the same
        # tracker and both tallies would be wrong. Refusing plainly beats
        # producing two confident, incorrect numbers.
        existing = app.state.live_session
        if existing is not None and not existing.closed:
            await websocket.send_json({
                "error": "busy",
                "detail": "a camera session is already running in another tab",
            })
            await websocket.close()
            return

        tracker: Detector = app.state.tracker
        tracker.reset_tracking()          # ids restart at 1 for the new session
        session = Session.start()
        app.state.live_session = session

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
                        control = json.loads(text)
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue      # a malformed control message changes nothing

                    if control.get("action") == "end":
                        # The polite ending: the browser wants the summary back
                        # before it drops the socket.
                        await websocket.send_json({"session_closed": session.close()})
                        break

                    try:
                        conf = float(control.get("conf", conf))
                    except (ValueError, TypeError):
                        pass
                    continue

                frame = message.get("bytes")
                if not frame:
                    continue

                seq += 1
                current = conf

                def work() -> dict:
                    image = _open_image(frame)
                    result = tracker.detect(image, conf=current)
                    session.update(result)
                    return result.as_dict()

                try:
                    payload = await run_in_threadpool(work)
                except HTTPException as exc:
                    # Every frame gets exactly one reply, even a broken one. The
                    # client sends the next frame only after the previous answer
                    # arrives, so staying silent here would not skip a frame - it
                    # would stall the stream for good.
                    payload = {**_EMPTY_RESULT, "error": str(exc.detail)}

                await websocket.send_json(
                    {"seq": seq, **payload, "session": session.live_state()}
                )
        except WebSocketDisconnect:
            pass
        finally:
            # Whichever way the session ended - Stop pressed, tab closed, cable
            # pulled - it is recorded here. `close` is idempotent, so the polite
            # path having already written the file costs nothing.
            session.close()
            if app.state.live_session is session:
                app.state.live_session = None

    return app


app = create_app()

__all__ = ["app", "create_app"]
