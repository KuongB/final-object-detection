"""Draw detections onto an image, for the half of the app that returns a picture.

The colours come from `CLASS_COLORS_RGB` in `src/config.py`, so a banana is the
same amber here as in `reports/figures/` and in the notebooks. Only the palette
is shared, though - `src/evaluation/figures.py::draw_boxes` renders into a
matplotlib axis, and importing that module pins the backend to Agg. Building a
figure per HTTP request would be both slow and not thread-safe, so this is a
second renderer over the same colours rather than a duplicated one.

The upload route also needs the result as something an `<img>` can display, so
`to_data_url` lives here too - encoding is part of rendering.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from typing import TYPE_CHECKING

from src.config import CLASS_COLORS_RGB

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

    from src.webapp.detector import DetectionResult

#: Long-side cap for the picture sent back to the browser. Drawing happens at
#: full resolution first, so line weights stay proportional; this only keeps a
#: 4000px phone photo from turning into a multi-megabyte base64 string.
MAX_TRANSPORT_SIDE = 1600

_FALLBACK_COLOR = (110, 110, 110)


@lru_cache(maxsize=16)
def _font(size: int):
    """A scalable font, whichever of them this machine happens to have.

    Pillow's bitmap default is fixed at ~11px and would be unreadable on a
    large photo. `load_default(size=...)` has been scalable since Pillow 10.1
    and is the reason this can fall through to it safely instead of shipping a
    font file.
    """
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def annotate(image: "Image", result: "DetectionResult") -> "Image":
    """Return a copy of `image` with one labelled box per detection."""
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    # Everything scales off the long side: a 640px COCO image and a 4000px
    # phone photo should end up looking the same once displayed at one width.
    long_side = max(canvas.width, canvas.height)
    line_width = max(2, round(long_side * 0.004))
    font = _font(max(13, round(long_side * 0.022)))
    pad = max(2, line_width)

    for det in result.detections:
        colour = CLASS_COLORS_RGB.get(det.label, _FALLBACK_COLOR)
        x0, y0, x1, y1 = det.box
        draw.rectangle((x0, y0, x1, y1), outline=colour, width=line_width)

        text = f"{det.label} {det.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        plate_h = text_h + 2 * pad

        # Above the box normally, tucked inside it when the box starts at the
        # top edge and there is no room outside.
        plate_y = y0 - plate_h
        if plate_y < 0:
            plate_y = min(y0, canvas.height - plate_h)
        plate_x = min(x0, max(0, canvas.width - (text_w + 2 * pad)))

        draw.rectangle(
            (plate_x, plate_y, plate_x + text_w + 2 * pad, plate_y + plate_h),
            fill=colour,
        )
        draw.text((plate_x + pad - left, plate_y + pad - top), text,
                  fill=(255, 255, 255), font=font)

    return canvas


def to_data_url(image: "Image", max_side: int = MAX_TRANSPORT_SIDE, quality: int = 90) -> str:
    """Encode as a JPEG `data:` URL the browser can drop straight into `src`.

    JPEG rather than PNG: these are photographs, and a PNG of one is several
    times larger for no visible gain once it is base64'd into the response.
    """
    from PIL import Image as PILImage

    if max(image.width, image.height) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), PILImage.LANCZOS)

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def hex_colors() -> dict[str, str]:
    """The palette as CSS hex, so the browser draws boxes in the same colours."""
    return {
        name: "#{:02x}{:02x}{:02x}".format(*rgb)
        for name, rgb in CLASS_COLORS_RGB.items()
    }


__all__ = ["annotate", "hex_colors", "to_data_url"]
