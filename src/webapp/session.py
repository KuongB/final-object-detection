"""Count how many distinct fruits a camera session saw, and write it down.

A session runs from "Start camera" to "Stop camera". Within it the question is
*how many apples went past*, which is not the same as *how many apples were on
screen* - and it is emphatically not the sum of the per-frame counts. At 30
frames per second one apple sitting still for ten seconds contributes 300 to
that sum, so the total would measure the frame rate rather than the fruit.

The number that answers the question is the count of distinct tracker ids. The
tracker gives an object an id when it appears and keeps that id while it stays
in view; once it leaves and something turns up later, that is a new id and a
new apple. Counting the ids we have ever seen, per class, is therefore exactly
what was asked for, and it is a set union rather than an addition - the same
apple seen in 300 frames still lands in the set once.

Sessions are held on the server rather than in the browser because the files
have to be written somewhere, and because a session that ends by closing the
laptop lid still needs to be recorded: the socket dropping is itself the signal
that the session is over.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import CLASSES, PROJECT_ROOT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.webapp.detector import DetectionResult

SESSIONS_DIR = PROJECT_ROOT / "webapp" / "sessions"

#: One row per session, appended across runs - the table to open in a
#: spreadsheet when comparing sessions. The per-session JSON beside it holds
#: the same numbers for a single session.
INDEX_CSV = SESSIONS_DIR / "sessions.csv"

CSV_COLUMNS = (
    ["session_id", "started_at", "ended_at", "duration_seconds", "frames"]
    + list(CLASSES)
    + ["total"]
)


def _now() -> datetime:
    """Local wall-clock time, carrying its offset so the stamp is unambiguous."""
    return datetime.now().astimezone()


@dataclass
class Session:
    """One camera session's running tally.

    `seen` is the working state and `counts` is derived from it: the tally is a
    set of ids per class, so re-observing an object cannot inflate anything no
    matter how many frames it survives.
    """

    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    frames: int = 0
    #: class name -> the tracker ids ever seen for it this session
    seen: dict[str, set[int]] = field(
        default_factory=lambda: {name: set() for name in CLASSES}
    )
    closed: bool = False

    # ------------------------------------------------------------------ #

    @classmethod
    def start(cls) -> "Session":
        started = _now()
        return cls(session_id=started.strftime("%Y%m%d_%H%M%S"), started_at=started)

    def update(self, result: "DetectionResult") -> None:
        """Fold one frame's detections into the tally."""
        self.frames += 1
        for det in result.detections:
            # An object the tracker has not confirmed yet has no id. It is still
            # drawn on the overlay; it simply is not countable until the tracker
            # settles on it, usually within a frame or two.
            if det.track_id is not None and det.label in self.seen:
                self.seen[det.label].add(det.track_id)

    @property
    def counts(self) -> dict[str, int]:
        """All five classes, absent ones at 0 - what the front end renders."""
        return {name: len(ids) for name, ids in self.seen.items()}

    @property
    def total(self) -> int:
        return sum(len(ids) for ids in self.seen.values())

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or _now()
        return round((end - self.started_at).total_seconds(), 1)

    # ------------------------------------------------------------------ #

    def live_state(self) -> dict:
        """The running tally, sent alongside every frame's result."""
        return {
            "id": self.session_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "counts": self.counts,
            "total": self.total,
            "frames": self.frames,
            "duration_seconds": self.duration_seconds,
        }

    def summary(self) -> dict:
        """The finished record - exactly what gets written to the JSON file."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": (self.ended_at or _now()).isoformat(timespec="seconds"),
            "duration_seconds": self.duration_seconds,
            "frames": self.frames,
            "counts": self.counts,
            "total": self.total,
        }

    def close(self) -> dict | None:
        """End the session and write it out. Safe to call twice.

        Idempotent on purpose: a session can end two ways - the browser asking
        politely, or the socket simply dropping - and both paths run through
        here. Whichever happens first wins, and the second call is a no-op.

        Returns `None` for a session that never saw a frame, which also means
        nothing is written: opening the camera and closing it again should not
        leave a file behind.
        """
        if self.closed:
            return None
        self.closed = True
        self.ended_at = _now()

        if self.frames == 0:
            return None

        record = self.summary()
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        path = SESSIONS_DIR / f"session_{self.session_id}.json"
        # Two sessions can start inside the same second - rare on a stopwatch,
        # not rare when testing - and the id is second-resolution. Suffix rather
        # than overwrite, and keep the id in step with the filename.
        suffix = 1
        while path.exists():
            suffix += 1
            self.session_id = f"{record['session_id']}_{suffix}"
            record["session_id"] = self.session_id
            path = SESSIONS_DIR / f"session_{self.session_id}.json"

        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        _append_to_index(record)
        return record


def _append_to_index(record: dict) -> None:
    """Add one row to `sessions.csv`, writing the header if the file is new."""
    fresh = not INDEX_CSV.exists()
    with INDEX_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if fresh:
            writer.writeheader()
        row = {k: record[k] for k in
               ("session_id", "started_at", "ended_at", "duration_seconds", "frames")}
        row.update(record["counts"])
        row["total"] = record["total"]
        writer.writerow(row)


def list_sessions(limit: int = 20) -> list[dict]:
    """Saved sessions, newest first - read back from the JSON files themselves.

    The files are the record, not a cache of one, so nothing has to be kept in
    memory across restarts and a session written by an earlier run still shows.
    """
    if not SESSIONS_DIR.is_dir():
        return []

    records = []
    for path in sorted(SESSIONS_DIR.glob("session_*.json"), reverse=True)[:limit]:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue  # a half-written file should not break the listing
    return records


__all__ = ["INDEX_CSV", "SESSIONS_DIR", "Session", "list_sessions"]
