"""Figures for the evaluation write-up.

Three pictures, each answering a question a table cannot:

* **Learning curves** - what happened *during* training, which is where the
  shape of a result comes from. A single final number cannot distinguish a
  model still improving at the last epoch from one that peaked early.
* **Per-class AP** - whether a score is spread evenly across the five classes
  or carried by one of them.
* **Qualitative detections** - the same test images through all three models,
  which shows the failure *modes* behind the numbers.

Colours come from `CLASS_COLORS_RGB` so a class is the same colour here, in the
dataset sample sheets, and in the web app.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import (  # noqa: E402
    CLASS_COLORS_RGB,
    CLASSES,
    COCO_ID_TO_CLASS,
    DISPLAY_SCORE_THRESHOLD,
    FIGURES_DIR,
    RESULTS_DIR,
    RUNS_DIR,
)

#: One colour per model, stable across every figure.
MODEL_COLORS = {
    "ssdlite": "#1f77b4",
    "yolo11s": "#2ca02c",
    "dfine": "#d62728",
    "yolo26m": "#ff7f0e",
}

_RGB = {name: tuple(c / 255 for c in rgb) for name, rgb in CLASS_COLORS_RGB.items()}


def _epoch_series(model_key: str) -> tuple[list[int], list[float]]:
    """Validation mAP per epoch, from whichever schema the trainer wrote.

    The run directory comes from the index, not from the model key: a run
    started with `--tag` lives elsewhere, and plotting `runs/<key>` would draw
    a different run from the one the metrics beside it describe.
    """
    from src.config import PROJECT_ROOT
    from src.training.artifacts import read_index

    entry = read_index().get(model_key, {})
    run_dir = PROJECT_ROOT / entry["run_dir"] if entry.get("run_dir") else RUNS_DIR / model_key

    path = run_dir / "history.json"
    if not path.is_file():
        return [], []
    history = json.loads(path.read_text(encoding="utf-8"))
    epochs, values = [], []
    for e in history["epochs"]:
        value = e.get("val_mAP_50_95", e.get("metrics/mAP50-95(B)"))
        if value is not None:
            epochs.append(e["epoch"])
            values.append(float(value))
    return epochs, values


def learning_curves(payload: dict, path: Path | None = None) -> Path:
    """val mAP against epoch for all three models on one axis."""
    path = path or FIGURES_DIR / "eval_learning_curves.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    longest = 0
    for key, result in payload["models"].items():
        epochs, values = _epoch_series(key)
        if not epochs:
            continue
        longest = max(longest, epochs[-1])
        label = f"{result['display_name']} ({epochs[-1]} epoch)"
        ax.plot(epochs, values, marker="o", markersize=3.5, linewidth=1.8,
                color=MODEL_COLORS.get(key), label=label)
        # Mark where each model ended, since that is the checkpoint scored.
        ax.scatter([epochs[-1]], [values[-1]], s=55, zorder=5,
                   facecolor="white", edgecolor=MODEL_COLORS.get(key), linewidth=1.8)

    ax.set_xlabel("epoch")
    ax.set_ylabel("val mAP@[.5:.95]")
    ax.set_title("Validation mAP theo epoch", fontsize=11)
    # Runs no longer share an epoch count, so derive the ticks.
    step = max(1, longest // 16)
    ax.set_xticks(range(step, longest + 1, step))
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def per_class_ap(payload: dict, path: Path | None = None) -> Path:
    """Grouped bars: AP@[.5:.95] per class, per model, on the test split."""
    path = path or FIGURES_DIR / "eval_per_class_ap.png"
    models = list(payload["models"].values())
    x = np.arange(len(CLASSES))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for i, result in enumerate(models):
        heights = [result["test"]["per_class"][c]["AP_50_95"] for c in CLASSES]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width * 0.92,
                      color=MODEL_COLORS.get(result["model_key"]),
                      label=result["display_name"])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1.5)

    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("AP@[.5:.95]")
    ax.set_title(f"AP theo từng lớp — tập {payload['split']}", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_boxes(ax, image, records, title: str) -> None:
    """Draw COCO-style records (`bbox` in xywh, `category_id`, optional `score`).

    Public because the EDA notebook draws the same way; note that importing
    this module sets matplotlib's backend to Agg, so a notebook should follow
    the import with `%matplotlib inline`.
    """
    from matplotlib.patches import Rectangle

    ax.imshow(image)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    for r in records:
        x, y, w, h = r["bbox"]
        name = COCO_ID_TO_CLASS.get(r["category_id"], "?")
        colour = _RGB.get(name, (0.2, 0.2, 0.2))
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=colour, linewidth=1.6))
        label = name if "score" not in r else f"{name} {r['score']:.2f}"
        ax.text(x, max(y - 3, 8), label, fontsize=6, color="white",
                bbox=dict(facecolor=colour, edgecolor="none", pad=1.0, alpha=0.9))


def qualitative(
    payload: dict,
    n_images: int = 3,
    threshold: float = DISPLAY_SCORE_THRESHOLD,
    path: Path | None = None,
) -> Path:
    """Ground truth beside each model's detections, on the same test images.

    Images are picked deterministically by percentile of ground-truth box
    count, not by maximum. The densest images in this split are 25-box market
    stalls and photo collages that every model fails on, so showing only those
    would say more about the split than about the models.
    """
    from src.data.coco_dataset import CocoRecords

    path = path or FIGURES_DIR / "eval_qualitative.png"
    split = payload["split"]
    records = CocoRecords(split)

    by_density = sorted(
        (i for i in range(len(records)) if records.records[i]["annotations"]),
        key=lambda i: len(records.records[i]["annotations"]),
    )
    # Spread the picks across the distribution: sparse, typical, crowded.
    percentiles = np.linspace(0.35, 0.9, n_images)
    ranked = [by_density[min(int(p * len(by_density)), len(by_density) - 1)]
              for p in percentiles]

    checkpoint = payload.get("checkpoint", "last")
    detections: dict[str, dict[int, list]] = {}
    for key in payload["models"]:
        file = RESULTS_DIR / f"detections_{key}_{split}_{checkpoint}.json"
        if not file.is_file():
            continue
        grouped: dict[int, list] = {}
        for d in json.loads(file.read_text(encoding="utf-8")):
            if d["score"] >= threshold:
                grouped.setdefault(d["image_id"], []).append(d)
        detections[key] = grouped

    columns = 1 + len(detections)
    fig, axes = plt.subplots(
        len(ranked), columns, figsize=(3.1 * columns, 2.9 * len(ranked)), dpi=150
    )
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(ranked):
        record = records.records[idx]
        image = records.image(idx)
        truth = [
            {"bbox": a["bbox"], "category_id": a["category_id"]}
            for a in record["annotations"]
        ]
        draw_boxes(axes[row][0], image, truth, f"ground truth ({len(truth)} hộp)")
        for col, (key, grouped) in enumerate(detections.items(), start=1):
            found = grouped.get(record["image_id"], [])
            name = payload["models"][key]["display_name"]
            draw_boxes(axes[row][col], image, found, f"{name} ({len(found)} hộp)")

    fig.suptitle(
        f"Phát hiện trên tập {split} — ngưỡng score {threshold}", fontsize=10, y=0.999
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_all(payload: dict) -> list[Path]:
    made = [learning_curves(payload), per_class_ap(payload)]
    try:
        made.append(qualitative(payload))
    except Exception as exc:  # noqa: BLE001 - a missing figure must not fail the run
        print(f"[figures] qualitative skipped: {exc}")
    for p in made:
        print(f"[figures] {p}")
    return made
