"""Figures that describe the dataset, for chapter 1 of the report.

Four pictures, each answering a question about the data rather than about a
model:

* **Class distribution** - how uneven the five classes are, which is the first
  thing to check before reading any per-class AP.
* **Instances per image** - whether a typical image holds a couple of objects
  or a crowd of them; the crowded tail is where every model loses recall.
* **Object scale** - the split into COCO's small/medium/large buckets. The
  small-object share is what makes SSDLite's 320x320 input expensive.
* **Centre heatmap** - where objects sit in the frame.

The notebook `01_eda.ipynb` renders the same four inline by calling these
functions, so a chart in the report and its counterpart in the notebook cannot
drift apart.

Class colours come from `CLASS_COLORS_RGB`, shared with the evaluation figures
and the web app.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import (  # noqa: E402
    CLASS_COLORS_RGB,
    CLASSES,
    COCO_ID_TO_CLASS,
    FIGURES_DIR,
    SPLITS,
)
from src.data.coco_dataset import CocoRecords  # noqa: E402

#: Class colours as matplotlib 0-1 floats.
_RGB = {name: tuple(c / 255 for c in rgb) for name, rgb in CLASS_COLORS_RGB.items()}

#: One colour per split, stable across every figure here.
SPLIT_COLORS = {"train": "#4c72b0", "val": "#dd8452", "test": "#55a868"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_records(splits: tuple[str, ...] = SPLITS) -> dict[str, CocoRecords]:
    """`{split: CocoRecords}` - reading the annotations is the slow part, so a
    caller rendering several figures should load once and pass the result in."""
    return {split: CocoRecords(split) for split in splits}


def _resolve(records: dict[str, CocoRecords] | None) -> dict[str, CocoRecords]:
    return records if records is not None else load_records()


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Statistics - returned alongside the figures so the report can quote numbers
# that came from the same pass that drew the picture.
# --------------------------------------------------------------------------- #

def split_overview(records: dict[str, CocoRecords] | None = None) -> dict[str, dict]:
    """Images, instances and empty images per split, plus a `total` row."""
    records = _resolve(records)
    out: dict[str, dict] = {}
    for split, rec in records.items():
        anns = [a for r in rec.records for a in r["annotations"]]
        out[split] = {
            "images": len(rec),
            "instances": len(anns),
            "instances_per_image": round(len(anns) / len(rec), 2),
            "images_without_object": sum(1 for r in rec.records if not r["annotations"]),
        }
    total_img = sum(v["images"] for v in out.values())
    total_inst = sum(v["instances"] for v in out.values())
    out["total"] = {
        "images": total_img,
        "instances": total_inst,
        "instances_per_image": round(total_inst / total_img, 2),
        "images_without_object": sum(v["images_without_object"] for v in out.values()),
    }
    return out


def class_counts(records: dict[str, CocoRecords] | None = None) -> dict[str, dict[str, int]]:
    """`{split: {class: n_instances}}`, every class present even at zero."""
    records = _resolve(records)
    out = {}
    for split, rec in records.items():
        counts = dict.fromkeys(CLASSES, 0)
        for r in rec.records:
            for a in r["annotations"]:
                counts[COCO_ID_TO_CLASS[a["category_id"]]] += 1
        out[split] = counts
    return out


def scale_buckets(records: dict[str, CocoRecords] | None = None,
                  split: str = "train") -> dict[str, int]:
    """Instance counts in COCO's three area buckets, on one split.

    COCO's own mAP breakdown uses these thresholds, so the share of the small
    bucket is what the AP-small column is measured over.
    """
    records = _resolve(records)
    areas = np.array([
        a["bbox"][2] * a["bbox"][3]
        for r in records[split].records for a in r["annotations"]
        if a["bbox"][2] > 0 and a["bbox"][3] > 0
    ])
    return {
        "small": int((areas < 32 ** 2).sum()),
        "medium": int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum()),
        "large": int((areas >= 96 ** 2).sum()),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def class_distribution(records: dict[str, CocoRecords] | None = None,
                       path: Path | None = None) -> Path:
    """Instances per class: train alone, then stacked across the three splits."""
    path = path or FIGURES_DIR / "data_class_distribution.png"
    counts = class_counts(records)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)

    train = [counts["train"][c] for c in CLASSES]
    bars = axes[0].bar(CLASSES, train, color=[_RGB[c] for c in CLASSES])
    axes[0].bar_label(bars, fmt="%d", fontsize=8, padding=2)
    axes[0].set_title("Số instance mỗi lớp — tập train", fontsize=11)
    axes[0].set_ylabel("số instance")

    bottom = np.zeros(len(CLASSES))
    for split in SPLITS:
        values = np.array([counts[split][c] for c in CLASSES], dtype=float)
        axes[1].bar(CLASSES, values, bottom=bottom, label=split,
                    color=SPLIT_COLORS.get(split))
        bottom += values
    axes[1].set_title("Số instance mỗi lớp, chia theo split", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, path)


def instances_per_image(records: dict[str, CocoRecords] | None = None,
                        path: Path | None = None) -> Path:
    """How many objects an image holds: histogram on train, boxplot per split."""
    path = path or FIGURES_DIR / "data_instances_per_image.png"
    records = _resolve(records)
    per_image = {s: np.array([len(r["annotations"]) for r in rec.records])
                 for s, rec in records.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)

    axes[0].hist(per_image["train"], bins=range(0, 41), color=SPLIT_COLORS["train"])
    axes[0].set_xlabel("số instance trong một ảnh")
    axes[0].set_ylabel("số ảnh")
    axes[0].set_title("Phân bố trên tập train (cắt ở 40)", fontsize=11)

    axes[1].boxplot([per_image[s] for s in SPLITS], tick_labels=list(SPLITS),
                    showfliers=False)
    axes[1].set_ylabel("số instance trong một ảnh")
    axes[1].set_title("So sánh giữa các split", fontsize=11)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, path)


def object_scale(records: dict[str, CocoRecords] | None = None,
                 path: Path | None = None, split: str = "train") -> Path:
    """Object size: COCO buckets, box area relative to the image, aspect ratio."""
    path = path or FIGURES_DIR / "data_object_scale.png"
    records = _resolve(records)

    rel_areas, ratios = [], []
    for r in records[split].records:
        for a in r["annotations"]:
            w, h = a["bbox"][2], a["bbox"][3]
            if w <= 0 or h <= 0:
                continue
            rel_areas.append((w * h) / (r["width"] * r["height"]))
            ratios.append(w / h)

    buckets = scale_buckets(records, split)
    total = sum(buckets.values())
    labels = ["nhỏ\n(<32×32)", "trung bình", "lớn\n(>96×96)"]
    values = [buckets["small"], buckets["medium"], buckets["large"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150)

    bars = axes[0].bar(labels, values, color=["#c44e52", "#dd8452", "#55a868"])
    axes[0].bar_label(bars, labels=[f"{v}\n{100 * v / total:.1f}%" for v in values],
                      fontsize=8, padding=2)
    axes[0].set_title("Số instance theo nhóm kích thước COCO", fontsize=11)
    axes[0].set_ylabel("số instance")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].margins(y=0.18)

    axes[1].hist(np.sqrt(rel_areas), bins=50, color=SPLIT_COLORS["train"])
    axes[1].set_xlabel("căn bậc hai của tỉ lệ diện tích hộp trên ảnh")
    axes[1].set_ylabel("số instance")
    axes[1].set_title("Kích thước hộp so với ảnh", fontsize=11)

    axes[2].hist(np.clip(ratios, 0, 4), bins=50, color="#8172b3")
    axes[2].set_xlabel("chiều rộng / chiều cao")
    axes[2].set_title("Tỉ lệ khung hình của hộp", fontsize=11)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, path)


def center_heatmap(records: dict[str, CocoRecords] | None = None,
                   path: Path | None = None, split: str = "train") -> Path:
    """Where box centres fall, in relative image coordinates."""
    path = path or FIGURES_DIR / "data_center_heatmap.png"
    records = _resolve(records)

    cx, cy = [], []
    for r in records[split].records:
        for a in r["annotations"]:
            x, y, w, h = a["bbox"]
            cx.append((x + w / 2) / r["width"])
            cy.append((y + h / 2) / r["height"])

    fig, ax = plt.subplots(figsize=(4.8, 4.3), dpi=150)
    hist = ax.hist2d(cx, cy, bins=40, range=[[0, 1], [0, 1]], cmap="viridis")
    ax.invert_yaxis()  # image coordinates: y grows downwards
    ax.set_xlabel("toạ độ x tương đối")
    ax.set_ylabel("toạ độ y tương đối")
    ax.set_title(f"Mật độ tâm hộp — tập {split}", fontsize=11)
    ax.grid(False)
    fig.colorbar(hist[3], ax=ax, shrink=0.8)

    fig.tight_layout()
    return _save(fig, path)


def build_all(records: dict[str, CocoRecords] | None = None) -> list[Path]:
    """Render every dataset figure into `reports/figures/`."""
    records = _resolve(records)
    return [
        class_distribution(records),
        instances_per_image(records),
        object_scale(records),
        center_heatmap(records),
    ]
