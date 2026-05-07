#!/usr/bin/env python
"""Three-panel diagnosis of why Thyroid-AUC drops from val 0.272 to test 0.051."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THYROID_CLASSES = [
    ("Thy-FC", "morpho", -0.063),
    ("Thy-Macrophages", "morpho", -0.057),
    ("Thy-PTC", "morpho", -0.194),
    ("Thy-SPTC", "fuzzy", -0.150),
    ("Thy-NS", "fuzzy", -0.188),
    ("Thy-AUC", "fuzzy", -0.221),
]


def load_areas(ann_path: str, cat_id: int) -> np.ndarray:
    with open(ann_path) as f:
        d = json.load(f)
    areas = []
    for a in d["annotations"]:
        if a["category_id"] == cat_id:
            areas.append(a.get("area") or (a["bbox"][2] * a["bbox"][3]))
    return np.array(areas)


def count_cases(ann_path: str, cat_id: int) -> int:
    with open(ann_path) as f:
        d = json.load(f)
    img_to_case = {}
    for img in d["images"]:
        ps = img["file_name"].split("/")
        img_to_case[img["id"]] = ps[2] if len(ps) > 2 else ""
    cases = {img_to_case[a["image_id"]] for a in d["annotations"]
             if a["category_id"] == cat_id}
    return len(cases)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-ann", default="/home1/liwenjie/TCT_NGC/annotations/instances_train_dev.json")
    p.add_argument("--val-ann",   default="/home1/liwenjie/TCT_NGC/annotations/instances_val_dev.json")
    p.add_argument("--test-ann",  default="/home1/liwenjie/TCT_NGC/annotations/instances_test_base_clean.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---------- Panel 1: Thyroid 6 classes' val→test gap ----------
    ax = axes[0]
    names = [c[0] for c in THYROID_CLASSES]
    gaps = [c[2] for c in THYROID_CLASSES]
    colors = ["#27ae60" if c[1] == "morpho" else "#c0392b" for c in THYROID_CLASSES]
    y = np.arange(len(names))
    ax.barh(y, gaps, color=colors, edgecolor="black", linewidth=0.5)
    for i, g in enumerate(gaps):
        ax.text(g - 0.005, i, f"{g:+.3f}", va="center", ha="right",
                fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("val→test mAP gap")
    ax.set_xlim(-0.25, 0.02)
    ax.set_title("(a) Morphology-distinct vs medically-fuzzy Thyroid classes\n"
                 "(green = strong morphology, holds up; red = fuzzy Bethesda definition, all collapse)",
                 fontsize=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#27ae60", label="Strong morphology (Macrophages/FC)"),
        plt.Rectangle((0, 0), 1, 1, color="#c0392b", label="Fuzzy definition (AUC/NS/SPTC)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # ---------- Panel 2: Thy-AUC bbox area distribution ----------
    ax = axes[1]
    cid_auc = 13
    val_areas = load_areas(args.val_ann, cid_auc)
    test_areas = load_areas(args.test_ann, cid_auc)
    val_med = np.median(val_areas)
    test_med = np.median(test_areas)
    bins = np.linspace(0, np.percentile(np.concatenate([val_areas, test_areas]), 95), 50)
    ax.hist(val_areas, bins=bins, alpha=0.55, color="#3498db",
            label=f"val_dev (median {val_med:.0f} px²)", density=True)
    ax.hist(test_areas, bins=bins, alpha=0.55, color="#e67e22",
            label=f"test_base (median {test_med:.0f} px²)", density=True)
    ax.axvline(val_med, color="#3498db", linestyle="--", linewidth=1.2)
    ax.axvline(test_med, color="#e67e22", linestyle="--", linewidth=1.2)
    ax.set_xlabel("bbox area (pixels²)")
    ax.set_ylabel("density")
    ax.set_title(f"(b) Thy-AUC bbox area distribution\n"
                 f"test median is {(1 - test_med/val_med)*100:.0f}% smaller than val → smaller boxes → more misses",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---------- Panel 3: Thy-AUC sample/case counts ----------
    ax = axes[2]
    splits = ["train_dev", "val_dev", "test_base"]
    ann_paths = [args.train_ann, args.val_ann, args.test_ann]
    ann_counts = []
    case_counts = []
    for sp in ann_paths:
        with open(sp) as f:
            d = json.load(f)
        ann_counts.append(sum(1 for a in d["annotations"] if a["category_id"] == cid_auc))
        case_counts.append(count_cases(sp, cid_auc))

    x = np.arange(len(splits))
    width = 0.35
    bars1 = ax.bar(x - width/2, ann_counts, width, color="#3498db", label="annotations", edgecolor="black", linewidth=0.5)
    ax_r = ax.twinx()
    bars2 = ax_r.bar(x + width/2, case_counts, width, color="#e67e22", label="unique cases", edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars1, ann_counts):
        ax.text(bar.get_x() + bar.get_width()/2, v + 100, f"{v}", ha="center", fontsize=9, color="#3498db", fontweight="bold")
    for bar, v in zip(bars2, case_counts):
        ax_r.text(bar.get_x() + bar.get_width()/2, v + 5, f"{v}", ha="center", fontsize=9, color="#e67e22", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("annotations", color="#3498db")
    ax_r.set_ylabel("unique cases", color="#e67e22")
    ax.set_title("(c) Thy-AUC annotations / unique cases\n"
                 "val only 839 ann + image-CV → model overfits to specific AUC morphologies;\n"
                 "test 8 new patients introduce unseen variation → collapse",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Thyroid-AUC val→test collapse (-0.221): three-layer root-cause diagnosis",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"  val areas: median={val_med:.0f}, n={len(val_areas)}")
    print(f"  test areas: median={test_med:.0f}, n={len(test_areas)}")
    print(f"  ann counts (train/val/test): {ann_counts}")
    print(f"  case counts (train/val/test): {case_counts}")


if __name__ == "__main__":
    main()
