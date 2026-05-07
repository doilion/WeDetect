#!/usr/bin/env python
"""Visualize the patient/case-level overlap between train_dev / val_dev / test_base_clean.

Drives Q1 of the analysis: val is in-distribution (image-level CV within shared cohort);
test_base_clean is the true patient-level hold-out."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_cases(ann_path: str, exclude_organs: tuple[str, ...] = ("TCT_CCD",)) -> set[str]:
    """Return unique case IDs from path parts[2], skipping organs that lack WSI info."""
    with open(ann_path) as f:
        d = json.load(f)
    cases = set()
    for img in d["images"]:
        organ = img["file_name"].split("/")[0]
        if organ in exclude_organs:
            continue
        ps = img["file_name"].split("/")
        if len(ps) > 2:
            cases.add(ps[2])
    return cases


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="/home1/liwenjie/TCT_NGC/annotations/instances_train_dev.json")
    p.add_argument("--val",   default="/home1/liwenjie/TCT_NGC/annotations/instances_val_dev.json")
    p.add_argument("--test",  default="/home1/liwenjie/TCT_NGC/annotations/instances_test_base_clean.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    train = load_cases(args.train)
    val = load_cases(args.val)
    test = load_cases(args.test)

    n_train = len(train)
    n_val = len(val)
    n_test = len(test)
    n_train_only = len(train - val - test)
    n_val_in_train = len(val & train)
    n_test_in_train = len(test & train)
    n_test_only = len(test - train - val)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Horizontal stacked-bar visualization
    bar_h = 0.55
    y_train, y_val, y_test = 2.0, 1.0, 0.0

    # train_dev: solid blue, full width
    ax.barh(y_train, n_train, height=bar_h, color="#3498db", edgecolor="black", linewidth=0.8)
    ax.text(n_train / 2, y_train, f"train_dev: {n_train} cases",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    # val_dev: green where in train (overlap), light otherwise
    ax.barh(y_val, n_val_in_train, height=bar_h, color="#e74c3c", alpha=0.7,
            edgecolor="black", linewidth=0.8, hatch="///",
            label=f"val_dev ⊂ train_dev ({n_val_in_train} cases)")
    if n_val - n_val_in_train > 0:
        ax.barh(y_val, n_val - n_val_in_train, left=n_val_in_train, height=bar_h,
                color="#7AC274", edgecolor="black", linewidth=0.8,
                label=f"val_dev unique ({n_val - n_val_in_train} cases)")
    ax.text(n_val / 2, y_val, f"val_dev: {n_val} cases\n({n_val_in_train}/{n_val}={n_val_in_train/n_val*100:.0f}% same patients as train)",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    # test_base: mostly disjoint
    if n_test_in_train > 0:
        ax.barh(y_test, n_test_in_train, height=bar_h, color="#e74c3c", alpha=0.7,
                edgecolor="black", linewidth=0.8, hatch="///")
    ax.barh(y_test, n_test - n_test_in_train, left=n_test_in_train, height=bar_h,
            color="#7AC274", edgecolor="black", linewidth=0.8)
    ax.text(n_test / 2, y_test, f"test_base_clean: {n_test} cases\n"
            f"({n_test_in_train}/{n_test}={n_test_in_train/n_test*100:.1f}% overlap, true hold-out)",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_yticks([y_train, y_val, y_test])
    ax.set_yticklabels(["train_dev\n(116k imgs)", "val_dev\n(13k imgs)", "test_base_clean\n(26k imgs)"],
                       fontsize=11)
    ax.set_xlabel("unique patient cases (TCT_CCD excluded — no WSI info)")
    ax.set_title("Patient-cohort overlap — val is image-level CV within shared cohort; test_base is true hold-out\n"
                 "↳ this is why val mAP 0.413 is structurally optimistic vs test mAP 0.323 by 0.09",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#3498db",
                      label=f"train_dev ({n_train} cases)"),
        plt.Rectangle((0, 0), 1, 1, color="#e74c3c", alpha=0.7, hatch="///",
                      label="Same patients as train (image-level 9:1 split)"),
        plt.Rectangle((0, 0), 1, 1, color="#7AC274",
                      label="Patient-level new (true hold-out)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"  train: {n_train}, val: {n_val} ({n_val_in_train} in train), test: {n_test} ({n_test_in_train} in train)")


if __name__ == "__main__":
    main()
