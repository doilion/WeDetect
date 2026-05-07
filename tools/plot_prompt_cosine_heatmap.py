#!/usr/bin/env python
"""Plot pairwise cosine similarity heatmap for the 32 dev32 fullname prompts.

Highlights prompt clusters where cos > 0.97 — these are the families where
the contrastive head must rely entirely on visual features (text is a noop)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


SHORT_NAMES = (
    "respiratory tract-Neutrophil",
    "respiratory tract-Alveolar macrophages",
    "respiratory tract-Ciliated columnar epithelial cells",
    "respiratory tract-Lymphocyte",
    "respiratory tract-Impurity",
    "respiratory tract-Squamous epithelial cells",
    "respiratory tract-Diseased cells",
    "Serous effusion-Negative samples",
    "Serous effusion-Diseased cells",
    "Thyroid gland-PTC",
    "Thyroid gland-SPTC",
    "Thyroid gland-NS",
    "Thyroid gland-Macrophages",
    "Thyroid gland-AUC",
    "Thyroid gland-Negative samples",
    "Thyroid gland-FC",
    "Urine-NILM",
    "Urine-Negative",
    "Urine-SHGUC",
    "Urine-AUC",
    "Urine-Negative Degeneration",
    "Urine-HGUC",
    "TCT_CCD-normal",
    "TCT_CCD-ascus",
    "TCT_CCD-asch",
    "TCT_CCD-lsil",
    "TCT_CCD-hsil_scc_omn",
    "TCT_CCD-agc_adenocarcinoma_em",
    "TCT_CCD-vaginalis",
    "TCT_CCD-monilia",
    "TCT_CCD-dysbacteriosis_herpes_act",
    "TCT_CCD-ec",
)

ORGAN_RANGES = [
    ("respiratory tract", 0, 7, "#4C9AFF"),
    ("Serous effusion", 7, 9, "#7AC274"),
    ("Thyroid gland", 9, 16, "#F2A93B"),
    ("Urine", 16, 22, "#E0584C"),
    ("TCT_CCD", 22, 32, "#9E7BD0"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text-json", default="data/texts/tct_ngc_fullnames_32.json")
    p.add_argument("--text-emb", default="data/texts/tct_ngc_fullnames_32_embeddings_wedetect_tiny.pth")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.text_json) as f:
        groups = json.load(f)
    prompts = [g[0] for g in groups]
    if len(prompts) != 32:
        raise SystemExit(f"expected 32 prompts, got {len(prompts)}")

    bank = torch.load(args.text_emb, map_location="cpu", weights_only=False)
    vecs = torch.stack([bank[p] for p in prompts]).float()
    vecs = F.normalize(vecs, dim=1)
    cos = (vecs @ vecs.T).numpy()

    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(cos, vmin=0.85, vmax=1.0, cmap="RdYlGn_r", aspect="equal")

    n = 32
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SHORT_NAMES, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(SHORT_NAMES, fontsize=7)

    # Annotate values; bold if cos >= 0.97 (concerning)
    for i in range(n):
        for j in range(n):
            v = cos[i, j]
            if i == j:
                ax.text(j, i, "1.0", ha="center", va="center", fontsize=6, color="#aaaaaa")
            elif v >= 0.97:
                ax.text(j, i, f"{v:.2f}"[1:], ha="center", va="center",
                        fontsize=6, fontweight="bold", color="white")
            elif v >= 0.93:
                ax.text(j, i, f"{v:.2f}"[1:], ha="center", va="center",
                        fontsize=6, color="black")
            else:
                pass  # too low, skip to reduce visual noise

    # Draw organ boundary rectangles
    for name, lo, hi, color in ORGAN_RANGES:
        ax.add_patch(plt.Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo,
                                   fill=False, edgecolor=color, linewidth=2.5))
        # Annotate organ name above the diagonal block
        ax.text((lo + hi) / 2 - 0.5, lo - 1.2, name,
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.set_title("32-class prompt cosine similarity (XLM-R cached embeddings)\n"
                 "deep red = cos ≥ 0.97 (model can only use visual features to disambiguate)",
                 fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("cosine similarity")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)

    # Quick numeric summary of high-cos pairs
    high = []
    for i in range(n):
        for j in range(i + 1, n):
            if cos[i, j] >= 0.97:
                high.append((cos[i, j], SHORT_NAMES[i], SHORT_NAMES[j]))
    high.sort(reverse=True)
    print(f"wrote {out}")
    print(f"\n=== {len(high)} pairs with cos >= 0.97 ===")
    for v, a, b in high[:25]:
        print(f"  {v:.4f}  {a:<46} ↔ {b}")


if __name__ == "__main__":
    main()
