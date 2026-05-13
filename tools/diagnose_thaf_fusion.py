#!/usr/bin/env python
"""Diagnose why THAF (Trainable Hierarchical Attribute Fusion) collapses on
novel classes despite improving base mAP.

Phase 3 eval (2026-05-10) showed:
  - THAF + BiomedCLIP base 25-cls = 0.327 (+1.7 vs clean dev30 baseline 0.310)
  - THAF + BiomedCLIP avg novel mAP  = 0.052 (vs v2 baseline 0.095 — collapsed)
  - THAF + XLM-R avg novel mAP        = 0.020 (worse collapse)

This tool loads a trained THAF backbone, runs the fusion module on all 39
classes' (30 base + 9 novel) 5-attribute inputs, and inspects the geometric
properties of the fused class vectors. Decision tree per plan Phase 3.5
identifies which of hypotheses A-E is most consistent with the data.

Outputs to --out-dir:
  - cosine_heatmap_trained.png      (39×39 trained fusion cosine)
  - cosine_heatmap_attr_mean.png    (39×39 untrained attr-mean baseline)
  - summary.json                    (structured stats)
  - summary.md                      (human-readable + decision tree hits)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmengine.registry import MODELS


ATTR_FIELDS = (
    "organ_specimen",
    "diagnostic_code",
    "cytomorphology",
    "background_and_immunoprofile",
    "key_distinguishing_feature",
)

# Phase 2.1 cos heatmap reference numbers (already on disk, cited for context).
PHASE2_REFERENCES = {
    "v2_psc_single_prompt": 0.996,
    "5attr_static_sum": 0.993,
    "5attr_static_weighted": 0.991,
    "5attr_static_concat": 0.971,
    "5attr_static_only_distinguish": 0.971,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="THAF config (xlmr or biomedclip)")
    p.add_argument("--checkpoint", required=True, help="THAF best ckpt")
    p.add_argument(
        "--attr-json",
        default="data/texts/tct_ngc_fullnames_39_attr.json",
        help="dict[class_name -> dict[attr_field -> str]] for 39 classes",
    )
    p.add_argument(
        "--base-json",
        default="data/texts/tct_ngc_attr_base30.json",
        help="list-of-list of base class names in cat_id order",
    )
    p.add_argument(
        "--novel-json",
        default="data/texts/tct_ngc_attr_novel9.json",
        help="list-of-list of novel class names",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cpu", help="forward device (cpu is enough)")
    return p.parse_args()


def load_class_order(base_json: Path, novel_json: Path) -> tuple[list[str], int]:
    """Returns (all_class_names_in_base_then_novel_order, n_base)."""
    base = json.loads(base_json.read_text(encoding="utf-8"))
    novel = json.loads(novel_json.read_text(encoding="utf-8"))
    base_names = [grp[0] for grp in base]
    novel_names = [grp[0] for grp in novel]
    return base_names + novel_names, len(base_names)


def build_pseudo_backbone(cfg: Config) -> torch.nn.Module:
    """Build only the Pseudo*Hierarchical backbone (skip building full detector)."""
    text_cfg = cfg.model.backbone.text_model.copy()
    if "attr_emb_cache_path" not in text_cfg:
        raise SystemExit(
            f"config text_model {text_cfg.get('type')!r} doesn't use "
            f"attr_emb_cache_path — this tool only handles THAF backbones."
        )
    return MODELS.build(text_cfg)


def load_thaf_state(checkpoint: Path, backbone: torch.nn.Module) -> None:
    """Load `backbone.text_model.*` weights from a full-detector ckpt."""
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    prefix = "backbone.text_model."
    text_state = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    if not text_state:
        raise SystemExit(f"no {prefix}* keys in {checkpoint}")
    incompat = backbone.load_state_dict(text_state, strict=False)
    # Cache attr_emb_table is a buffer set at __init__ — expect it in missing.
    expected_missing = {"attr_emb_table"}
    actually_missing = set(incompat.missing_keys) - expected_missing
    if actually_missing:
        raise SystemExit(
            f"unexpectedly missing keys after load: {sorted(actually_missing)[:5]}..."
        )
    if incompat.unexpected_keys:
        raise SystemExit(
            f"unexpected keys in ckpt: {sorted(incompat.unexpected_keys)[:5]}..."
        )
    print(f"[load] loaded {len(text_state)} param tensors; fusion ready")


def build_text_input(
    attr: dict, class_names: list[str]
) -> list[list[list[str]]]:
    """Build [batch=1, num_classes=N, num_attrs=5] nested list of strings."""
    return [[[attr[c][f].strip() for f in ATTR_FIELDS] for c in class_names]]


def get_attr_mean_vectors(
    backbone: torch.nn.Module, text: list[list[list[str]]]
) -> torch.Tensor:
    """Return the L2-normalized attr-mean baseline (alpha=0 equivalent).

    Mimics what fusion would output if alpha were 0: just mean-pool the 5
    cached per-attr embeddings per class, then L2-normalize.
    """
    idx = backbone._lookup_indices(text).to(backbone.attr_emb_table.device)
    B = len(text)
    C = len(text[0])
    A = backbone.num_attr_types
    D = backbone.embed_dim
    attr = backbone.attr_emb_table.index_select(0, idx).reshape(B, C, A, D)
    mean = attr.mean(dim=2)  # [B, C, D]
    return F.normalize(mean, dim=-1)


def cos_stats(
    matrix: np.ndarray, n_base: int
) -> dict:
    """Compute base↔base / novel↔novel / novel↔base cosine stats."""
    n = matrix.shape[0]
    base_block = matrix[:n_base, :n_base]
    novel_block = matrix[n_base:, n_base:]
    cross_block = matrix[n_base:, :n_base]

    # off-diagonal of base/novel blocks (exclude self-cosine = 1.0)
    base_offdiag = base_block[~np.eye(n_base, dtype=bool)]
    novel_offdiag = novel_block[~np.eye(n - n_base, dtype=bool)]
    cross_flat = cross_block.flatten()

    return {
        "base_base_off_diag": {
            "mean": float(base_offdiag.mean()),
            "max": float(base_offdiag.max()),
            "std": float(base_offdiag.std()),
        },
        "novel_novel_off_diag": {
            "mean": float(novel_offdiag.mean()),
            "max": float(novel_offdiag.max()),
            "std": float(novel_offdiag.std()),
        },
        "novel_base": {
            "mean": float(cross_flat.mean()),
            "max": float(cross_flat.max()),
        },
    }


def plot_heatmap(
    cos: np.ndarray,
    class_names: list[str],
    n_base: int,
    title: str,
    out_path: Path,
) -> None:
    n = cos.shape[0]
    fig, ax = plt.subplots(figsize=(0.32 * n + 4, 0.32 * n + 4))
    im = ax.imshow(cos, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    labels = [f"[base] {c[:32]}" for c in class_names[:n_base]] + [
        f"[novel] {c[:32]}" for c in class_names[n_base:]
    ]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axhline(y=n_base - 0.5, color="red", lw=1, alpha=0.7)
    ax.axvline(x=n_base - 0.5, color="red", lw=1, alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="cosine")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def decide_hypothesis(
    trained_stats: dict, mean_stats: dict, alpha: float
) -> list[str]:
    """Apply Phase 3.5 decision tree — return list of hypothesis hits."""
    hits = []
    nn_t = trained_stats["novel_novel_off_diag"]["max"]
    nn_m = mean_stats["novel_novel_off_diag"]["max"]
    bb_t_mean = trained_stats["base_base_off_diag"]["mean"]
    nb_t_mean = trained_stats["novel_base"]["mean"]

    # A1: novel novel collapses above attr-mean baseline
    if nn_t > 0.99 and nn_t > nn_m:
        hits.append(
            f"**A confirmed (fusion collapses novel)**: "
            f"novel↔novel max cos trained={nn_t:.3f} > {nn_m:.3f} (attr_mean)"
        )
    elif nn_t < nn_m:
        hits.append(
            f"**A refuted (fusion separates novel better)**: "
            f"trained novel↔novel max {nn_t:.3f} < attr_mean {nn_m:.3f}"
        )
    else:
        hits.append(
            f"A inconclusive on novel↔novel max: "
            f"trained {nn_t:.3f}, attr_mean {nn_m:.3f}"
        )

    # A2: novel→base cosine is high vs base→base
    if nb_t_mean > bb_t_mean + 0.05:
        hits.append(
            f"**A strong: novel→base avg cos {nb_t_mean:.3f} > base↔base avg {bb_t_mean:.3f} + 0.05** "
            f"(novel vectors pulled into base cluster)"
        )
    elif nb_t_mean < bb_t_mean:
        hits.append(
            f"novel→base avg {nb_t_mean:.3f} ≤ base↔base avg {bb_t_mean:.3f} — "
            f"novel not pulled into base cluster"
        )

    # A3: alpha
    if alpha > 0.9:
        hits.append(
            f"**A strong: alpha={alpha:.3f} > 0.9** — fusion almost fully "
            f"overrides attr_mean, projection learned base-specific features"
        )
    elif 0.5 < alpha <= 0.9:
        hits.append(f"alpha={alpha:.3f} (init=0.3) — fusion-dominant but coexisting with attr_mean")
    else:
        hits.append(f"alpha={alpha:.3f} ≈ init — fusion light, attr_mean dominant")

    return hits


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Class order: 30 base + 9 novel
    class_names, n_base = load_class_order(
        Path(args.base_json), Path(args.novel_json)
    )
    n = len(class_names)
    print(f"[setup] {n_base} base + {n - n_base} novel = {n} classes")

    # 2) Load 5-attr dict
    attr = json.loads(Path(args.attr_json).read_text(encoding="utf-8"))
    for c in class_names:
        if c not in attr:
            raise SystemExit(f"class {c!r} missing from attr JSON")

    # 3) Build trained THAF backbone (PseudoHierarchical*LanguageBackbone)
    cfg = Config.fromfile(args.config)
    backbone = build_pseudo_backbone(cfg)
    load_thaf_state(Path(args.checkpoint), backbone)
    backbone.to(args.device).eval()

    encoder_name = type(backbone).__name__.replace("PseudoHierarchical", "").replace(
        "LanguageBackbone", ""
    ).lower()
    embed_dim = backbone.embed_dim

    # 4) Forward fusion for all 39 classes in one batch
    text = build_text_input(attr, class_names)  # [1][39][5]
    with torch.no_grad():
        fused = backbone(text).squeeze(0).cpu().numpy()  # [39, D]
        # also compute attr-mean baseline (alpha=0 equivalent)
        mean_vec = get_attr_mean_vectors(backbone, text).squeeze(0).cpu().numpy()

    # Already L2-normalized by both forwards
    cos_trained = fused @ fused.T  # [39, 39]
    cos_mean = mean_vec @ mean_vec.T

    # 5) Stats
    trained_stats = cos_stats(cos_trained, n_base)
    mean_stats = cos_stats(cos_mean, n_base)
    alpha_val = float(backbone.alpha.item())
    attr_type_norms = [
        float(backbone.attr_type_embed.weight[i].norm().item())
        for i in range(backbone.num_attr_types)
    ]

    # 6) Decision tree
    hypothesis_hits = decide_hypothesis(trained_stats, mean_stats, alpha_val)

    # 7) Plots
    plot_heatmap(
        cos_trained,
        class_names,
        n_base,
        f"THAF {encoder_name} ({embed_dim}d) — trained fusion cosine (39 classes)",
        out_dir / "cosine_heatmap_trained.png",
    )
    plot_heatmap(
        cos_mean,
        class_names,
        n_base,
        f"THAF {encoder_name} — attr-mean baseline (alpha=0 equivalent)",
        out_dir / "cosine_heatmap_attr_mean.png",
    )

    # 8) JSON summary
    summary = {
        "encoder": encoder_name,
        "checkpoint": str(args.checkpoint),
        "embed_dim": embed_dim,
        "num_classes_total": n,
        "num_base": n_base,
        "num_novel": n - n_base,
        "alpha_trained": alpha_val,
        "attr_type_embed_l2_norms": attr_type_norms,
        "attr_field_order": list(ATTR_FIELDS),
        "trained_fusion": trained_stats,
        "attr_mean_baseline": mean_stats,
        "phase2_references_max_novel_novel_cos": PHASE2_REFERENCES,
        "hypothesis_hits": hypothesis_hits,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[json] wrote {out_dir/'summary.json'}")

    # 9) Markdown summary
    md_lines = [
        f"# THAF fusion diagnostic — {encoder_name} ({embed_dim}d)",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Classes analyzed: {n_base} base + {n - n_base} novel = {n}",
        "",
        "## Trained fusion stats",
        "",
        f"- alpha (learnable residual weight): **{alpha_val:.4f}** (init=0.3)",
        f"- attr_type_embed L2 norms (5 channels in order "
        f"{', '.join(ATTR_FIELDS)}): "
        f"{[round(x, 4) for x in attr_type_norms]}",
        "",
        f"| metric | trained fusion | attr-mean baseline | Δ |",
        f"|---|---:|---:|---:|",
        f"| base↔base off-diag mean cos | "
        f"{trained_stats['base_base_off_diag']['mean']:.3f} | "
        f"{mean_stats['base_base_off_diag']['mean']:.3f} | "
        f"{trained_stats['base_base_off_diag']['mean'] - mean_stats['base_base_off_diag']['mean']:+.3f} |",
        f"| base↔base off-diag max  cos | "
        f"{trained_stats['base_base_off_diag']['max']:.3f} | "
        f"{mean_stats['base_base_off_diag']['max']:.3f} | "
        f"{trained_stats['base_base_off_diag']['max'] - mean_stats['base_base_off_diag']['max']:+.3f} |",
        f"| novel↔novel off-diag mean cos | "
        f"{trained_stats['novel_novel_off_diag']['mean']:.3f} | "
        f"{mean_stats['novel_novel_off_diag']['mean']:.3f} | "
        f"{trained_stats['novel_novel_off_diag']['mean'] - mean_stats['novel_novel_off_diag']['mean']:+.3f} |",
        f"| novel↔novel off-diag max  cos | "
        f"**{trained_stats['novel_novel_off_diag']['max']:.3f}** | "
        f"{mean_stats['novel_novel_off_diag']['max']:.3f} | "
        f"{trained_stats['novel_novel_off_diag']['max'] - mean_stats['novel_novel_off_diag']['max']:+.3f} |",
        f"| novel→base avg cos | "
        f"{trained_stats['novel_base']['mean']:.3f} | "
        f"{mean_stats['novel_base']['mean']:.3f} | "
        f"{trained_stats['novel_base']['mean'] - mean_stats['novel_base']['mean']:+.3f} |",
        "",
        "## Phase 2 reference (Phase 2.1 cos heatmap, single-encoder static aggregation)",
        "",
        f"| method | novel↔novel max cos |",
        f"|---|---:|",
    ]
    for k, v in PHASE2_REFERENCES.items():
        md_lines.append(f"| {k} | {v:.3f} |")
    md_lines.extend(
        [
            "",
            "## Decision tree hits",
            "",
        ]
    )
    for h in hypothesis_hits:
        md_lines.append(f"- {h}")
    md_lines.extend(
        [
            "",
            "## Plots",
            "",
            f"- `cosine_heatmap_trained.png` — 39×39 cosine, red lines split base/novel",
            f"- `cosine_heatmap_attr_mean.png` — alpha=0 equivalent (untrained fusion)",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[md] wrote {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
