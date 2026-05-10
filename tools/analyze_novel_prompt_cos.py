#!/usr/bin/env python
"""Diagnose whether the XLM-Roberta text encoder is the novel zero-shot
bottleneck by computing pairwise prompt cosine across base + novel classes.

Decision rule (from external OVD reviewer + this session's experiments):
- if fine-grained novel pairs (Bethesda V vs VI, Adeno vs SCC vs Small Cell)
  have cos > 0.90 in XLM-Roberta space, the text encoder cannot disambiguate
  these classes regardless of inference tricks → switch to BiomedCLIP /
  PubMedBERT (item 15)
- if cos < 0.85 across novel↔base and novel↔novel but mAP is still bad,
  the bottleneck is in image-text alignment, not text → visual prompt (item 13)
- if cos sits in [0.85, 0.95], both routes are worth trying

Outputs:
- ${OUT_DIR}/novel_prompt_cos_heatmap.png — full heatmap
- ${OUT_DIR}/novel_prompt_cos_finegrained.png — focused subview on novel-only
- ${OUT_DIR}/novel_prompt_cos_summary.txt — top collisions + decision flag
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-emb",
        default="data/texts/tct_ngc_fullnames_30_embeddings_wedetect_tiny.pth",
        help="dev30 base 30-class embedding cache",
    )
    p.add_argument(
        "--base-json",
        default="data/texts/tct_ngc_fullnames_30.json",
        help="dev30 base prompt JSON (drives display order)",
    )
    p.add_argument(
        "--novel-embs",
        nargs="+",
        default=[
            "data/texts/tct_ngc_novel_main_3_emb.pth",
            "data/texts/tct_ngc_novel_pseudo_2_emb.pth",
            "data/texts/tct_ngc_novel_hard_4_emb.pth",
        ],
        help="novel embedding caches; full_5 is omitted (subset of main_3+pseudo_2)",
    )
    p.add_argument(
        "--novel-jsons",
        nargs="+",
        default=[
            "data/texts/tct_ngc_novel_main_3.json",
            "data/texts/tct_ngc_novel_pseudo_2.json",
            "data/texts/tct_ngc_novel_hard_4.json",
        ],
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--collision-threshold", type=float, default=0.95)
    return p.parse_args()


def short_label(prompt: str, max_len: int = 36) -> str:
    """Compact display label for axis ticks. Strips standard prefixes."""
    for prefix in (
        "Positive for malignancy (PSC Category VI: Malignant): ",
        "Malignant, secondary (MAL-S): ",
    ):
        if prompt.startswith(prefix):
            return prompt[len(prefix):][:max_len]
    return prompt[:max_len]


def load_class_vectors(emb_path: str, json_path: str) -> tuple[list[str], np.ndarray]:
    """Returns (display_labels, [N, D] matrix) in JSON order."""
    groups = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cache: dict[str, torch.Tensor] = torch.load(emb_path, map_location="cpu")
    labels = []
    vecs = []
    for grp in groups:
        primary = grp[0]  # LoadText pattern
        if primary not in cache:
            raise KeyError(f"prompt {primary!r} not in cache {emb_path}")
        labels.append(short_label(primary))
        vecs.append(cache[primary].numpy())
    return labels, np.stack(vecs, axis=0)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_labels, base_vecs = load_class_vectors(args.base_emb, args.base_json)
    novel_labels: list[str] = []
    novel_vecs_list: list[np.ndarray] = []
    novel_split_ranges: list[tuple[str, int, int]] = []  # (split_name, start, end)
    for emb_path, json_path in zip(args.novel_embs, args.novel_jsons):
        labels, vecs = load_class_vectors(emb_path, json_path)
        split = Path(emb_path).stem.replace("tct_ngc_novel_", "").replace("_emb", "")
        start = len(novel_labels)
        novel_labels.extend(f"[{split}] {x}" for x in labels)
        novel_vecs_list.append(vecs)
        novel_split_ranges.append((split, start, len(novel_labels)))
    novel_vecs = np.concatenate(novel_vecs_list, axis=0)

    # Stack all in: 30 base classes first, then novel
    all_labels = [f"[base] {x}" for x in base_labels] + novel_labels
    all_vecs = np.concatenate([base_vecs, novel_vecs], axis=0)
    n_base = base_vecs.shape[0]
    n_novel = novel_vecs.shape[0]

    # L2-normalize for cosine
    norm = np.linalg.norm(all_vecs, axis=-1, keepdims=True)
    unit_vecs = all_vecs / np.clip(norm, 1e-8, None)
    cos = unit_vecs @ unit_vecs.T  # [N, N]

    # ---- Plot 1: full heatmap (base + novel) ----
    n = cos.shape[0]
    fig, ax = plt.subplots(figsize=(0.32 * n + 4, 0.32 * n + 4))
    im = ax.imshow(cos, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(all_labels, fontsize=7)
    ax.axhline(y=n_base - 0.5, color="red", lw=1, alpha=0.6)
    ax.axvline(x=n_base - 0.5, color="red", lw=1, alpha=0.6)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="cosine similarity")
    ax.set_title(
        f"Prompt cosine similarity — base ({n_base}) + novel ({n_novel}) = {n} classes\n"
        f"(red lines split base / novel quadrants)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "novel_prompt_cos_heatmap.png", dpi=160)
    plt.close(fig)

    # ---- Plot 2: novel-only finegrained (the one that decides direction) ----
    novel_cos = cos[n_base:, n_base:]
    fig, ax = plt.subplots(figsize=(0.45 * n_novel + 3, 0.45 * n_novel + 3))
    im = ax.imshow(novel_cos, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(n_novel))
    ax.set_yticks(range(n_novel))
    ax.set_xticklabels(novel_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(novel_labels, fontsize=8)
    for i in range(n_novel):
        for j in range(n_novel):
            ax.text(
                j, i, f"{novel_cos[i, j]:.2f}",
                ha="center", va="center",
                fontsize=7,
                color="white" if novel_cos[i, j] < 0.85 else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title(f"Novel↔novel cosine ({n_novel} classes) — fine-grained collision check")
    fig.tight_layout()
    fig.savefig(out_dir / "novel_prompt_cos_finegrained.png", dpi=160)
    plt.close(fig)

    # ---- Summary text: top collisions + decision flag ----
    THR = args.collision_threshold
    lines: list[str] = []
    lines.append(f"# Novel prompt cosine diagnostic (XLM-Roberta @ wedetect_tiny.pth)")
    lines.append(f"# threshold = {THR}")
    lines.append("")

    # 1. novel ↔ novel (off-diagonal)
    lines.append("## Novel ↔ Novel (cross-class collisions)")
    novel_pairs = []
    for i in range(n_novel):
        for j in range(i + 1, n_novel):
            novel_pairs.append((novel_cos[i, j], novel_labels[i], novel_labels[j]))
    novel_pairs.sort(reverse=True)
    for c, a, b in novel_pairs[:15]:
        flag = "⚠ COLLISION" if c >= THR else ""
        lines.append(f"  {c:.3f}  {a}  ↔  {b}  {flag}")
    lines.append("")

    # 2. novel ↔ base
    lines.append("## Novel ↔ Base (most-similar base class for each novel)")
    for ni in range(n_novel):
        best_bi = int(np.argmax(cos[n_base + ni, :n_base]))
        best_c = cos[n_base + ni, best_bi]
        flag = "⚠ COLLISION" if best_c >= THR else ""
        lines.append(
            f"  {best_c:.3f}  {novel_labels[ni]}  →  [base] {base_labels[best_bi]}  {flag}"
        )
    lines.append("")

    # 3. decision
    novel_max_offdiag = max(c for c, _, _ in novel_pairs) if novel_pairs else 0.0
    novel_to_base_max = float(cos[n_base:, :n_base].max())
    lines.append("## Decision flag")
    lines.append(f"  novel↔novel max off-diagonal cos  : {novel_max_offdiag:.3f}")
    lines.append(f"  novel↔base   max cos              : {novel_to_base_max:.3f}")
    if novel_max_offdiag > 0.90:
        lines.append("  → fine-grained novel pairs are >0.90: **text encoder is bottleneck**.")
        lines.append("    Recommended: TODO.md item 15 (BiomedCLIP / PubMedBERT swap).")
    elif novel_max_offdiag < 0.85 and novel_to_base_max < 0.90:
        lines.append("  → text geometry is fine: **bottleneck is image-text alignment**.")
        lines.append("    Recommended: TODO.md item 13 (visual exemplar prototype).")
    else:
        lines.append("  → mid-range collisions: **try item 13 (visual prototype) first**,")
        lines.append("    if it doesn't fix Resp/Thyroid novel, then item 15.")

    summary_path = out_dir / "novel_prompt_cos_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print()
    print(f"wrote {out_dir/'novel_prompt_cos_heatmap.png'}")
    print(f"wrote {out_dir/'novel_prompt_cos_finegrained.png'}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
