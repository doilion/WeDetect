#!/usr/bin/env python
"""Post-training diagnostic for Phase 5e SAVPE-v2.

Quantifies how aligned the trained `vis_emb` is to BiomedCLIP `text_emb` per class,
and how well-separated the 30 base class vis_emb are from each other.

Two paths:

  A) Visproto pth path (preferred, fast):
       Read a built visproto cache (dict primary_key → tensor).
       Compute cos(visproto[c], text_emb[c]) per class.
       Compute pairwise cos matrix between classes.
       No GPU needed.

  B) Live SAVPE forward path (slower, only if you need to inspect intermediate):
       Build N exemplars per class, forward through ConvNext + neck + SAVPE.

Outputs to stdout (and optionally JSON):
    - mean / median / per-class cos(vis, text)
    - top-5 most-aligned and least-aligned classes
    - 30×30 pairwise cos heatmap stats (off-diag mean / max)
    - verdict: COLLAPSED (>0.95) / GOOD (0.5-0.9) / UNDER-ALIGNED (<0.3) / MIXED

Usage (path A, recommended):
    PYTHONPATH=. python tools/diagnose_savpe_v2_alignment.py \\
        --visproto-pth data/texts/tct_ngc_base30_savpe_v2_aligned.pth \\
        --text-cache data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth \\
        --fullnames-json data/texts/tct_ngc_fullnames_30.json \\
        --out-json /tmp/savpe_v2_cos_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--visproto-pth", required=True,
                   help="Built visproto dict {primary_key → tensor[D]}")
    p.add_argument("--text-cache", required=True,
                   help="BiomedCLIP text emb cache, same key schema")
    p.add_argument("--fullnames-json", required=True,
                   help="data/texts/tct_ngc_fullnames_30.json (or _32.json etc)")
    p.add_argument("--out-json", default=None)
    p.add_argument("--verdict-collapse", type=float, default=0.95)
    p.add_argument("--verdict-good-lo", type=float, default=0.5)
    p.add_argument("--verdict-good-hi", type=float, default=0.9)
    p.add_argument("--verdict-under", type=float, default=0.3)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[load] visproto: {args.visproto_pth}")
    visproto: Dict[str, torch.Tensor] = torch.load(
        args.visproto_pth, map_location="cpu", weights_only=False
    )
    print(f"[load] text cache: {args.text_cache}")
    text_cache: Dict[str, torch.Tensor] = torch.load(
        args.text_cache, map_location="cpu", weights_only=False
    )
    print(f"[load] fullnames: {args.fullnames_json}")
    with open(args.fullnames_json) as f:
        fullnames: List[List[str]] = json.load(f)

    # Primary keys (first variant) — must match visproto cache schema
    primary_keys = [grp[0] for grp in fullnames]
    n_classes = len(primary_keys)
    print(f"[init] {n_classes} classes")

    # Filter to classes that have both vis and text emb
    missing_vis = [k for k in primary_keys if k not in visproto]
    missing_text = [k for k in primary_keys if k not in text_cache]
    if missing_vis:
        print(f"[warn] {len(missing_vis)} keys missing from visproto: "
              f"{missing_vis[:3]}{'...' if len(missing_vis)>3 else ''}")
    if missing_text:
        print(f"[warn] {len(missing_text)} keys missing from text cache: "
              f"{missing_text[:3]}{'...' if len(missing_text)>3 else ''}")

    valid_keys = [k for k in primary_keys if k in visproto and k in text_cache]
    print(f"[init] valid pairs: {len(valid_keys)}/{n_classes}")
    if len(valid_keys) == 0:
        raise SystemExit("no class with both vis and text emb — abort")

    # Stack into tensors, L2-norm
    vis_stack = torch.stack([visproto[k].float() for k in valid_keys], dim=0)
    text_stack = torch.stack([text_cache[k].float() for k in valid_keys], dim=0)
    vis_n = F.normalize(vis_stack, dim=-1, p=2)
    text_n = F.normalize(text_stack, dim=-1, p=2)
    print(f"[init] vis shape {tuple(vis_n.shape)}, text shape {tuple(text_n.shape)}")

    # ── 1. cos(vis_c, text_c) per class ─────────────────────────────────
    cos_aligned = (vis_n * text_n).sum(dim=-1)  # [C]
    print("\n=== cos(vis_emb, text_emb) per class ===")
    cos_sorted, idx = cos_aligned.sort(descending=True)
    for rank, i in enumerate(idx.tolist()):
        print(f"  [{rank+1:2d}] cos={cos_aligned[i].item():+.4f}  {valid_keys[i][:80]}")

    cos_mean = cos_aligned.mean().item()
    cos_median = cos_aligned.median().item()
    cos_min = cos_aligned.min().item()
    cos_max = cos_aligned.max().item()
    print(f"\n  mean={cos_mean:+.4f}  median={cos_median:+.4f}  "
          f"min={cos_min:+.4f}  max={cos_max:+.4f}")

    # ── 2. pairwise cos(vis_i, vis_j) — class separation ───────────────
    pair = vis_n @ vis_n.T  # [C, C]
    eye = torch.eye(pair.shape[0])
    off_diag = pair - eye * 2.0  # so self-cos doesn't enter max
    pair_off_mean = (pair - eye).sum() / (pair.numel() - pair.shape[0])
    pair_off_max = off_diag.max().item()
    pair_off_min = off_diag[off_diag > -1.5].min().item()  # exclude diag-masked

    print(f"\n=== pairwise cos(vis_i, vis_j) off-diagonal ===")
    print(f"  mean={pair_off_mean.item():+.4f}  max={pair_off_max:+.4f}  "
          f"min={pair_off_min:+.4f}")
    print(f"  (mean → 0 = well-separated;  mean → 1 = all vectors collapsed)")

    # Top 5 most-similar pairs (potential collapse pairs)
    triu = torch.triu(pair, diagonal=1)
    flat = triu.flatten()
    top_idx = flat.argsort(descending=True)[:5]
    print(f"\n  top 5 most-similar class pairs (potential collapse):")
    for k in top_idx.tolist():
        i = k // pair.shape[0]
        j = k % pair.shape[0]
        if i == j:
            continue
        print(f"    cos={pair[i, j].item():+.4f}  "
              f"{valid_keys[i][:50]}  ⟷  {valid_keys[j][:50]}")

    # ── 3. verdict ───────────────────────────────────────────────────────
    print("\n=== VERDICT ===")
    if cos_median > args.verdict_collapse:
        verdict = "COLLAPSED"
        msg = (f"cos median {cos_median:.3f} > {args.verdict_collapse}: "
               f"vis_emb has fully collapsed to text_emb. λ_align is too high. "
               f"Retrain with λ_align=0.3 or add visual reconstruction loss.")
    elif cos_median < args.verdict_under:
        verdict = "UNDER-ALIGNED"
        msg = (f"cos median {cos_median:.3f} < {args.verdict_under}: "
               f"L_align had little effect. Either L_cell dominated or λ_align too low. "
               f"Consider raising λ_align or check L_align curve.")
    elif args.verdict_good_lo <= cos_median <= args.verdict_good_hi:
        verdict = "GOOD"
        msg = (f"cos median {cos_median:.3f} ∈ [{args.verdict_good_lo}, "
               f"{args.verdict_good_hi}]: aligned but not collapsed. "
               f"Proceed to mAP eval.")
    else:
        verdict = "MIXED"
        msg = (f"cos median {cos_median:.3f}: between thresholds. "
               f"Proceed to mAP eval; if novel ≥ baseline this is fine, "
               f"otherwise tune λ_align.")
    print(f"  {verdict}: {msg}")

    if args.out_json is not None:
        report = dict(
            verdict=verdict,
            message=msg,
            cos_mean=cos_mean,
            cos_median=cos_median,
            cos_min=cos_min,
            cos_max=cos_max,
            cos_per_class={
                valid_keys[i]: cos_aligned[i].item() for i in range(len(valid_keys))
            },
            pair_off_mean=pair_off_mean.item(),
            pair_off_max=pair_off_max,
            pair_off_min=pair_off_min,
            n_classes=len(valid_keys),
            n_missing_vis=len(missing_vis),
            n_missing_text=len(missing_text),
        )
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[save] report: {args.out_json}")


if __name__ == "__main__":
    main()
