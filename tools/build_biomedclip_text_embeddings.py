#!/usr/bin/env python
"""Build a BiomedCLIP-encoded text-embedding cache for PseudoLanguageBackbone.

Usage:
    python tools/build_biomedclip_text_embeddings.py \\
        --texts data/texts/tct_ngc_fullnames_30.json \\
        --out   data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth

Input JSON format: list-of-list, inner list = variants for one class
(WeDetect's standard class-text format). Each leaf string is encoded by
BiomedCLIP-PubMedBERT and stored at the leaf's string key.

For multi-variant classes (inner list len > 1), we average the per-variant
embeddings (CLIP-style ensembling) and store under the FIRST variant's
string — matches `tools/build_text_embeddings.py` behaviour.

Output: dict[class_string -> Tensor(512)], drop-in for
`PseudoLanguageBackbone(text_embed_path=...)`.

Why not extend `tools/build_text_embeddings.py`: that script is hardwired
to load WeDetect ckpt weights into XLMRobertaLanguageBackbone. BiomedCLIP
loads from HuggingFace (`open_clip.create_model_from_pretrained`) and has
no WeDetect ckpt to merge with — different code path is cleaner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--texts",
        required=True,
        help="Class text JSON file (list-of-lists; inner list = variants for one class).",
    )
    p.add_argument("--out", required=True, help="Output .pth file (dict[str, Tensor]).")
    p.add_argument(
        "--biomedclip-name",
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        help="open_clip model identifier for BiomedCLIP.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--per-variant-l2-norm",
        action="store_true",
        help="L2-normalize each variant before within-class averaging (standard "
        "CLIP zero-shot recipe). Only meaningful when inner lists have >1 variant. "
        "Default off; final per-class embedding is always L2-normalized.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import open_clip

    texts_path = Path(args.texts)
    classes: list[list[str]] = json.loads(texts_path.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not all(isinstance(c, list) and c for c in classes):
        raise SystemExit(
            f"{texts_path} must be a non-empty list-of-non-empty-lists; got "
            f"top-level type {type(classes).__name__}"
        )

    flat: list[str] = []
    offsets: list[tuple[int, int]] = []  # [(start, end_exclusive), ...] per class
    for variants in classes:
        start = len(flat)
        flat.extend(variants)
        offsets.append((start, len(flat)))
    print(
        f"[encode] {texts_path} has {len(classes)} classes, "
        f"{len(flat)} total variant strings"
    )

    print(f"[encode] loading BiomedCLIP from {args.biomedclip_name}")
    model, _ = open_clip.create_model_from_pretrained(args.biomedclip_name)
    tokenizer = open_clip.get_tokenizer(args.biomedclip_name)
    model.to(args.device).eval()

    tokens = tokenizer(flat).to(args.device)
    with torch.no_grad():
        embs = model.encode_text(tokens)  # [N_flat, 512]
    if args.per_variant_l2_norm:
        embs = embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    embs = embs.detach().cpu().float()

    if embs.shape[0] != len(flat) or embs.shape[1] != 512:
        raise RuntimeError(
            f"unexpected embedding shape {tuple(embs.shape)}; "
            f"expected ({len(flat)}, 512)"
        )

    cache: dict[str, torch.Tensor] = {}
    for variants, (s, e) in zip(classes, offsets):
        avg = embs[s:e].mean(dim=0)
        avg = avg / avg.norm(dim=-1).clamp(min=1e-9)  # final L2 normalize
        cache[variants[0]] = avg.contiguous()

    if len(cache) != len(classes):
        raise RuntimeError(
            f"duplicate first-variant strings detected: {len(cache)} unique "
            f"keys for {len(classes)} classes. Disambiguate input JSON."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"[save] {out_path}  ({len(cache)} entries, dim=512)")


if __name__ == "__main__":
    main()
