#!/usr/bin/env python
"""Generate the post-training class-embedding cache for THAF eval.

After training, the fusion module has learned to produce per-class vectors
from the 5 attribute strings. We bake those vectors into a frozen cache
keyed by class name so the existing eval pipeline (PseudoLanguageBackbone +
eval_novel_split.py) can consume them without instantiating the trainable
fusion module.

Output:
    data/texts/tct_ngc_thaf_<encoder>_dev30_emb.pth
        dict[class_name -> Tensor[D]]

The cache is consumed by eval_novel_split.py via the existing
`--text-emb <path>` flag — PseudoLanguageBackbone.forward_text looks up
`text.split("/")[0]` against the dict, and class_name has no "/" so the
lookup matches.

Usage:
    python tools/build_hier_class_embeddings.py \
        --config config/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu.py \
        --checkpoint work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu/best_*.pth \
        --attr-json data/texts/tct_ngc_fullnames_39_attr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


ATTR_FIELDS_ORDERED = (
    "organ_specimen",
    "diagnostic_code",
    "cytomorphology",
    "background_and_immunoprofile",
    "key_distinguishing_feature",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--attr-json",
        default="data/texts/tct_ngc_fullnames_39_attr.json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output .pth path; default derived from config name",
    )
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def derive_default_out(config_path: str) -> str:
    """thaf_<encoder>_2gpu.py → tct_ngc_thaf_<encoder>_dev30_emb.pth"""
    name = Path(config_path).stem
    if "thaf_xlmr" in name:
        return "data/texts/tct_ngc_thaf_xlmr_dev30_emb.pth"
    if "thaf_biomedclip" in name:
        return "data/texts/tct_ngc_thaf_biomedclip_dev30_emb.pth"
    raise SystemExit(
        f"unknown config naming {name!r}; pass --out explicitly"
    )


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = derive_default_out(args.config)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = "/tmp/build_hier_class_embeddings"
    # Suppress dataloader build by minimizing it (we don't run train/val)
    runner = Runner.from_cfg(cfg)
    text_model = runner.model.backbone.text_model
    text_model.to(args.device).eval()

    # Load trained weights into the model. Allow strict=False because eval
    # cache build only needs text_model.* params and a few image-side
    # buffers; the rest can be random.
    print(f"[load] {args.checkpoint}")
    load_checkpoint(runner.model, args.checkpoint, map_location="cpu", strict=False)

    # Read 39 classes × 5 attrs in canonical order
    attr = json.loads(Path(args.attr_json).read_text(encoding="utf-8"))
    if len(attr) != 39:
        raise SystemExit(f"expected 39 entries, got {len(attr)}")

    class_names = list(attr.keys())
    nested_text = [
        [
            [attr[name][f].strip() for f in ATTR_FIELDS_ORDERED]
            for name in class_names
        ]
    ]  # shape: [batch=1][num_classes=39][num_attrs=5]
    print(f"[forward] {len(class_names)} classes × {len(ATTR_FIELDS_ORDERED)} attrs")

    with torch.no_grad():
        out = text_model(nested_text)  # [1, 39, D]
    if out.dim() != 3 or out.shape[0] != 1 or out.shape[1] != len(class_names):
        raise RuntimeError(f"unexpected fusion output shape {tuple(out.shape)}")

    embed_dim = out.shape[-1]
    cache = {
        name: out[0, i].detach().cpu().float().contiguous()
        for i, name in enumerate(class_names)
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"[save] {out_path}  ({len(cache)} class vectors, dim={embed_dim})")

    # Quick sanity: pairwise cos
    vecs = torch.stack(list(cache.values()))
    cos = vecs @ vecs.T
    cos.fill_diagonal_(0)
    print(f"[sanity] class-class cos: mean={cos.abs().mean():.4f}  max={cos.max():.4f}")


if __name__ == "__main__":
    main()
