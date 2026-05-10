#!/usr/bin/env python
"""Zero-shot eval for THAF-trained checkpoints (XLM-R or BiomedCLIP backbone).

Differs from `tools/eval_novel_split.py`:
- Does NOT replace the model.backbone.text_model. The trained
  `PseudoHierarchical{XLMR,BiomedCLIP}LanguageBackbone` carries the
  cross-attention fusion weights we need at eval time. Replacing it with
  PseudoLanguageBackbone (the v1 path) would discard the fusion module.
- Eval JSON must be 5-attr list-of-list: `[[organ, diag, morph, bg, distinguish], ...]`
  one entry per class in cat_id order (matching the test ann file).
- The per-attr cache (`attr_emb_cache_path` baked into the THAF config) is
  shared across base + all novel splits; it indexes attr strings by exact
  match, so all attr strings in the eval JSON must already be in the cache.
  We verify this up front and fail loudly if not.

Usage:
    python tools/eval_novel_thaf.py \
        --config config/wedetect_tiny_tct_ngc_dev30_thaf_<ENC>_2gpu.py \
        --checkpoint work_dirs/.../best_coco_bbox_mAP_epoch_*.pth \
        --data-root /home1/liwenjie/TCT_NGC/ \
        --ann-file annotations/instances_test_main_novel.json \
        --text-json data/texts/tct_ngc_attr_main_3_eval.json \
        --work-dir work_dirs/.../eval_novel_main_3_thaf
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

from wedetect.utils import resolve_latest_checkpoint


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data-root", default="/home1/liwenjie/TCT_NGC/")
    p.add_argument("--ann-file", required=True,
                   help="relative to --data-root, e.g. annotations/instances_test_main_novel.json")
    p.add_argument("--text-json", required=True,
                   help="5-attr eval JSON (list-of-list-of-5-strings) in cat_id order")
    p.add_argument("--work-dir", required=True)
    p.add_argument(
        "--outfile-prefix",
        default=None,
        help="persist predictions to <prefix>.bbox.json for later score fusion",
    )
    return p.parse_args()


def resolve_ann_file(data_root: str, ann_file: str) -> Path:
    ann_path = Path(ann_file)
    if ann_path.is_absolute():
        return ann_path
    return Path(data_root) / ann_path


def derive_metainfo(ann_path: Path) -> tuple[tuple[str, ...], int]:
    with open(ann_path, "r") as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda c: c["id"])
    return tuple(c["name"] for c in cats), len(cats)


def verify_cache_covers_eval(
    cache_path: str, text_json_path: str, num_attr_types: int
) -> None:
    cache = torch.load(cache_path, map_location="cpu")
    if not isinstance(cache, dict):
        raise SystemExit(f"cache must be dict, got {type(cache).__name__}")
    keys = set(cache.keys())
    eval_data = json.loads(Path(text_json_path).read_text(encoding="utf-8"))
    missing = []
    for i, cls_attrs in enumerate(eval_data):
        if not isinstance(cls_attrs, list) or len(cls_attrs) != num_attr_types:
            raise SystemExit(
                f"eval JSON entry {i} expected {num_attr_types} attrs, "
                f"got {len(cls_attrs) if isinstance(cls_attrs, list) else 'non-list'}"
            )
        for s in cls_attrs:
            if s not in keys:
                missing.append(s[:80])
    if missing:
        raise SystemExit(
            f"{len(missing)} attr string(s) not in cache {cache_path!r}. "
            f"Sample: {missing[:3]}. Rebuild the cache via "
            f"tools/build_per_attr_emb_cache.py with the same encoder."
        )


def main() -> None:
    args = parse()

    abs_ann = resolve_ann_file(args.data_root, args.ann_file)
    classes, n_classes = derive_metainfo(abs_ann)

    eval_data = json.loads(Path(args.text_json).read_text(encoding="utf-8"))
    if len(eval_data) != n_classes:
        raise SystemExit(
            f"text-json class count {len(eval_data)} != ann categories {n_classes}; "
            f"build a 5-attr JSON in cat_id order."
        )

    cfg = Config.fromfile(args.config)
    cfg.load_from = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)

    # Verify the trained backbone's per-attr cache covers the eval attrs
    text_model_cfg = cfg.model.backbone.text_model
    if "attr_emb_cache_path" not in text_model_cfg:
        raise SystemExit(
            f"config text_model {text_model_cfg.get('type')!r} does not use "
            f"attr_emb_cache_path; this tool only handles THAF backbones. "
            f"For PseudoLanguageBackbone use eval_novel_split.py instead."
        )
    num_attr_types = text_model_cfg.get("num_attr_types", 5)
    verify_cache_covers_eval(
        text_model_cfg["attr_emb_cache_path"], args.text_json, num_attr_types
    )

    # Resize test-time class count
    cfg.model.num_test_classes = n_classes

    # Override dataloader for the (novel or base) split
    novel_metainfo = dict(classes=classes)
    cfg.test_dataloader = dict(
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type="MultiModalDataset",
            dataset=dict(
                type="WeCocoDataset",
                data_root=args.data_root,
                test_mode=True,
                ann_file=args.ann_file,
                data_prefix=dict(img="images/"),
                batch_shapes_cfg=None,
                metainfo=novel_metainfo,
            ),
            class_text_path=args.text_json,
            pipeline=cfg.test_pipeline,
        ),
    )

    cfg.test_evaluator = dict(
        type="CocoMetric",
        ann_file=str(abs_ann),
        metric="bbox",
        classwise=True,
    )
    if args.outfile_prefix:
        cfg.test_evaluator["outfile_prefix"] = args.outfile_prefix

    cfg.work_dir = args.work_dir

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
