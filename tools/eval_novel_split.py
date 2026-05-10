#!/usr/bin/env python
"""Zero-shot evaluate a checkpoint on a TCT_NGC novel split.

Each split JSON has 2-5 novel categories (cat_ids 21-30) that the dev32 base
model has never seen. We swap PseudoLanguageBackbone's cached embeddings to the
novel prompts and override the test dataloader/evaluator to point at the novel
ann file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    p.add_argument("--text-json", required=True)
    p.add_argument("--text-emb", required=True)
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


def main() -> None:
    args = parse()

    abs_ann = resolve_ann_file(args.data_root, args.ann_file)
    classes, n_classes = derive_metainfo(abs_ann)

    with open(args.text_json, "r") as f:
        n_prompts = len(json.load(f))
    if n_prompts != n_classes:
        raise SystemExit(
            f"prompt count {n_prompts} != novel ann categories {n_classes}; "
            f"build a JSON with one prompt per category in cat_id order."
        )

    cfg = Config.fromfile(args.config)
    cfg.load_from = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)

    # 0) Refuse to run on THAF configs — replacing the text_model with a
    # PseudoLanguageBackbone (step 1 below) would silently discard the
    # trained cross-attention fusion module weights and produce wrong
    # numbers. THAF eval has its own tool that preserves the fusion module.
    text_model_type = cfg.model.backbone.text_model.get("type", "")
    if "Hierarchical" in text_model_type:
        raise SystemExit(
            f"text_model type {text_model_type!r} is a THAF backbone with a "
            f"trained fusion module. Replacing it with PseudoLanguageBackbone "
            f"would discard those weights and yield bogus mAP. Use "
            f"tools/eval_novel_thaf.py instead (no --text-emb arg; the "
            f"trained backbone reads its own attr_emb_cache_path)."
        )

    # 1) Replace text_model with PseudoLanguageBackbone backed by the novel
    # cached embeddings. We do a full replacement (rather than only overriding
    # text_embed_path) because the training config may specify a different
    # backbone family (e.g. PseudoHierarchicalBiomedCLIPLanguageBackbone for
    # THAF) whose constructor doesn't accept `text_embed_path`. The eval
    # cache stores per-class fused vectors keyed by class_name (or by prompt
    # string), which PseudoLanguageBackbone consumes natively.
    cfg.model.backbone.text_model = dict(
        type="PseudoLanguageBackbone",
        text_embed_path=args.text_emb,
    )

    # 2) Resize test-time class count
    cfg.model.num_test_classes = n_classes
    if "num_classes" in cfg.model.bbox_head.head_module:
        # head_module.num_classes is set to num_training_classes at build time;
        # YOLO-World head uses text_feats[0].shape[0] for actual class count at test,
        # so this override is cosmetic but kept for consistency.
        pass

    # 3) Override dataloader for the novel split
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

    # 4) Override evaluator
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
