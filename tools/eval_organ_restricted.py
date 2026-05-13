#!/usr/bin/env python
"""Organ-conditional inference with per-organ AP breakdown.

This is the eval entry for the new clinical protocol:
  - At inference time, organ_id is derived from the image path (no GT
    leakage) and used to mask cross-organ class scores BEFORE NMS in
    YOLOWorldHead.predict_by_feat.
  - Per-organ AP, overall macro, all-class flat, and instance-weighted
    mAP are reported by OrganRestrictedCocoMetric.

Two intended use cases:

  (a) Base 25-class eval (row 2 of the ablation table):
      bash:
        PYTHONPATH=. python tools/eval_organ_restricted.py \\
          --config config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py \\
          --checkpoint work_dirs/.../best_*.pth \\
          --data-root /home1/liwenjie/TCT_NGC/ \\
          --ann-file annotations/instances_test_base_clean_dev30.json \\
          --mask-file data/texts/tct_ngc_class_organ_mask_base30.pt \\
          --exclude-class-names \\
            'respiratory tract-Impurity,Serous effusion-Negative samples,Thyroid gland-Negative samples,Urine-NHGUC,TCT_CCD-normal' \\
          --work-dir work_dirs/.../organ_restricted_base25

  (b) Merged 9-class novel eval:
      bash:
        PYTHONPATH=. python tools/eval_organ_restricted.py \\
          --config config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py \\
          --checkpoint work_dirs/.../best_*.pth \\
          --data-root /home1/liwenjie/TCT_NGC/ \\
          --ann-file annotations/instances_test_novel_merged_9.json \\
          --text-json data/texts/tct_ngc_novel_merged_9.json \\
          --text-emb data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth \\
          --mask-file data/texts/tct_ngc_class_organ_mask_novel_merged.pt \\
          --work-dir work_dirs/.../organ_restricted_novel9
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
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--data-root', default='/home1/liwenjie/TCT_NGC/')
    p.add_argument('--ann-file', required=True,
                   help='relative to --data-root')
    p.add_argument('--text-json', default=None,
                   help='novel class prompts JSON; omit for base eval (uses cfg default)')
    p.add_argument('--text-emb', default=None,
                   help='novel class emb cache .pth; omit for base eval (uses cfg default)')
    p.add_argument('--mask-file', required=True,
                   help='class×organ mask .pt produced by tools/build_class_organ_mask.py')
    p.add_argument('--taxonomy', default='data/texts/tct_ngc_taxonomy.json',
                   help='for OrganExtractor pipeline transform')
    p.add_argument('--work-dir', required=True)
    p.add_argument('--outfile-prefix', default=None,
                   help='dump <prefix>.bbox.json so we can compare against legacy preds')
    p.add_argument('--exclude-class-names', default=None,
                   help='comma-sep names to drop from COCOeval catIds '
                        '(use to keep base 25-class protocol parity with noTHAF 0.321)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--no-mask', action='store_true',
                   help='skip head.organ_class_mask attachment (eval same ann but '
                        'with 30/9 unrestricted class scoring) — produces row 1 '
                        'numbers under the same per-organ AP breakdown.')
    return p.parse_args()


def resolve_abs(data_root: str, ann_file: str) -> Path:
    p = Path(ann_file)
    return p if p.is_absolute() else Path(data_root) / p


def derive_metainfo(abs_ann: Path) -> tuple[tuple[str, ...], int]:
    data = json.loads(abs_ann.read_text())
    cats = sorted(data['categories'], key=lambda c: c['id'])
    return tuple(c['name'] for c in cats), len(cats)


def patch_pipeline_with_organ_extractor(pipeline: list, taxonomy_path: str) -> list:
    """Insert OrganExtractor before PackDetInputs and extend meta_keys."""
    patched = []
    inserted = False
    for step in pipeline:
        if step.get('type') == 'PackDetInputs':
            if not inserted:
                patched.append(dict(
                    type='OrganExtractor',
                    taxonomy_path=taxonomy_path,
                    strict=True,
                ))
                inserted = True
            new_step = dict(step)
            mk = tuple(new_step.get('meta_keys', ()))
            for k in ('organ_id', 'organ_name'):
                if k not in mk:
                    mk = mk + (k,)
            new_step['meta_keys'] = mk
            patched.append(new_step)
        else:
            patched.append(step)
    if not inserted:
        raise SystemExit('test_pipeline has no PackDetInputs step — cannot inject OrganExtractor')
    return patched


def main():
    args = parse()
    abs_ann = resolve_abs(args.data_root, args.ann_file)
    classes, n_classes = derive_metainfo(abs_ann)
    metainfo = dict(classes=classes)

    cfg = Config.fromfile(args.config)
    cfg.load_from = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)

    # Guard: THAF backbones would lose their fusion module if we swap text_model.
    text_model_type = cfg.model.backbone.text_model.get('type', '')
    if 'Hierarchical' in text_model_type and args.text_emb:
        raise SystemExit(
            f'text_model {text_model_type!r} is a THAF backbone; swapping to '
            'PseudoLanguageBackbone with --text-emb would discard trained '
            'fusion weights. For organ-restricted THAF eval, write a sibling '
            'entry that preserves the THAF backbone (cf. eval_novel_thaf.py).')

    # 1) Swap text_model emb cache (novel path) — base path: keep config as-is.
    if args.text_emb is not None:
        cfg.model.backbone.text_model = dict(
            type='PseudoLanguageBackbone',
            text_embed_path=args.text_emb,
        )
        cfg.model.num_test_classes = n_classes

    # 2) Inject OrganExtractor + extend meta_keys
    cfg.test_pipeline = patch_pipeline_with_organ_extractor(
        cfg.test_pipeline, args.taxonomy)

    # 3) Override test_dataloader (and mirror to val_dataloader, since some
    # config inheritance chains set `test_dataloader = val_dataloader` and a
    # later override of only one side leaves the other stale).
    cfg.test_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='MultiModalDataset',
            dataset=dict(
                type='WeCocoDataset',
                data_root=args.data_root,
                test_mode=True,
                ann_file=args.ann_file,
                data_prefix=dict(img='images/'),
                batch_shapes_cfg=None,
                metainfo=metainfo,
            ),
            class_text_path=(
                args.text_json
                if args.text_json is not None
                else cfg.test_dataloader.dataset.class_text_path
            ),
            pipeline=cfg.test_pipeline,
        ),
    )
    cfg.val_dataloader = cfg.test_dataloader

    # 4) Override evaluator (drop default), use OrganRestrictedCocoMetric.
    eval_kwargs = dict(
        type='OrganRestrictedCocoMetric',
        ann_file=str(abs_ann),
        metric='bbox',
        classwise=True,
        organ_mask_path=args.mask_file,
    )
    if args.outfile_prefix:
        eval_kwargs['outfile_prefix'] = args.outfile_prefix
    cfg.test_evaluator = eval_kwargs

    if args.exclude_class_names:
        cfg.test_evaluator['exclude_class_names'] = [
            n.strip() for n in args.exclude_class_names.split(',') if n.strip()
        ]

    cfg.work_dir = args.work_dir

    runner = Runner.from_cfg(cfg)

    # 5) Attach organ_class_mask to bbox_head so predict_by_feat applies it.
    mask_pkg = torch.load(args.mask_file, weights_only=False)
    mask_tensor = mask_pkg['mask']                                # [C, O]
    if mask_tensor.shape[0] != n_classes:
        raise SystemExit(
            f'mask file has {mask_tensor.shape[0]} classes but ann has {n_classes}. '
            f'Regenerate mask with: '
            f'python tools/build_class_organ_mask.py --ann {abs_ann} ...')
    if args.no_mask:
        runner.model.bbox_head.set_organ_class_mask(None)
        print(f'[eval_organ_restricted] --no-mask: cleared any config-loaded '
              f'mask (row 1 unrestricted protocol)')
    else:
        # Also cross-check class_ids alignment up-front (P2 #8) — earlier than
        # the metric's own check inside compute_metrics.
        mask_class_ids = list(mask_pkg['class_ids'])
        ann_data = json.loads(abs_ann.read_text())
        ann_class_ids = [c['id']
                         for c in sorted(ann_data['categories'], key=lambda c: c['id'])]
        if mask_class_ids != ann_class_ids:
            raise SystemExit(
                f'mask file class_ids {mask_class_ids[:5]}... do not match ann '
                f'category ids {ann_class_ids[:5]}... — regenerate the mask '
                f'against this ann (tools/build_class_organ_mask.py --ann {abs_ann}).')
        runner.model.bbox_head.set_organ_class_mask(mask_tensor)
        print(f'[eval_organ_restricted] attached mask {mask_tensor.shape} to bbox_head')
    print(f'[eval_organ_restricted] ann: {abs_ann}')
    print(f'[eval_organ_restricted] classes ({n_classes}): {classes[:3]}...')

    runner.test()


if __name__ == '__main__':
    main()
