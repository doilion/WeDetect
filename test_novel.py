#!/usr/bin/env python
"""Test script to evaluate model on novel classes."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from mmdet.utils import register_all_modules
register_all_modules()

from mmengine.config import Config
from mmengine.runner import Runner

from wedetect.utils import resolve_latest_checkpoint


def assert_text_count_matches(class_text_path: str, expected: int) -> None:
    """Loud failure when the prompt JSON does not 1:1 match the dataset metainfo.

    Without this guard, MMEngine silently feeds N prompts into an M-class evaluator
    (M != N), and CocoMetric scores garbage AP. Hit by the dev32 cache640 default,
    where ``test_class_text_path`` ships 32 fullnames but the novel split has 11
    classes — see audit P1."""
    with open(class_text_path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    if len(groups) != expected:
        raise SystemExit(
            f"prompt count mismatch: {class_text_path} has {len(groups)} entries "
            f"but novel_metainfo declares {expected} classes. "
            f"Pass --text path/to/tct_ngc_novel_{expected}_texts.json with the right count."
        )


def main():
    parser = argparse.ArgumentParser(description='Test on novel classes')
    parser.add_argument(
        '--config',
        default='config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu.py',
        help='config file',
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='checkpoint file; defaults to latest best checkpoint in config work_dir',
    )
    parser.add_argument('--text', default=None,
                        help='class text path (overrides default)')
    parser.add_argument(
        '--data-root',
        default=None,
        help=(
            'override cfg.data_root for the novel split. '
            'Required when the config points at the cache640 dataset, '
            'which only ships train_dev/val_dev (not test_novel_v2.json).'
        ),
    )
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)
    if args.data_root:
        cfg.data_root = args.data_root

    # 定义 novel 类的完整元信息 (与 test_novel_v2.json 中的类别对应)
    novel_metainfo = dict(
        classes=(
            'hsil_scc_omn',
            'monilia',
            'Serous effusion-Ovarian cancer',
            'Serous effusion-adenocarcinoma',
            'Thyroid gland-Suspicious papillary cancer',
            'Thyroid gland-AUC',
            'Thyroid gland-Malignant tumour',
            'Thyroid gland-NS',
            'Urine-HGUC',
            'respiratory tract-Squamous cell cinoma',
            'respiratory tract-Small cell carcinoma',
        )
    )

    class_text_path = (
        args.text
        if args.text
        else cfg.get('test_class_text_path', 'data/texts/tct_ngc_fullnames_32.json')
    )
    assert_text_count_matches(class_text_path, len(novel_metainfo['classes']))

    # 配置 novel 测试数据加载器
    cfg.test_dataloader = dict(
        batch_size=1,  # 使用小 batch size 避免问题
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='MultiModalDataset',
            dataset=dict(
                type='WeCocoDataset',
                data_root=cfg.data_root,
                ann_file='annotations/test_novel_v2.json',
                data_prefix=dict(img=''),
                test_mode=True,
                batch_shapes_cfg=None,
                metainfo=novel_metainfo,
            ),
            class_text_path=class_text_path,
            pipeline=cfg.test_pipeline,
        )
    )

    # 配置评估器
    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=cfg.data_root + 'annotations/test_novel_v2.json',
        metric='bbox',
        classwise=True,
    )

    cfg.work_dir = './work_dirs/test_novel'

    # Build and run
    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    main()
