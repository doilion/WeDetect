#!/usr/bin/env python
"""Test script to evaluate model excluding negative classes."""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mmdet.utils import register_all_modules
register_all_modules()

from mmengine.config import Config
from mmengine.runner import Runner

from wedetect.utils import resolve_latest_checkpoint

# 负样本类别名称（内部索引从0开始）
NEGATIVE_CLASS_NAMES = [
    'respiratory tract-Impurity',
    'Serous effusion-Negative samples',
    'Thyroid gland-Negative samples',
    'Urine-NILM',
    'Urine-Negative',
    'Urine-Negative Degeneration',
    'TCT_CCD-normal',
]


def resolve_ann_file(data_root: str, ann_file: str) -> str:
    ann_path = Path(ann_file)
    if ann_path.is_absolute():
        return str(ann_path)
    return str(Path(data_root) / ann_path)


def refresh_evaluator_ann_file(cfg):
    ann_file = cfg.val_dataloader.dataset.dataset.ann_file
    cfg.val_evaluator.ann_file = resolve_ann_file(cfg.data_root, ann_file)
    cfg.test_evaluator = cfg.val_evaluator


def main():
    parser = argparse.ArgumentParser(description='Test with excluded negative classes')
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
    parser.add_argument('--exclude-negative', action='store_true', default=True,
                        help='exclude negative classes from evaluation')
    parser.add_argument('--work-dir', default='./work_dirs/test_exclude_negative',
                        help='work dir for evaluation outputs')
    parser.add_argument('--data-root', default=None,
                        help='override cfg.data_root (e.g. /home1/liwenjie/TCT_NGC/ '
                             'when running held-out test on the non-cache640 host)')
    parser.add_argument('--ann-file', default=None,
                        help='override the test ann_file relative to --data-root '
                             '(e.g. annotations/instances_test_base_clean.json)')
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Set checkpoint
    cfg.load_from = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)

    if args.data_root:
        cfg.data_root = args.data_root
        cfg.test_dataloader.dataset.dataset.data_root = args.data_root
        cfg.val_dataloader.dataset.dataset.data_root = args.data_root
        refresh_evaluator_ann_file(cfg)
    if args.ann_file:
        cfg.test_dataloader.dataset.dataset.ann_file = args.ann_file
        cfg.val_dataloader.dataset.dataset.ann_file = args.ann_file
        refresh_evaluator_ann_file(cfg)

    # Modify evaluator to use ExcludeClassCocoMetric
    if args.exclude_negative:
        cfg.val_evaluator = dict(
            type='ExcludeClassCocoMetric',
            ann_file=cfg.val_evaluator.ann_file,
            metric='bbox',
            exclude_class_id=NEGATIVE_CLASS_NAMES,  # 排除负样本类别
            classwise=True,
        )
        cfg.test_evaluator = cfg.val_evaluator

    # Set work dir for this test
    cfg.work_dir = args.work_dir

    # Build runner
    runner = Runner.from_cfg(cfg)

    # Run test
    runner.test()

if __name__ == '__main__':
    main()
