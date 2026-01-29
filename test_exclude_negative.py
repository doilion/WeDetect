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

# 负样本类别名称（内部索引从0开始）
NEGATIVE_CLASS_NAMES = [
    'normal',  # 内部索引 0
    'Serous effusion-Negative samples',  # 内部索引 8
    'Thyroid gland-Negative samples',  # 内部索引 12
    'Urine-Negative',  # 内部索引 14
    'respiratory tract-Negative samples',  # 内部索引 17
]

def main():
    parser = argparse.ArgumentParser(description='Test with excluded negative classes')
    parser.add_argument('--config', default='config/wedetect_tiny_tct.py', help='config file')
    parser.add_argument('--checkpoint', default='work_dirs/wedetect_tiny_tct/best_coco_bbox_mAP_epoch_9.pth',
                        help='checkpoint file')
    parser.add_argument('--exclude-negative', action='store_true', default=True,
                        help='exclude negative classes from evaluation')
    parser.add_argument('--work-dir', default='./work_dirs/test_exclude_negative',
                        help='work dir for evaluation outputs')
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Set checkpoint
    cfg.load_from = args.checkpoint

    # Modify evaluator to use ExcludeClassCocoMetric
    if args.exclude_negative:
        cfg.val_evaluator = dict(
            type='ExcludeClassCocoMetric',
            ann_file=cfg.data_root + 'annotations/test_base_v2.json',
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
