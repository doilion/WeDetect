#!/usr/bin/env python
"""Test script to evaluate model on novel classes."""

import argparse
import sys
sys.path.insert(0, '/root/code/WeDetect')

from mmdet.utils import register_all_modules
register_all_modules()

from mmengine.config import Config
from mmengine.runner import Runner

# Novel 类在 31 类文本中的索引 (0-based, 即第 21-31 个类)
# 对应 all_classes 中索引 20-30
NOVEL_CLASS_INDICES = list(range(20, 31))  # [20, 21, 22, ..., 30]

def main():
    parser = argparse.ArgumentParser(description='Test on novel classes')
    parser.add_argument('--config', default='config/wedetect_tiny_tct.py', help='config file')
    parser.add_argument('--checkpoint', default='work_dirs/wedetect_tiny_tct/best_coco_bbox_mAP_epoch_9.pth',
                        help='checkpoint file')
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint

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
            class_text_path='data/texts/tct_ngc_v2_class_texts.json',  # 使用完整 31 类文本
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
