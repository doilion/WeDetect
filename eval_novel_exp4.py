#!/usr/bin/env python
"""Manual evaluation script for novel classes with proper label mapping."""

import sys
sys.path.insert(0, '/root/code/WeDetect')

import torch
import json
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.utils import register_all_modules
register_all_modules()

from mmengine.config import Config
from mmdet.apis import init_detector
from mmengine.dataset import Compose

# Novel 类在 31 类中的索引映射到 novel test set 的类别 ID
# 模型预测 label 20 -> novel test set category_id 1 (hsil_scc_omn)
# 模型预测 label 21 -> novel test set category_id 2 (monilia)
# ...
NOVEL_LABEL_TO_CAT_ID = {
    20: 1,   # hsil_scc -> hsil_scc_omn
    21: 2,   # monilia
    22: 3,   # Serous effusion-Ovarian cancer
    23: 4,   # Serous effusion-adenocarcinoma
    24: 5,   # Thyroid gland-Suspicious papillary cancer
    25: 6,   # Thyroid gland-AUC
    26: 7,   # Thyroid gland-Malignant tumour
    27: 8,   # Thyroid gland-NS
    28: 9,   # Urine-HGUC
    29: 10,  # respiratory tract-Squamous cell cinoma
    30: 11,  # respiratory tract-Small cell carcinoma
}

def main():
    # 加载配置和模型
    cfg = Config.fromfile('config/wedetect_tiny_tct_exp4.py')
    checkpoint = 'work_dirs/wedetect_tiny_tct_exp4/best_coco_bbox_mAP_epoch_7.pth'

    print("Loading model...")
    model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')

    # 加载 31 类文本用于推理
    with open('data/texts/tct_ngc_v2_class_texts.json', 'r') as f:
        texts = json.load(f)
    texts = [[t[0]] for t in texts] + [[' ']]  # 添加背景类

    # Reparameterize 模型使用完整 31 类
    print("Reparameterizing model with 31 classes...")
    model.reparameterize(texts)

    # 加载 novel 测试集
    ann_file = '/home1/liwenjie/TCT_NGC/annotations/test_novel_v2.json'
    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()

    print(f"Total novel test images: {len(img_ids)}")
    print(f"Novel categories: {coco_gt.cats}")

    # 构建测试 pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    # 收集预测结果
    results = []

    print("Running inference...")
    for img_id in tqdm(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = f"/home1/liwenjie/TCT_NGC/{img_info['file_name']}"

        # 准备数据
        data_info = dict(img_id=img_id, img_path=img_path, texts=texts)
        data_info = test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info['inputs'].unsqueeze(0).cuda(),
            data_samples=[data_info['data_samples']]
        )

        # 推理
        with torch.no_grad():
            output = model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        # 过滤低置信度预测
        scores = pred_instances.scores.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()

        # 只保留 novel 类的预测 (labels 20-30)
        for i in range(len(labels)):
            label = int(labels[i])
            if label in NOVEL_LABEL_TO_CAT_ID:
                # 转换 bbox 格式: xyxy -> xywh
                x1, y1, x2, y2 = bboxes[i]
                w, h = x2 - x1, y2 - y1

                results.append({
                    'image_id': img_id,
                    'category_id': NOVEL_LABEL_TO_CAT_ID[label],
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(scores[i])
                })

    print(f"Total predictions for novel classes: {len(results)}")

    # 保存预测结果
    pred_file = 'work_dirs/test_novel_exp4/novel_predictions.json'
    import os
    os.makedirs('work_dirs/test_novel_exp4', exist_ok=True)
    with open(pred_file, 'w') as f:
        json.dump(results, f)

    # COCO 评估
    print("\nRunning COCO evaluation...")
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 打印每个类别的 AP
    print("\n" + "=" * 60)
    print("Per-class AP for Novel Classes:")
    print("=" * 60)

    precisions = coco_eval.eval['precision']
    cat_ids = coco_gt.getCatIds()

    for idx, cat_id in enumerate(cat_ids):
        cat_name = coco_gt.cats[cat_id]['name']
        # precision shape: [T, R, K, A, M] - IoU thresholds, recall thresholds, categories, areas, max dets
        precision = precisions[:, :, idx, 0, -1]  # all IoU, all recall, this category, all areas, max dets
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else 0.0

        # AP50
        precision_50 = precisions[0, :, idx, 0, -1]
        precision_50 = precision_50[precision_50 > -1]
        ap_50 = np.mean(precision_50) if precision_50.size else 0.0

        # 获取该类的标注数量
        ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        num_anns = len(ann_ids)

        print(f"{cat_name:45s}  AP: {ap:.4f}  AP50: {ap_50:.4f}  (samples: {num_anns})")

    print("=" * 60)

if __name__ == '__main__':
    main()
