# 实验总结: 基准实验 (Baseline)

**生成时间**: 2025-12-31 (训练完成时间)

**Checkpoint**: `work_dirs/wedetect_tiny_tct/best_coco_bbox_mAP_epoch_9.pth`

---

## 配置

```python
base_lr = 2e-4
max_epochs = 15
weight_decay = 0.05
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
frozen_modules = []  # 不冻结
```

---

## 1. 整体性能

| 评估集 | mAP | mAP_50 | mAP_75 | 最佳Epoch |
|--------|-----|--------|--------|-----------|
| Base (全部20类) | 0.254 | 0.390 | 0.287 | 9 |
| Base (排除负样本, 15类) | 0.321 | 0.487 | 0.364 | - |
| Novel (零样本, 11类) | 0.068 | 0.092 | 0.075 | - |

---

## 2. Base 类详细结果 (排除负样本)

| 类别 | mAP | mAP_50 | 表现 |
|------|-----|--------|------|
| Thyroid gland-Suspicious for Malignancy | 0.621 | 0.902 | 极好 |
| Serous effusion-Breast cancer | 0.557 | 0.742 | 极好 |
| dysbacteriosis_herpes_act | 0.538 | 0.764 | 极好 |
| Thyroid gland-Papillary cancer | 0.506 | 0.700 | 好 |
| Serous effusion-Diseased cells | 0.404 | 0.603 | 好 |
| Urine-SHGUC | 0.334 | 0.455 | 中等 |
| ascus | 0.315 | 0.533 | 中等 |
| lsil | 0.298 | 0.537 | 中等 |
| agc_adenocarcinoma_em | 0.282 | 0.431 | 中等 |
| respiratory tract-Diseased cells | 0.273 | 0.378 | 中等 |
| vaginalis | 0.220 | 0.463 | 较差 |
| asch | 0.197 | 0.376 | 较差 |
| ec | 0.168 | 0.298 | 较差 |
| Urine-AUC | 0.081 | 0.109 | 差 |
| respiratory tract-adenocarcinoma | 0.015 | 0.019 | 极差 |

---

## 3. Novel 类详细结果 (零样本)

| 类别 | AP | AP_50 | 样本数 | 表现 |
|------|-----|-------|--------|------|
| Urine-HGUC | 0.5703 | 0.7211 | 44 | 极好 |
| hsil_scc_omn | 0.0735 | 0.1455 | 1942 | 中等 |
| Thyroid gland-Malignant tumour | 0.0428 | 0.0510 | 31 | 较差 |
| Thyroid gland-Suspicious papillary cancer | 0.0404 | 0.0661 | 2529 | 较差 |
| respiratory tract-Squamous cell carcinoma | 0.0165 | 0.0179 | 27 | 差 |
| Serous effusion-Ovarian cancer | 0.0071 | 0.0079 | 19 | 差 |
| Serous effusion-adenocarcinoma | 0.0001 | 0.0002 | 4 | 失败 |
| monilia | 0.0000 | 0.0000 | 459 | 失败 |
| Thyroid gland-AUC | 0.0000 | 0.0000 | 158 | 失败 |
| Thyroid gland-NS | 0.0000 | 0.0000 | 32 | 失败 |
| respiratory tract-Small cell carcinoma | 0.0000 | 0.0000 | 396 | 失败 |

---

## 4. 训练曲线

| Epoch | mAP | mAP_50 | 备注 |
|-------|-----|--------|------|
| 1 | 0.154 | 0.251 | |
| 2 | 0.197 | 0.306 | |
| 3 | 0.212 | 0.329 | |
| 4 | 0.220 | 0.344 | |
| 5 | 0.240 | 0.371 | |
| 6 | 0.241 | 0.371 | |
| 7 | 0.248 | 0.382 | |
| 8 | 0.247 | 0.379 | |
| **9** | **0.254** | **0.390** | **最佳** |
| 10 | 0.236 | 0.364 | 开始下降 |
| 11 | 0.235 | 0.362 | |
| 12 | 0.232 | 0.355 | |
| 13 | 0.223 | 0.341 | |
| 14 | 0.217 | 0.332 | |
| 15 | 0.214 | 0.328 | |

---

## 5. 关键发现

- **最佳 Epoch**: 9 (之后开始过拟合)
- **最佳 Novel 类**: Urine-HGUC (AP: 0.5703) - 与 Base 类 Urine-SHGUC 特征相似
- **失败类 (AP=0)**: 5个类，主要是文本描述太简单或特征差异大
- **过拟合明显**: Epoch 9 后 mAP 持续下降

---

*此报告基于训练日志和评估结果整理*
