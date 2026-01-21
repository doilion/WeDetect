# WeDetect TCT_NGC V2 数据集实验总结

本文档汇总了在 TCT_NGC V2 数据集上进行的所有 WeDetect 实验。

---

## 实验概览

| # | 实验 | 配置 | Best Epoch | Base mAP | Base mAP (排除负样本) | Novel mAP | 状态 |
|---|------|------|------------|----------|----------------------|-----------|------|
| 1 | 基准 (Baseline) | lr=2e-4, epochs=15 | 9 | 25.4% | **32.1%** | 6.8% | ✅ 完成 |
| 2 | 实验1: 降低学习率 | lr=1e-4, epochs=20 | 9 | 24.5% | 30.7% | 7.2% | ✅ 完成 |
| 3 | 实验2: 冻结Backbone | 冻结前两层, lr=2e-4 | 8 | 22.5% | 28.4% | 8.1% | ✅ 完成 |
| 4 | 实验3: 调整Loss权重 | cls=1.0, bbox=5.0 | 9 | 23.2% | 29.6% | **9.1%** | ✅ 完成 |
| 5 | 实验4: 组合策略 | 冻结+Loss调整 | 7 | 19.9% | 24.3% | 9.0% | ✅ 完成 |

**运行环境**:
- **GPU**: NVIDIA RTX 5880 Ada Generation (48GB)
- **服务器**: 单 GPU

---

## 数据集信息

| 项目 | 值 |
|------|-----|
| 数据集 | TCT_NGC V2 (细胞病理学) |
| 数据路径 | `/root/datasets/TCT_NGC/` |
| Base 类别 | 20 类 (包含 5 个 negative) |
| Novel 类别 | 11 类 (零样本评估) |
| 训练集 | `annotations/train_base_v2.json` |
| Base 测试集 | `annotations/test_base_v2.json` |
| Novel 测试集 | `annotations/test_novel_v2.json` |

### 类别划分

**Base 类别 (20类)** - 用于训练:
- **宫颈**: normal, ascus, asch, lsil, agc_adenocarcinoma_em, vaginalis, dysbacteriosis_herpes_act, ec
- **浆膜腔**: Serous effusion-Negative samples, Serous effusion-Diseased cells, Serous effusion-Breast cancer
- **甲状腺**: Thyroid gland-Papillary cancer, Thyroid gland-Negative samples, Thyroid gland-Suspicious for Malignancy
- **尿液**: Urine-Negative, Urine-SHGUC, Urine-AUC
- **呼吸道**: respiratory tract-Negative samples, respiratory tract-Diseased cells, respiratory tract-adenocarcinoma

**Novel 类别 (11类)** - 零样本检测:
- **宫颈**: hsil_scc, monilia
- **浆膜腔**: Serous effusion-Ovarian cancer, Serous effusion-Adenocarcinoma
- **甲状腺**: Thyroid gland-Suspicious for Papillary Cancer, Thyroid gland-Atypia of Undetermined Significance, Thyroid gland-Malignant, Thyroid gland-Nondiagnostic or Unsatisfactory
- **尿液**: Urine-HGUC
- **呼吸道**: respiratory tract-squamous carcinoma, respiratory tract-small cell carcinoma

**评估时排除的 Negative 类别 (5类)**:
- normal, Serous effusion-Negative samples, Thyroid gland-Negative samples, Urine-Negative, respiratory tract-Negative samples

---

## 实验 1: 基准 (Baseline)

### 配置信息

| 项目 | 值 |
|------|-----|
| 配置文件 | `config/wedetect_tiny_tct.py` |
| 工作目录 | `work_dirs/wedetect_tiny_tct/` |
| Checkpoint | `best_coco_bbox_mAP_epoch_9.pth` |
| 训练日志 | `20251231_024844/20251231_024844.log` |
| 总 Epochs | 15 |
| 最佳 Epoch | 9 |

### 模型配置

```python
backbone=dict(
    type="MultiModalYOLOBackbone",
    image_model=dict(
        type="ConvNextVisionBackbone",
        model_name="tiny",
        frozen_modules=[],  # 不冻结
    ),
    text_model=dict(
        type="XLMRobertaLanguageBackbone",
        model_name="./xlm-roberta-base/",
        model_size="tiny",
        frozen_modules=[],
    ),
)
```

### 训练参数

| 参数 | 值 |
|------|-----|
| Base LR | 2e-4 |
| Text Model LR | 0.01x (2e-6) |
| Weight Decay | 0.05 |
| Batch Size | 28 |
| 优化器 | AdamW |
| Loss Cls Weight | 0.5 |
| Loss BBox Weight | 7.5 |
| Loss DFL Weight | 0.375 |
| 预训练权重 | `checkpoints/wedetect_tiny.pth` |

### 总体结果

| 评估集 | mAP | mAP@50 | mAP@75 |
|--------|-----|--------|--------|
| **Base** (全部20类) | 25.4% | 39.0% | 28.7% |
| **Base** (排除负样本, 15类) | **32.1%** | 48.7% | 36.4% |
| **Novel** (零样本, 11类) | **6.8%** | 9.2% | 7.5% |

### Base Per-Class AP (排除负样本，15类)

| 类别 | AP | AP@50 | 域 |
|------|-----|-------|-----|
| Thyroid gland-Suspicious for Malignancy | 62.1% | 90.2% | 甲状腺 |
| Serous effusion-Breast cancer | 55.7% | 74.2% | 浆膜腔 |
| dysbacteriosis_herpes_act | 53.8% | 76.4% | 宫颈 |
| Thyroid gland-Papillary cancer | 50.6% | 70.0% | 甲状腺 |
| Serous effusion-Diseased cells | 40.4% | 60.3% | 浆膜腔 |
| Urine-SHGUC | 33.4% | 45.5% | 尿液 |
| ascus | 31.5% | 53.3% | 宫颈 |
| lsil | 29.8% | 53.7% | 宫颈 |
| agc_adenocarcinoma_em | 28.2% | 43.1% | 宫颈 |
| respiratory tract-Diseased cells | 27.3% | 37.8% | 呼吸道 |
| vaginalis | 22.0% | 46.3% | 宫颈 |
| asch | 19.7% | 37.6% | 宫颈 |
| ec | 16.8% | 29.8% | 宫颈 |
| Urine-AUC | 8.1% | 10.9% | 尿液 |
| respiratory tract-adenocarcinoma | 1.5% | 1.9% | 呼吸道 |

### Novel Per-Class AP (零样本，11类)

| 类别 | AP | AP@50 | 样本数 | 域 |
|------|-----|-------|--------|-----|
| **Urine-HGUC** | **57.0%** | 72.1% | 44 | 尿液 |
| hsil_scc_omn | 7.4% | 14.6% | 1942 | 宫颈 |
| Thyroid gland-Malignant tumour | 4.3% | 5.1% | 31 | 甲状腺 |
| Thyroid gland-Suspicious papillary cancer | 4.0% | 6.6% | 2529 | 甲状腺 |
| respiratory tract-Squamous cell carcinoma | 1.7% | 1.8% | 27 | 呼吸道 |
| Serous effusion-Ovarian cancer | 0.7% | 0.8% | 19 | 浆膜腔 |
| Serous effusion-adenocarcinoma | 0.0% | 0.0% | 4 | 浆膜腔 |
| monilia | 0.0% | 0.0% | 459 | 宫颈 |
| Thyroid gland-AUC | 0.0% | 0.0% | 158 | 甲状腺 |
| Thyroid gland-NS | 0.0% | 0.0% | 32 | 甲状腺 |
| respiratory tract-Small cell carcinoma | 0.0% | 0.0% | 396 | 呼吸道 |

### 训练曲线

| Epoch | mAP | mAP@50 | 备注 |
|-------|-----|--------|------|
| 1 | 15.4% | 25.1% | |
| 2 | 19.7% | 30.6% | |
| 3 | 21.2% | 32.9% | |
| 4 | 22.0% | 34.4% | |
| 5 | 24.0% | 37.1% | |
| 6 | 24.1% | 37.1% | |
| 7 | 24.8% | 38.2% | |
| 8 | 24.7% | 37.9% | |
| **9** | **25.4%** | **39.0%** | **最佳** |
| 10 | 23.6% | 36.4% | 开始下降 |
| 11 | 23.5% | 36.2% | |
| 12 | 23.2% | 35.5% | |
| 13 | 22.3% | 34.1% | |
| 14 | 21.7% | 33.2% | |
| 15 | 21.4% | 32.8% | |

### 关键发现

- **过拟合明显**: Epoch 9 后 mAP 持续下降
- **最佳 Novel 类**: Urine-HGUC (57.0%) - 与 Base 类 Urine-SHGUC 特征相似
- **失败类 (mAP=0)**: 5 个 Novel 类，主要是文本描述太简单或特征差异大

---

## 实验 2: 降低学习率

### 配置信息

| 项目 | 值 |
|------|-----|
| 配置文件 | `config/wedetect_tiny_tct.py` (修改后) |
| 工作目录 | `work_dirs/wedetect_tiny_tct/` |
| 训练日志 | `20260112_170036/20260112_170036.log` |
| 总 Epochs | 20 |
| 最佳 Epoch | 9 |

### 配置变更 (vs 基准)

| 参数 | 基准 | 实验1 |
|------|------|-------|
| Base LR | 2e-4 | **1e-4** |
| Max Epochs | 15 | **20** |

### 总体结果

| 评估集 | mAP | mAP@50 | vs 基准 |
|--------|-----|--------|---------|
| **Base** (全部20类) | 24.5% | 37.4% | -0.9% ❌ |
| **Base** (排除负样本, 15类) | 30.7% | 46.5% | -1.4% ❌ |
| **Novel** (零样本, 11类) | **7.2%** | 9.9% | +0.4% ✅ |

### Base Per-Class AP (排除负样本，15类)

| 类别 | AP | AP@50 | vs 基准 | 域 |
|------|-----|-------|---------|-----|
| Thyroid gland-Suspicious for Malignancy | 59.3% | 84.2% | -2.8% | 甲状腺 |
| Serous effusion-Breast cancer | 56.2% | 75.0% | +0.5% | 浆膜腔 |
| dysbacteriosis_herpes_act | 53.3% | 75.9% | -0.5% | 宫颈 |
| Thyroid gland-Papillary cancer | 49.2% | 67.6% | -1.4% | 甲状腺 |
| ascus | 32.9% | 55.1% | +1.4% | 宫颈 |
| lsil | 32.1% | 55.5% | +2.3% | 宫颈 |
| agc_adenocarcinoma_em | 31.4% | 47.9% | +3.2% | 宫颈 |
| Urine-SHGUC | 31.2% | 42.9% | -2.2% | 尿液 |
| Serous effusion-Diseased cells | 25.6% | 36.3% | -14.8% | 浆膜腔 |
| vaginalis | 23.0% | 47.7% | +1.0% | 宫颈 |
| asch | 22.1% | 41.2% | +2.4% | 宫颈 |
| ec | 17.3% | 31.0% | +0.5% | 宫颈 |
| Urine-AUC | 13.5% | 18.3% | +5.4% | 尿液 |
| respiratory tract-Diseased cells | 12.6% | 16.9% | -14.7% | 呼吸道 |
| respiratory tract-adenocarcinoma | 1.1% | 1.4% | -0.4% | 呼吸道 |

### Novel Per-Class AP (零样本，11类)

| 类别 | AP | AP@50 | vs 基准 | 样本数 | 域 |
|------|-----|-------|---------|--------|-----|
| **Urine-HGUC** | **54.3%** | 70.2% | -2.7% | 44 | 尿液 |
| Serous effusion-Ovarian cancer | 9.7% | 11.3% | +9.0% ✅ | 19 | 浆膜腔 |
| hsil_scc_omn | 8.9% | 17.4% | +1.5% | 1942 | 宫颈 |
| Thyroid gland-Suspicious papillary cancer | 3.7% | 5.3% | -0.3% | 2529 | 甲状腺 |
| respiratory tract-Squamous cell carcinoma | 1.7% | 2.3% | 0.0% | 27 | 呼吸道 |
| respiratory tract-Small cell carcinoma | 1.1% | 1.5% | +1.1% ✅ | 396 | 呼吸道 |
| Thyroid gland-Malignant tumour | 0.5% | 0.5% | -3.8% | 31 | 甲状腺 |
| Serous effusion-adenocarcinoma | 0.0% | 0.0% | 0.0% | 4 | 浆膜腔 |
| monilia | 0.0% | 0.0% | 0.0% | 459 | 宫颈 |
| Thyroid gland-AUC | 0.0% | 0.0% | 0.0% | 158 | 甲状腺 |
| Thyroid gland-NS | 0.0% | 0.0% | 0.0% | 32 | 甲状腺 |

### 训练曲线

| Epoch | mAP | mAP@50 | 备注 |
|-------|-----|--------|------|
| 1 | 11.6% | 18.3% | |
| 2 | 15.9% | 24.5% | |
| 3 | 17.9% | 27.8% | |
| 4 | 19.6% | 30.6% | |
| 5 | 21.8% | 34.1% | |
| 6 | 22.8% | 35.5% | |
| 7 | 23.2% | 36.1% | |
| 8 | 24.1% | 37.0% | |
| **9** | **24.5%** | **37.4%** | **最佳** |
| 10 | 24.0% | 36.8% | 开始下降 |
| ... | ... | ... | |
| 20 | 19.2% | 29.4% | |

### 关键发现

- **Base 性能更差**: mAP 30.7% vs 基准 32.1% (-1.4%)
- **Novel 性能略好**: mAP 7.2% vs 基准 6.8% (+0.4%)
- **最佳 Epoch 仍是 9**: 过拟合问题与学习率无关
- **部分类别提升明显**:
  - Serous effusion-Ovarian cancer: +9.0% (0.7% → 9.7%)
  - respiratory tract-Small cell carcinoma: +1.1% (0% → 1.1%)
- **结论**: lr=1e-4 收敛太慢，但可能有助于 Novel 泛化

---

## 实验 3: 冻结 Backbone 前两层

### 配置信息

| 项目 | 值 |
|------|-----|
| 配置文件 | `config/wedetect_tiny_tct_exp2.py` |
| 工作目录 | `work_dirs/wedetect_tiny_tct_exp2/` |
| Checkpoint | `best_coco_bbox_mAP_epoch_8.pth` |
| 训练日志 | `20260115_201341/vis_data/*.log` |
| 总 Epochs | 12 |
| 最佳 Epoch | 8 |
| 状态 | ✅ 完成 |

### 配置变更 (vs 基准)

| 参数 | 基准 | 实验2 |
|------|------|-------|
| Max Epochs | 15 | **12** |
| Image Backbone | 不冻结 | **冻结前两层** |
| Batch Size | 28 | **14** |

### 总体结果

| 评估集 | mAP | mAP@50 | vs 基准 |
|--------|-----|--------|---------|
| **Base** (全部20类) | 22.5% | 34.8% | -2.9% ❌ |
| **Base** (排除负样本, 15类) | 28.4% | 43.1% | -3.7% ❌ |
| **Novel** (零样本, 11类) | **8.1%** | 10.7% | +1.3% ✅ |

### Base Per-Class AP (排除负样本，15类)

| 类别 | AP | vs 基准 | 域 |
|------|-----|---------|-----|
| Serous effusion-Breast cancer | 65.9% | +10.2% ✅ | 浆膜腔 |
| dysbacteriosis_herpes_act | 53.2% | -0.6% | 宫颈 |
| Thyroid gland-Suspicious for Malignancy | 52.7% | -9.4% | 甲状腺 |
| Thyroid gland-Papillary cancer | 49.5% | -1.1% | 甲状腺 |
| agc_adenocarcinoma_em | 39.5% | +11.3% ✅ | 宫颈 |
| ascus | 30.9% | -0.6% | 宫颈 |
| lsil | 30.5% | +0.7% | 宫颈 |
| Urine-SHGUC | 27.4% | -6.0% | 尿液 |
| vaginalis | 20.1% | -1.9% | 宫颈 |
| asch | 18.7% | -1.0% | 宫颈 |
| ec | 16.4% | -0.4% | 宫颈 |
| respiratory tract-Diseased cells | 11.2% | -16.1% ❌ | 呼吸道 |
| Urine-AUC | 4.7% | -3.4% | 尿液 |
| Serous effusion-Diseased cells | 3.7% | -36.7% ❌ | 浆膜腔 |
| respiratory tract-adenocarcinoma | 1.6% | +0.1% | 呼吸道 |

### Novel Per-Class AP (零样本，11类)

| 类别 | AP | AP@50 | vs 基准 | 样本数 | 域 |
|------|-----|-------|---------|--------|-----|
| **Urine-HGUC** | **55.1%** | 69.4% | -1.9% | 44 | 尿液 |
| respiratory tract-Small cell carcinoma | 7.3% | 9.7% | +7.3% ✅ | 396 | 呼吸道 |
| hsil_scc_omn | 7.2% | 14.6% | -0.2% | 1942 | 宫颈 |
| Thyroid gland-Suspicious papillary cancer | 7.1% | 10.8% | +3.1% ✅ | 2529 | 甲状腺 |
| Serous effusion-Ovarian cancer | 5.6% | 6.3% | +4.9% ✅ | 19 | 浆膜腔 |
| Serous effusion-adenocarcinoma | 4.0% | 4.5% | +4.0% ✅ | 4 | 浆膜腔 |
| Thyroid gland-Malignant tumour | 1.9% | 2.5% | -2.4% | 31 | 甲状腺 |
| Thyroid gland-AUC | 0.2% | 0.3% | +0.2% | 158 | 甲状腺 |
| respiratory tract-Squamous cell carcinoma | 0.1% | 0.2% | -1.6% | 27 | 呼吸道 |
| monilia | 0.0% | 0.0% | 0.0% | 459 | 宫颈 |
| Thyroid gland-NS | 0.0% | 0.0% | 0.0% | 32 | 甲状腺 |

### 训练曲线

| Epoch | mAP | mAP@50 | 备注 |
|-------|-----|--------|------|
| 1 | 17.0% | 27.6% | |
| 2 | 19.2% | 30.6% | |
| 3 | 19.3% | 30.6% | |
| 4 | 21.3% | 32.9% | |
| 5 | 22.3% | 34.4% | |
| 6 | 21.6% | 33.3% | |
| 7 | 21.8% | 33.7% | |
| **8** | **22.5%** | **34.8%** | **最佳** |
| 9 | 21.5% | 33.3% | 开始下降 |
| 10 | 22.0% | 34.1% | |
| 11 | 21.3% | 32.8% | |
| 12 | 21.1% | 32.4% | |

### 关键发现

- **Base 性能下降**: mAP 28.4% vs 基准 32.1% (-3.7%)
- **Novel 性能提升**: mAP 8.1% vs 基准 6.8% (+1.3%)
- **最佳 Epoch 提前到 8**: 冻结 backbone 减缓了过拟合
- **部分 Novel 类显著提升**:
  - respiratory tract-Small cell carcinoma: +7.3% (0% → 7.3%)
  - Serous effusion-Ovarian cancer: +4.9% (0.7% → 5.6%)
  - Serous effusion-adenocarcinoma: +4.0% (0% → 4.0%)
- **Base 类严重下降**:
  - Serous effusion-Diseased cells: -36.7% (40.4% → 3.7%)
  - respiratory tract-Diseased cells: -16.1% (27.3% → 11.2%)
- **结论**: 冻结 backbone 有助于 Novel 泛化，但损害 Base 性能

---

## 实验 4: 调整 Loss 权重

### 配置信息

| 项目 | 值 |
|------|-----|
| 配置文件 | `config/wedetect_tiny_tct_exp3.py` |
| 工作目录 | `work_dirs/wedetect_tiny_tct_exp3/` |
| Checkpoint | `best_coco_bbox_mAP_epoch_9.pth` |
| 训练日志 | `20260115_203938/vis_data/*.log` |
| 总 Epochs | 12 |
| 最佳 Epoch | 9 |
| 状态 | ✅ 完成 |

### 配置变更 (vs 基准)

| 参数 | 基准 | 实验3 |
|------|------|-------|
| Max Epochs | 15 | **12** |
| Loss Cls Weight | 0.5 | **1.0** |
| Loss BBox Weight | 7.5 | **5.0** |
| Batch Size | 28 | **10** |

### 总体结果

| 评估集 | mAP | mAP@50 | vs 基准 |
|--------|-----|--------|---------|
| **Base** (全部20类) | 23.2% | 36.1% | -2.2% ❌ |
| **Base** (排除负样本, 15类) | 29.6% | 45.5% | -2.5% ❌ |
| **Novel** (零样本, 11类) | **9.1%** | 11.8% | **+2.3%** ✅ |

### Base Per-Class AP (排除负样本，15类)

| 类别 | AP | vs 基准 | 域 |
|------|-----|---------|-----|
| Serous effusion-Breast cancer | 58.9% | +3.2% | 浆膜腔 |
| dysbacteriosis_herpes_act | 52.8% | -1.0% | 宫颈 |
| Thyroid gland-Suspicious for Malignancy | 50.2% | -11.9% | 甲状腺 |
| Thyroid gland-Papillary cancer | 44.2% | -6.4% | 甲状腺 |
| Serous effusion-Diseased cells | 32.3% | -8.1% | 浆膜腔 |
| agc_adenocarcinoma_em | 32.6% | +4.4% | 宫颈 |
| ascus | 30.5% | -1.0% | 宫颈 |
| lsil | 30.6% | +0.8% | 宫颈 |
| vaginalis | 22.5% | +0.5% | 宫颈 |
| respiratory tract-Diseased cells | 21.2% | -6.1% | 呼吸道 |
| Urine-SHGUC | 21.0% | -12.4% | 尿液 |
| asch | 20.4% | +0.7% | 宫颈 |
| ec | 15.6% | -1.2% | 宫颈 |
| Urine-AUC | 10.7% | +2.6% | 尿液 |
| respiratory tract-adenocarcinoma | 0.6% | -0.9% | 呼吸道 |

### Novel Per-Class AP (零样本，11类)

| 类别 | AP | AP@50 | vs 基准 | 样本数 | 域 |
|------|-----|-------|---------|--------|-----|
| **Urine-HGUC** | **68.2%** | 84.6% | **+11.2%** ✅ | 44 | 尿液 |
| Serous effusion-Ovarian cancer | 9.9% | 11.0% | +9.2% ✅ | 19 | 浆膜腔 |
| respiratory tract-Small cell carcinoma | 9.0% | 12.0% | +9.0% ✅ | 396 | 呼吸道 |
| hsil_scc_omn | 6.6% | 13.2% | -0.8% | 1942 | 宫颈 |
| Thyroid gland-Suspicious papillary cancer | 4.6% | 7.0% | +0.6% | 2529 | 甲状腺 |
| respiratory tract-Squamous cell carcinoma | 1.0% | 1.3% | -0.7% | 27 | 呼吸道 |
| Serous effusion-adenocarcinoma | 0.7% | 0.9% | +0.7% | 4 | 浆膜腔 |
| Thyroid gland-AUC | 0.1% | 0.1% | +0.1% | 158 | 甲状腺 |
| Thyroid gland-Malignant tumour | 0.0% | 0.0% | -4.3% | 31 | 甲状腺 |
| monilia | 0.0% | 0.0% | 0.0% | 459 | 宫颈 |
| Thyroid gland-NS | 0.0% | 0.0% | 0.0% | 32 | 甲状腺 |

### 训练曲线

| Epoch | mAP | mAP@50 | 备注 |
|-------|-----|--------|------|
| 1 | 17.5% | 28.1% | |
| 2 | 20.0% | 32.3% | |
| 3 | 19.4% | 31.0% | |
| 4 | 21.4% | 34.0% | |
| 5 | 21.1% | 33.0% | |
| 6 | 22.5% | 34.9% | |
| 7 | 22.7% | 35.6% | |
| 8 | 22.5% | 34.9% | |
| **9** | **23.2%** | **36.1%** | **最佳** |
| 10 | 22.2% | 34.5% | 开始下降 |
| 11 | 22.1% | 34.3% | |
| 12 | 21.4% | 33.0% | |

### 关键发现

- **Novel 性能最佳**: mAP 9.1% vs 基准 6.8% (+2.3%)，是所有实验中最高
- **Base 性能下降**: mAP 29.6% vs 基准 32.1% (-2.5%)
- **Urine-HGUC 突破**: 68.2% AP，比基准提升 11.2%
- **多个 Novel 类显著提升**:
  - Urine-HGUC: +11.2% (57.0% → 68.2%)
  - Serous effusion-Ovarian cancer: +9.2% (0.7% → 9.9%)
  - respiratory tract-Small cell carcinoma: +9.0% (0% → 9.0%)
- **结论**: 增加分类 loss 权重有效提升 Novel 类泛化能力

---

## 实验 5: 组合策略 (冻结 Backbone + 调整 Loss 权重)

### 配置信息

| 项目 | 值 |
|------|-----|
| 配置文件 | `config/wedetect_tiny_tct_exp4.py` |
| 工作目录 | `work_dirs/wedetect_tiny_tct_exp4/` |
| Checkpoint | `best_coco_bbox_mAP_epoch_7.pth` |
| 训练日志 | `work_dirs/exp4_train.log` |
| 总 Epochs | 12 |
| 最佳 Epoch | 7 |
| 状态 | ✅ 完成 |

### 配置变更 (vs 基准)

| 参数 | 基准 | 实验4 |
|------|------|-------|
| Max Epochs | 15 | **12** |
| Image Backbone | 不冻结 | **冻结前两层** (来自实验2) |
| Loss Cls Weight | 0.5 | **1.0** (来自实验3) |
| Loss BBox Weight | 7.5 | **5.0** (来自实验3) |
| Batch Size | 28 | **20** (2卡 × 10) |
| Base LR | 2e-4 | **1.43e-4** (按batch_size缩放) |

### 总体结果

| 评估集 | mAP | mAP@50 | vs 基准 |
|--------|-----|--------|---------|
| **Base** (全部20类) | 19.9% | 30.7% | -5.5% ❌ |
| **Base** (排除负样本, 15类) | 24.3% | 37.5% | **-7.8%** ❌ |
| **Novel** (零样本, 11类) | **9.0%** | 11.7% | +2.2% ✅ |

### Novel Per-Class AP (零样本，11类)

| 类别 | AP | AP@50 | vs 基准 | 样本数 | 域 |
|------|-----|-------|---------|--------|-----|
| **Urine-HGUC** | **64.9%** | 80.7% | +7.9% ✅ | 44 | 尿液 |
| respiratory tract-Small cell carcinoma | 7.9% | 10.5% | +7.9% ✅ | 396 | 呼吸道 |
| Serous effusion-Ovarian cancer | 7.3% | 7.6% | +6.6% ✅ | 19 | 浆膜腔 |
| Thyroid gland-Suspicious papillary cancer | 7.0% | 11.6% | +3.0% ✅ | 2529 | 甲状腺 |
| hsil_scc_omn | 6.5% | 13.2% | -0.9% | 1942 | 宫颈 |
| Thyroid gland-Malignant tumour | 3.4% | 4.5% | -0.9% | 31 | 甲状腺 |
| Serous effusion-adenocarcinoma | 0.9% | 1.1% | +0.9% | 4 | 浆膜腔 |
| respiratory tract-Squamous cell carcinoma | 0.6% | 1.1% | -1.1% | 27 | 呼吸道 |
| Thyroid gland-AUC | 0.1% | 0.2% | +0.1% | 158 | 甲状腺 |
| monilia | 0.0% | 0.0% | 0.0% | 459 | 宫颈 |
| Thyroid gland-NS | 0.0% | 0.0% | 0.0% | 32 | 甲状腺 |

### 训练曲线

| Epoch | mAP | mAP@50 | 备注 |
|-------|-----|--------|------|
| 1 | 18.7% | 29.8% | |
| 2 | 18.1% | 29.1% | |
| 3 | 20.5% | 32.6% | |
| 4 | 20.9% | 33.4% | |
| 5 | 23.7% | 37.2% | |
| 6 | 22.2% | 34.8% | |
| **7** | **24.3%** | **37.5%** | **最佳** |
| 8 | 22.2% | 34.7% | 开始下降 |
| 9 | 22.7% | 35.4% | |
| 10 | 21.2% | 32.9% | |
| 11 | 20.8% | 32.2% | |
| 12 | 19.9% | 30.7% | |

### 关键发现

- **Base 性能严重下降**: mAP 24.3% vs 基准 32.1% (**-7.8%**)，是所有实验中下降最多的
- **Novel 性能提升**: mAP 9.0% vs 基准 6.8% (+2.2%)，略低于实验3的 9.1%
- **组合策略未达预期**:
  - 实验2 (冻结): Base -3.7%, Novel +1.3%
  - 实验3 (Loss): Base -2.5%, Novel +2.3%
  - 实验4 (组合): Base **-7.8%**, Novel +2.2%
  - 组合策略的 Base 下降幅度超过了两者相加 (-3.7% + -2.5% = -6.2%)
- **最佳 Epoch 提前到 7**: 比基准和实验3都更早
- **结论**: 冻结 backbone 和调整 loss 权重的组合产生了负面叠加效应，严重损害 Base 性能，而 Novel 性能提升并未叠加

---

## 问题分析

### 1. 过拟合问题

- 两次实验最佳 Epoch 都在 9
- Epoch 9 后 mAP 持续下降
- **解决方案**: 实验2 冻结 backbone，实验3 调整 loss 权重

### 2. 低表现类别分析

#### Base 类低表现 (mAP < 20%):

| 类别 | mAP | 分析 |
|------|-----|------|
| respiratory tract-adenocarcinoma | 1.5% | 样本少，与其他呼吸道类混淆 |
| Urine-AUC | 8.1% | 特征不明显 |
| ec | 16.8% | 颈管细胞特征复杂 |
| asch | 19.7% | 与 ascus 混淆 |

#### Novel 类失败 (mAP = 0):

| 类别 | 样本数 | 分析 |
|------|--------|------|
| monilia | 459 | 真菌特征与训练集不相似 |
| Thyroid gland-AUC | 158 | 文本描述太简单 |
| Thyroid gland-NS | 32 | 文本描述太简单 |
| respiratory tract-Small cell carcinoma | 396 | 小细胞癌特征独特 |
| Serous effusion-adenocarcinoma | 4 | 样本太少 |

### 3. 文本描述改进建议

```json
// Novel 类文本描述改进建议
[
  ["hsil scc", "high grade squamous intraepithelial lesion", "squamous cell carcinoma"],
  ["monilia", "candida", "fungal infection", "yeast cells"],
  ["serous ovarian cancer", "ovarian carcinoma cells", "ovarian malignancy"],
  ["thyroid auc", "thyroid atypia undetermined significance", "thyroid uncertain"],
  ["thyroid ns", "thyroid nondiagnostic", "thyroid unsatisfactory sample"],
  ["respiratory squamous", "lung squamous cell carcinoma", "squamous lung cancer"],
  ["respiratory small cell", "small cell lung carcinoma", "oat cell carcinoma"]
]
```

---

## 文件路径汇总

### 配置文件

| 实验 | 配置文件 |
|------|---------|
| 基准 / 实验1 | `config/wedetect_tiny_tct.py` |
| 实验2 (冻结) | `config/wedetect_tiny_tct_exp2.py` |
| 实验3 (Loss) | `config/wedetect_tiny_tct_exp3.py` |
| 实验4 (组合) | `config/wedetect_tiny_tct_exp4.py` |

### Checkpoint

| 实验 | Checkpoint |
|------|-----------|
| 基准 | `work_dirs/wedetect_tiny_tct/best_coco_bbox_mAP_epoch_9.pth` |
| 实验2 (冻结) | `work_dirs/wedetect_tiny_tct_exp2/best_coco_bbox_mAP_epoch_8.pth` |
| 实验3 (Loss) | `work_dirs/wedetect_tiny_tct_exp3/best_coco_bbox_mAP_epoch_9.pth` |
| 实验4 (组合) | `work_dirs/wedetect_tiny_tct_exp4/best_coco_bbox_mAP_epoch_7.pth` |

### 评估脚本

| 评估类型 | 脚本 |
|----------|------|
| Base 评估 (排除负样本) | `test_exclude_negative.py` |
| Novel 评估 (实验2) | `eval_novel_exp2.py` |
| Novel 评估 (实验3) | `eval_novel_exp3.py` |
| Novel 评估 (实验4) | `eval_novel_exp4.py` |

---

## 运行命令

### 训练

```bash
# 实验2: 冻结 backbone
CUDA_VISIBLE_DEVICES=0 python train.py config/wedetect_tiny_tct_exp2.py

# 实验3: 调整 loss 权重
CUDA_VISIBLE_DEVICES=0 python train.py config/wedetect_tiny_tct_exp3.py
```

### 评估

```bash
# Base 评估 (排除负样本)
python test_exclude_negative.py --checkpoint work_dirs/wedetect_tiny_tct/best_coco_bbox_mAP_epoch_9.pth

# Novel 评估
python eval_novel_manual.py
```

---

## 总结与目标

| 指标 | 基准 | 实验2 (冻结) | 实验3 (Loss) | 实验4 (组合) | 最佳 |
|------|------|-------------|-------------|-------------|------|
| Base mAP (排除负样本) | **32.1%** | 28.4% | 29.6% | 24.3% | 基准 |
| Novel mAP (零样本) | 6.8% | 8.1% | **9.1%** | 9.0% | 实验3 |
| 最佳 Epoch | 9 | 8 | 9 | 7 | - |

### 关键结论

1. **Base vs Novel 权衡**: 所有改进 Novel 性能的方法都会降低 Base 性能
2. **最佳 Novel 策略**: 调整 Loss 权重 (cls=1.0, bbox=5.0) 提升 Novel mAP 最多 (+2.3%)
3. **组合策略失败**: 实验4组合冻结+Loss调整，Base性能下降-7.8%，Novel提升仅+2.2%，未达到叠加效果
4. **Urine-HGUC 效果最好**: 在实验3中达到 68.2% AP，是所有 Novel 类中最高
5. **顽固失败类**: monilia, Thyroid gland-NS 在所有实验中都是 0%，需要改进文本描述

### 实验对比分析

| 实验 | Base 变化 | Novel 变化 | 分析 |
|------|----------|-----------|------|
| 实验2 (冻结) | -3.7% | +1.3% | 冻结backbone减缓过拟合，但损害Base性能 |
| 实验3 (Loss) | -2.5% | +2.3% | 增加分类loss权重最有效提升Novel |
| 实验4 (组合) | **-7.8%** | +2.2% | 负面叠加效应，严重损害Base，Novel未叠加 |

### 下一步建议

1. ~~**组合策略**~~: 已完成，效果不佳（负面叠加）
2. **文本描述改进**: 针对 mAP=0 的 Novel 类（monilia, Thyroid gland-NS等）改进文本描述
3. **数据增强**: 增加更多数据增强策略减少过拟合
4. **Early Stopping**: 设置早停策略，在 epoch 8-9 左右停止训练
5. **单独策略优化**: 基于实验3（Loss调整）进一步优化，避免组合策略

---

*文档创建时间: 2026-01-13*
*最后更新时间: 2026-01-20*
