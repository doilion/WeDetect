# 实验结果汇总：exp5 / exp6 / exp7

## 实验配置

| 实验 | 文本语言 | 数据增强 | Best Epoch | Checkpoint |
|------|---------|---------|------------|------------|
| exp5 | 英文 | 弱增强 | epoch 8 | `best_coco_bbox_mAP_epoch_8.pth` |
| exp6 | 中文 | 强增强 | epoch 9 | `best_coco_bbox_mAP_epoch_9.pth` |
| exp7 | 英文 | 强增强 | epoch 10 | `best_coco_bbox_mAP_epoch_10.pth` |

---

## 一、Base 类别评估结果（排除 Negative 类别）

### 1.1 总体 mAP

| 实验 | mAP | mAP_50 | mAP_75 |
|------|-----|--------|--------|
| exp5 | 0.284 | 0.441 | 0.318 |
| exp6 | 0.302 | 0.463 | 0.341 |
| **exp7** | **0.314** | **0.482** | **0.351** |

### 1.2 每个类别的 AP 值

#### exp5 (英文 + 弱增强)

| 类别 | mAP | mAP_50 | mAP_75 |
|------|-----|--------|--------|
| ascus | 0.304 | 0.526 | 0.322 |
| asch | 0.202 | 0.385 | 0.194 |
| lsil | 0.292 | 0.542 | 0.278 |
| agc_adenocarcinoma_em | 0.348 | 0.551 | 0.388 |
| vaginalis | 0.232 | 0.494 | 0.165 |
| dysbacteriosis_herpes_act | 0.520 | 0.753 | 0.626 |
| ec | 0.177 | 0.309 | 0.190 |
| Serous effusion-Diseased cells | 0.232 | 0.336 | 0.273 |
| Serous effusion-Breast cancer | 0.515 | 0.710 | 0.599 |
| Thyroid gland-Papillary cancer | 0.534 | 0.758 | 0.618 |
| Thyroid gland-Suspicious for Malignancy | 0.368 | 0.530 | 0.451 |
| Urine-SHGUC | 0.258 | 0.350 | 0.327 |
| Urine-AUC | 0.120 | 0.159 | 0.150 |
| respiratory tract-Diseased cells | 0.147 | 0.203 | 0.173 |
| respiratory tract-adenocarcinoma | 0.009 | 0.012 | 0.012 |

#### exp6 (中文 + 强增强)

| 类别 | mAP | mAP_50 | mAP_75 |
|------|-----|--------|--------|
| ascus | 0.316 | 0.541 | 0.341 |
| asch | 0.209 | 0.395 | 0.202 |
| lsil | 0.301 | 0.535 | 0.318 |
| agc_adenocarcinoma_em | 0.434 | 0.696 | 0.481 |
| vaginalis | 0.216 | 0.473 | 0.143 |
| dysbacteriosis_herpes_act | 0.523 | 0.744 | 0.636 |
| ec | 0.187 | 0.335 | 0.194 |
| Serous effusion-Diseased cells | 0.261 | 0.376 | 0.306 |
| Serous effusion-Breast cancer | 0.515 | 0.674 | 0.595 |
| Thyroid gland-Papillary cancer | 0.556 | 0.784 | 0.641 |
| Thyroid gland-Suspicious for Malignancy | 0.516 | 0.715 | 0.671 |
| Urine-SHGUC | 0.254 | 0.354 | 0.310 |
| Urine-AUC | 0.046 | 0.067 | 0.058 |
| respiratory tract-Diseased cells | 0.169 | 0.233 | 0.197 |
| respiratory tract-adenocarcinoma | 0.025 | 0.031 | 0.030 |

#### exp7 (英文 + 强增强)

| 类别 | mAP | mAP_50 | mAP_75 |
|------|-----|--------|--------|
| ascus | 0.321 | 0.555 | 0.341 |
| asch | 0.210 | 0.395 | 0.206 |
| lsil | 0.315 | 0.548 | 0.340 |
| agc_adenocarcinoma_em | 0.436 | 0.696 | 0.482 |
| vaginalis | 0.219 | 0.477 | 0.144 |
| dysbacteriosis_herpes_act | 0.526 | 0.762 | 0.626 |
| ec | 0.194 | 0.343 | 0.207 |
| Serous effusion-Diseased cells | 0.414 | 0.599 | 0.484 |
| Serous effusion-Breast cancer | 0.537 | 0.716 | 0.608 |
| Thyroid gland-Papillary cancer | 0.541 | 0.761 | 0.625 |
| Thyroid gland-Suspicious for Malignancy | 0.360 | 0.501 | 0.425 |
| Urine-SHGUC | 0.311 | 0.428 | 0.396 |
| Urine-AUC | 0.088 | 0.122 | 0.103 |
| respiratory tract-Diseased cells | 0.224 | 0.310 | 0.263 |
| respiratory tract-adenocarcinoma | 0.014 | 0.022 | 0.017 |

### 1.3 Base 类别对比（按类别）

| 类别 | exp5 | exp6 | exp7 | 最优 |
|------|------|------|------|------|
| ascus | 0.304 | 0.316 | **0.321** | exp7 |
| asch | 0.202 | 0.209 | **0.210** | exp7 |
| lsil | 0.292 | 0.301 | **0.315** | exp7 |
| agc_adenocarcinoma_em | 0.348 | 0.434 | **0.436** | exp7 |
| vaginalis | **0.232** | 0.216 | 0.219 | exp5 |
| dysbacteriosis_herpes_act | 0.520 | 0.523 | **0.526** | exp7 |
| ec | 0.177 | 0.187 | **0.194** | exp7 |
| Serous effusion-Diseased cells | 0.232 | 0.261 | **0.414** | exp7 |
| Serous effusion-Breast cancer | 0.515 | 0.515 | **0.537** | exp7 |
| Thyroid gland-Papillary cancer | 0.534 | **0.556** | 0.541 | exp6 |
| Thyroid gland-Suspicious for Malignancy | 0.368 | **0.516** | 0.360 | exp6 |
| Urine-SHGUC | 0.258 | 0.254 | **0.311** | exp7 |
| Urine-AUC | **0.120** | 0.046 | 0.088 | exp5 |
| respiratory tract-Diseased cells | 0.147 | 0.169 | **0.224** | exp7 |
| respiratory tract-adenocarcinoma | 0.009 | **0.025** | 0.014 | exp6 |

---

## 二、Novel 类别评估结果（11类新类别）

### 2.1 总体 mAP

| 实验 | mAP | mAP_50 | mAP_75 |
|------|-----|--------|--------|
| exp5 | 0.073 | 0.097 | 0.081 |
| exp6 | 0.071 | 0.099 | 0.076 |
| **exp7** | **0.086** | **0.111** | **0.096** |

### 2.2 每个类别的 AP 值

| 类别 | exp5 | exp6 | exp7 | 最优 |
|------|------|------|------|------|
| hsil_scc_omn | 0.0493 | **0.0878** | 0.0413 | exp6 |
| monilia | 0.0000 | 0.0000 | 0.0000 | - |
| Serous effusion-Ovarian cancer | 0.0704 | 0.1124 | **0.1995** | exp7 |
| Serous effusion-adenocarcinoma | 0.0005 | **0.0624** | 0.0199 | exp6 |
| Thyroid gland-Suspicious papillary cancer | 0.0404 | **0.0908** | 0.0496 | exp6 |
| Thyroid gland-AUC | 0.0001 | **0.0017** | 0.0000 | exp6 |
| Thyroid gland-Malignant tumour | 0.0111 | 0.0008 | **0.0112** | exp7 |
| Thyroid gland-NS | 0.0000 | 0.0000 | 0.0000 | - |
| Urine-HGUC | 0.5654 | 0.2551 | **0.5947** | exp7 |
| respiratory tract-Squamous cell carcinoma | **0.0044** | 0.0030 | 0.0024 | exp5 |
| respiratory tract-Small cell carcinoma | 0.0584 | **0.1620** | 0.0263 | exp6 |

### 2.3 Novel AP50 值

| 类别 | exp5 | exp6 | exp7 |
|------|------|------|------|
| hsil_scc_omn | 0.1003 | 0.1823 | 0.0837 |
| monilia | 0.0000 | 0.0000 | 0.0000 |
| Serous effusion-Ovarian cancer | 0.0753 | 0.1188 | 0.2120 |
| Serous effusion-adenocarcinoma | 0.0005 | 0.0666 | 0.0205 |
| Thyroid gland-Suspicious papillary cancer | 0.0686 | 0.1658 | 0.0860 |
| Thyroid gland-AUC | 0.0001 | 0.0026 | 0.0001 |
| Thyroid gland-Malignant tumour | 0.0113 | 0.0010 | 0.0140 |
| Thyroid gland-NS | 0.0000 | 0.0000 | 0.0000 |
| Urine-HGUC | 0.7245 | 0.3123 | 0.7593 |
| respiratory tract-Squamous cell carcinoma | 0.0083 | 0.0062 | 0.0046 |
| respiratory tract-Small cell carcinoma | 0.0795 | 0.2296 | 0.0356 |

### 2.4 Novel 类别样本数

| 类别 | 样本数 |
|------|--------|
| hsil_scc_omn | 1942 |
| monilia | 459 |
| Serous effusion-Ovarian cancer | 19 |
| Serous effusion-adenocarcinoma | 4 |
| Thyroid gland-Suspicious papillary cancer | 2529 |
| Thyroid gland-AUC | 158 |
| Thyroid gland-Malignant tumour | 31 |
| Thyroid gland-NS | 32 |
| Urine-HGUC | 44 |
| respiratory tract-Squamous cell carcinoma | 27 |
| respiratory tract-Small cell carcinoma | 396 |

---

## 三、结论

### 3.1 Base 类别

- **总体最优**: exp7 (英文 + 强增强)，mAP = **0.314**
- exp7 在 15 个 Base 类别中有 **11 个类别取得最优**
- exp6 在甲状腺相关类别表现较好（Papillary cancer, Suspicious for Malignancy）
- exp5 在 vaginalis 和 Urine-AUC 上表现最优

### 3.2 Novel 类别

- **总体最优**: exp7 (英文 + 强增强)，mAP = **0.086**
- 但从单类别来看，**中文文本 (exp6)** 在多数 Novel 类别上表现更好，尤其是：
  - hsil_scc_omn: 0.0878
  - Thyroid gland-Suspicious papillary cancer: 0.0908
  - respiratory tract-Small cell carcinoma: 0.1620

- **英文文本 (exp7)** 在以下类别表现最优：
  - Serous effusion-Ovarian cancer: 0.1995
  - Urine-HGUC: **0.5947** (所有实验中最高)

- **零检测类别**（所有实验均为 0）：
  - monilia
  - Thyroid gland-NS

### 3.3 综合建议

| 场景 | 推荐实验 | 原因 |
|------|---------|------|
| Base 类别检测 | exp7 | 总体 mAP 最高 (0.314) |
| Novel 类别泛化 | exp6 | 多数新类别表现更好 |
| Urine-HGUC 检测 | exp7 | AP 达到 0.5947 |

---

## 四、评估命令

```bash
# Base mAP 评估（排除 negative）
PYTHONPATH=. python test_exclude_negative.py \
  --config config/wedetect_tiny_tct_exp{5,6,7}.py \
  --checkpoint work_dirs/wedetect_tiny_tct_exp{5,6,7}/best_*.pth \
  --work-dir work_dirs/eval/test_base_exclude_neg_exp{5,6,7}

# Novel mAP 评估
PYTHONPATH=. python eval_novel.py \
  --config config/wedetect_tiny_tct_exp{5,6,7}.py \
  --checkpoint work_dirs/wedetect_tiny_tct_exp{5,6,7}/best_*.pth \
  --text data/texts/tct_ngc_v2_class_texts{_en,}.json \
  --out-dir work_dirs/eval/test_novel_exp{5,6,7}
```

---

*评估日期: 2026-01-27*
