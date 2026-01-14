# WeDetect TCT_NGC 实验结果分析报告

## 1. 实验概览

### 数据集
- **Base类 (训练)**: 20类
- **Novel类 (零样本)**: 11类
- **总类别**: 31类

### 模型配置
- backbone: ConvNext-Tiny + XLM-Roberta
- 输入分辨率: 640x640
- 预训练权重: wedetect_tiny.pth

---

## 2. 实验结果对比

### 2.1 整体性能

| 实验 | 配置 | 最佳Epoch | mAP | mAP_50 | 状态 |
|------|------|-----------|-----|--------|------|
| 基准 | lr=2e-4, epochs=15 | 9 | **0.254** | **0.390** | 完成 |
| 实验1 | lr=1e-4, epochs=20 | 9 | 0.245 | 0.374 | 完成 |
| 实验2 | 冻结backbone前两层 | - | - | - | 待运行 |
| 实验3 | 调整loss权重 | - | - | - | 待运行 |

### 2.2 Base类详细结果 (排除负样本)

排除5个负样本类后的评估结果 (mAP: 0.321, mAP_50: 0.487):

| 类别 | mAP | mAP_50 | 表现 | 分析 |
|------|-----|--------|------|------|
| Thyroid gland-Suspicious for Malignancy | 0.621 | 0.902 | 极好 | 特征明显，样本充足 |
| Serous effusion-Breast cancer | 0.557 | 0.742 | 极好 | 乳腺癌特征明显 |
| dysbacteriosis_herpes_act | 0.538 | 0.764 | 极好 | 细菌/疱疹特征清晰 |
| Thyroid gland-Papillary cancer | 0.506 | 0.700 | 好 | 甲状腺乳头状癌特征 |
| Serous effusion-Diseased cells | 0.404 | 0.603 | 好 | 病变细胞特征 |
| Urine-SHGUC | 0.334 | 0.455 | 中等 | 尿液样本 |
| ascus | 0.315 | 0.533 | 中等 | 非典型鳞状细胞 |
| lsil | 0.298 | 0.537 | 中等 | 低级别鳞状上皮内病变 |
| agc_adenocarcinoma_em | 0.282 | 0.431 | 中等 | 腺癌 |
| respiratory tract-Diseased cells | 0.273 | 0.378 | 中等 | 呼吸道病变 |
| vaginalis | 0.220 | 0.463 | 较差 | 阴道炎 |
| asch | 0.197 | 0.376 | 较差 | 非典型鳞状细胞,不能排除HSIL |
| ec | 0.168 | 0.298 | 较差 | 颈管细胞 |
| Urine-AUC | 0.081 | 0.109 | 差 | 样本特征不明显 |
| respiratory tract-adenocarcinoma | 0.015 | 0.019 | 极差 | 样本少，特征复杂 |

### 2.3 Novel类零样本结果 (mAP: 0.068, mAP_50: 0.092)

| 类别 | mAP | mAP_50 | 样本数 | 分析 |
|------|-----|--------|--------|------|
| Urine-HGUC | 0.570 | 0.721 | 44 | 极好！与Base类Urine相似 |
| hsil_scc_omn | 0.074 | 0.146 | 1942 | 中等，高级别病变 |
| Thyroid gland-Malignant tumour | 0.043 | 0.051 | 31 | 较差 |
| Thyroid gland-Suspicious papillary | 0.040 | 0.066 | 2529 | 较差 |
| respiratory tract-Squamous carcinoma | 0.017 | 0.018 | 27 | 差 |
| Serous effusion-Ovarian cancer | 0.007 | 0.008 | 19 | 差 |
| Serous effusion-adenocarcinoma | 0.000 | 0.000 | 4 | 失败，样本太少 |
| monilia | 0.000 | 0.000 | 459 | 失败 |
| Thyroid gland-AUC | 0.000 | 0.000 | 158 | 失败 |
| Thyroid gland-NS | 0.000 | 0.000 | 32 | 失败 |
| respiratory tract-Small cell carcinoma | 0.000 | 0.000 | 396 | 失败 |

---

## 3. 问题分析

### 3.1 过拟合问题
- 两次实验最佳epoch都在9左右
- epoch 9后mAP持续下降
- **解决方案**: 实验2冻结backbone，实验3调整loss权重

### 3.2 低表现类别分析

#### Base类低表现 (mAP < 0.2):
1. **respiratory tract-adenocarcinoma** (0.015): 样本少，与其他呼吸道类混淆
2. **Urine-AUC** (0.081): 特征不明显
3. **ec** (0.168): 颈管细胞特征复杂
4. **asch** (0.197): 与ascus混淆

#### Novel类失败 (mAP = 0):
1. **monilia** (459样本): 真菌特征与训练集不相似
2. **Thyroid gland-AUC/NS**: 文本描述太简单
3. **respiratory tract-Small cell carcinoma**: 小细胞癌特征独特

### 3.3 文本描述问题
当前文本过于简单，例如:
- `["thyroid auc"]` 应改为 `["thyroid atypia undetermined significance", "thyroid uncertain cells"]`
- `["monilia"]` 应改为 `["monilia fungus", "candida infection", "yeast cells"]`

---

## 4. 改进建议

### 4.1 短期改进 (实验2-3)
- [x] 配置文件已创建
- [ ] 运行实验2: 冻结backbone前两层
- [ ] 运行实验3: loss_cls_weight=1.0, loss_bbox_weight=5.0

### 4.2 文本描述改进
建议创建更丰富的文本描述:

```json
// 改进后的 Novel类文本描述建议
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

### 4.3 数据层面
- 考虑样本重采样平衡类别
- 增加数据增强 (旋转、颜色抖动)

---

## 5. 实验运行命令

```bash
# 实验2: 冻结backbone
CUDA_VISIBLE_DEVICES=0 python tools/train.py config/wedetect_tiny_tct_exp2.py

# 实验3: 调整loss权重
CUDA_VISIBLE_DEVICES=1 python tools/train.py config/wedetect_tiny_tct_exp3.py

# 评估Base类 (排除负样本)
python test_exclude_negative.py --checkpoint work_dirs/wedetect_tiny_tct_exp2/best_*.pth

# 评估Novel类
python eval_novel_manual.py
```

---

## 6. 总结

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| Base mAP (排除负样本) | 0.321 | > 0.35 |
| Novel mAP (零样本) | 0.068 | > 0.10 |
| 最佳Epoch | 9 | 减少过拟合 |

**下一步**: 运行实验2和实验3，改进文本描述
