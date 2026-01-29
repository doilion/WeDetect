# TCT_NGC V3 数据集零样本目标检测实验总结

实验日期：2025年12月 - 2026年1月

本文档汇总在 TCT_NGC V3 数据集上的零样本开放词汇检测实验结果，覆盖 GLIP、Grounding DINO、YOLO-World 与 WeDetect 四类模型，并对文本编码器、文本表示和训练策略的影响进行对比分析。

---

## 1. 实验背景与目标

本实验旨在评估开放词汇目标检测模型在医学细胞病理学图像上的零样本泛化能力。我们在 TCT_NGC V3 数据集上对比主流模型，探索不同文本编码器、文本表示方法、训练策略和损失函数配置对零样本检测性能的影响。

---

## 2. 数据集介绍

### 2.1 TCT_NGC V3 数据集

TCT_NGC V3 采用可检测性驱动的类别级别划分策略，将检测难度较高的类别移至 Novel 集合，用于测试模型的零样本泛化能力。该划分方式支持域内语义迁移评估。

### 2.2 数据集统计

| 数据集 | 图像数 | 标注数 | 类别数 |
|--------|--------|--------|--------|
| 训练集 (Base) | 69,590 | 338,721 | 20 |
| 测试集 (Base) | 12,850 | 68,506 | 15* |
| 测试集 (Novel) | 2,201 | 5,641 | 11 |

*注：Base 测试集排除 5 个 Negative 类别后为 15 个类别参与评估。

---

## 3. 实验方法

### 3.1 GLIP

GLIP (Grounded Language-Image Pre-training) 使用 Swin-T 作为视觉骨干，并基于语言-图像对比学习实现开放词汇检测。对比单类名与展开类名两种文本格式，单类名版本在 Novel 上达到 13.8% mAP，表现最佳；展开类名则下降至 6.8%。

### 3.2 Grounding DINO

Grounding DINO 采用部分冻结的微调策略，冻结视觉骨干 (Swin-T) 与文本编码器 (BERT)，仅训练检测头与其他模块。

### 3.3 YOLO-World

YOLO-World 探索多种配置组合：
- 文本编码器：OpenAI CLIP vs BiomedCLIP
- 文本表示：单类名 vs 展开类名 vs 层次化 4 属性描述
- 训练策略：完全冻结 vs 0.1x 微调文本编码器
- 辅助损失：医学排序损失 (MedicalRankingLoss)

展开类名将医学缩写展开为完整描述（如 hsil_scc_omn → cervical high-grade squamous intraepithelial lesion and squamous cell carcinoma）。

### 3.4 WeDetect

WeDetect 采用多模态 YOLO 架构，结合 ConvNext-Tiny 视觉骨干与 XLM-RoBERTa 文本编码器。实验探索五种策略：
1. 基准配置 (lr=2e-4)
2. 降低学习率 (lr=1e-4)
3. 冻结视觉骨干前两层
4. 调整 Loss 权重 (cls=1.0, bbox=5.0)
5. 组合策略 (冻结 BB + Loss 调整)

---

## 4. 实验结果对比

### 4.1 总体性能对比（按模型分组）

| 方法 | 文本编码器 | 文本格式 | Base mAP | Novel mAP | Epoch | 骨干网络 |
|------|------------|----------|----------|-----------|-------|----------|
| GLIP | BERT | 单类名 | 31.7% | **13.8%** | 24 | Swin-T |
| GLIP | BERT | 展开类名 | 31.6% | 6.8% | 24 | Swin-T |
| Grounding DINO | BERT (冻结) | 单类名 | 32.2% | 10.3% | 18 | Swin-T |
| YOLO-World | CLIP (冻结) | 单类名 | **34.2%** | 3.7% | 56 | YOLOv8-L |
| YOLO-World | CLIP (冻结) | 展开类名 | 33.4% | 6.9% | 47 | YOLOv8-L |
| YOLO-World | BiomedCLIP (冻结) | 单类名 | 33.0% | 2.2% | 65 | YOLOv8-L |
| YOLO-World | BiomedCLIP (冻结) | 展开类名 | 31.1% | 6.0% | 37 | YOLOv8-L |
| YOLO-World | BiomedCLIP (冻结) | 层次化 4 属性 | 28.3% | 7.2% | 48 | YOLOv8-L |
| YOLO-World | BiomedCLIP (0.1x) | 层次化 4 属性 | 28.2% | 5.6% | 16 | YOLOv8-L |
| YOLO-World | CLIP (冻结)+融合层 | 层次化 4 属性 | 26.9% | 3.3% | 31 | YOLOv8-L |
| YOLO-World | CLIP (冻结)+融合层 | 层次化+排序损失 | 26.4% | 2.0% | 29 | YOLOv8-L |
| YOLO-World | BiomedCLIP (冻结) | 层次化+排序损失 | 27.3% | 4.6% | 44 | YOLOv8-L |
| WeDetect | XLM-RoBERTa (0.01x) | 单类名 (基准) | 32.1% | 6.8% | 9 | ConvNext-T |
| WeDetect | XLM-RoBERTa (0.01x) | 单类名 (lr↓) | 30.7% | 7.2% | 9 | ConvNext-T |
| WeDetect | XLM-RoBERTa (冻结) | 单类名 (冻结BB) | 28.4% | 8.1% | 8 | ConvNext-T |
| WeDetect | XLM-RoBERTa (0.01x) | 单类名 (Loss调整) | 29.6% | **9.1%** | 9 | ConvNext-T |
| WeDetect | XLM-RoBERTa (冻结) | 冻结BB+Loss调整 | 24.3% | 9.0% | 7 | ConvNext-T |

图例（左侧颜色条）：GLIP (蓝) / Grounding DINO (橙) / YOLO-World (绿) / WeDetect (棕)。浅蓝高亮 = Base 最佳，浅绿高亮 = Novel 最佳。

### 4.2 关键 Novel 类别零样本性能对比

| Novel 类别 | GLIP | G-DINO | YW-CLIP | YW-Expand | BM-Expand | YW-Hier | YW-Rank | WD-基准 | WD-Loss | WD-组合 |
|------------|------|--------|---------|-----------|-----------|---------|---------|--------|---------|--------|
| Urine-HGUC | 59.7% | 65.6% | 26.4% | 49.1% | 58.8% | 46.3% | 28.1% | 57.0% | **68.2%** | 64.9% |
| Thyroid-Malignant | 53.8% | 0.6% | 0.5% | 1.4% | 2.8% | 0.5% | 1.0% | 4.3% | 0.0% | 3.4% |
| respiratory-Small cell | 32.1% | 1.1% | 0.7% | 1.6% | 0.1% | 0.5% | 5.5% | 0.0% | 9.0% | 7.9% |
| Serous-Ovarian cancer | - | - | - | 3.0% | 0.4% | - | 4.0% | 0.7% | 9.9% | 7.3% |
| Serous-adenocarcinoma | 0.1% | 24.0% | 0.0% | 5.6% | 0.3% | 13.1% | 0.9% | 0.0% | 0.7% | 0.9% |
| hsil_scc_omn | 0.1% | 10.0% | 3.4% | 10.8% | 2.5% | 6.3% | 9.6% | 7.4% | 6.6% | 6.5% |
| Thyroid-Susp. papillary | 5.5% | 8.1% | 7.3% | 4.5% | 1.3% | 1.4% | 1.1% | 4.0% | 4.6% | 7.0% |

注：GLIP=单类名版本，YW-CLIP=YOLO-World(CLIP 单类名)，YW-Expand=YOLO-World(CLIP 展开类名)，BM-Expand=YOLO-World(BiomedCLIP 展开类名)，YW-Hier=YOLO-World(BiomedCLIP 层次化)，YW-Rank=YOLO-World(BiomedCLIP 层次化+排序损失)，WD-基准=WeDetect(lr=2e-4)，WD-Loss=WeDetect(调整 Loss 权重)，WD-组合=WeDetect(冻结BB+Loss调整)。

---

## 5. 分析与讨论

### 5.1 Base vs Novel 性能权衡

- YOLO-World + CLIP 在 Base 上表现最佳 (34.2%)，但 Novel 性能较弱 (3.7%)
- GLIP 在 Novel 上表现最佳 (13.8%)，同时保持竞争力的 Base 性能 (31.7%)
- WeDetect 通过调整 Loss 权重可提升 Novel 至 9.1%，但牺牲 Base 性能 (-2.5%)

### 5.2 语义迁移成功案例

Urine-HGUC 在所有方法中均表现突出（最高 68.2%）：
- Base 中包含语义相似类别 Urine-SHGUC
- HGUC 与 SHGUC 形态学高度相似
- 说明域内语义知识可有效迁移至未见类别

### 5.3 YOLO-World 系列实验分析

**文本编码器对比**
- 单类名设置下：CLIP (Novel 3.7%) 优于 BiomedCLIP (2.2%)
- 层次化设置下：BiomedCLIP (7.2%) 显著优于 CLIP (3.3%)

**层次化文本表示**
- BiomedCLIP 层次化 (冻结) 取得 Novel 最佳 7.2%，较单类名提升 5.0%
- 训练文本编码器 (0.1x) 早期过拟合，Novel 降至 5.6%

**排序损失实验**
- CLIP/ BiomedCLIP 的排序损失均未提升泛化，Novel 出现下降
- 结论：排序损失约束过强，可能与检测目标不一致

**展开类名实验**
- GLIP 展开类名：Novel 下降 7.0%
- CLIP/BiomedCLIP 展开类名：Novel 提升 3.2%/3.8%
- 结论：展开类名效果与编码器强相关

### 5.4 WeDetect 系列实验分析

**降低学习率 (lr=1e-4)**
- Base: 32.1% → 30.7% (-1.4%)，Novel: 6.8% → 7.2% (+0.4%)
- Serous-Ovarian cancer 从 0.7% 提升至 9.7%

**冻结 Backbone 前两层**
- Base: 32.1% → 28.4% (-3.7%)，Novel: 6.8% → 8.1% (+1.3%)
- 最佳 Epoch 提前到 8，过拟合减缓

**调整 Loss 权重 (cls=1.0, bbox=5.0)**
- Novel 最高：6.8% → 9.1% (+2.3%)
- Urine-HGUC 达到 68.2% AP，为所有实验最高

**组合策略 (冻结BB+Loss调整)**
- Base: 32.1% → 24.3% (-7.8%)，Novel: 6.8% → 9.0% (+2.2%)
- 组合产生负面叠加效应，Base 下降超过两者之和

### 5.5 模型特点对比

| 特点 | GLIP | Grounding DINO | YOLO-World | WeDetect |
|------|------|----------------|------------|---------|
| Base 性能 | 良好 (31.7%) | 良好 (32.2%) | 最佳 (34.2%) | 良好 (29.6-32.1%) |
| Novel 性能 | 最佳 (13.8%) | 良好 (10.3%) | 较弱 (2.0-7.2%) | 中等 (6.8-9.1%) |
| 训练稳定性 | 稳定 | 稳定 | 需调参 | 稳定 |
| 推理速度 | 较慢 | 中等 | 快速 | 快速 |
| 收敛速度 | 中等 (24 ep) | 中等 (18 ep) | 较慢 (16-65 ep) | 快速 (8-9 ep) |

---

## 6. 主要结论

1. Novel 零样本性能排名：GLIP (13.8%) > Grounding DINO (10.3%) > WeDetect-Loss (9.1%) > WeDetect-冻结BB (8.1%) > YOLO-World-BiomedCLIP 层次化 (7.2%)
2. Base 性能排名：YOLO-World-CLIP (34.2%) > YOLO-World-BiomedCLIP (33.0%) > Grounding DINO (32.2%) > WeDetect-基准 (32.1%) > GLIP (31.7%)
3. 语义迁移最佳案例：Urine-HGUC 在 WeDetect-Loss 上达到 68.2%，在 Grounding DINO 上达到 65.6%
4. 层次化文本表示：牺牲 Base 性能但提升 Novel 泛化，BiomedCLIP 层次化比单类名 Novel 提升 5.0%
5. 文本编码器选择：层次化设置下 BiomedCLIP 优于 CLIP，医学预训练知识更重要
6. WeDetect Loss 调整最有效：增加分类损失权重提升 Novel 性能最多 (+2.3%)
7. 排序损失效果不佳：未能提升泛化能力，Novel 下降至 2.0%
8. 收敛速度：WeDetect 最快 (8-9 epochs)，YOLO-World 最慢 (16-65 epochs)

---

## 7. 后续工作

1. 引入更多对比方法或更强的文本编码器，验证医学领域语义理解不足的问题
2. 针对低性能 Novel 类补充更细粒度文本描述与医学术语扩展
3. 进一步探索医学领域预训练语言模型或领域适配策略

---

*文档创建时间: 2026-01-27*
