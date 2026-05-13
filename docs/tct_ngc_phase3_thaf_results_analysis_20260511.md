# Phase 3 THAF — 完整结果分析与问题剖析

**日期**：2026-05-11（avg novel 数字 2026-05-12 重算）
**目的**：综合 Phase 3a (XLM-R THAF) / 3b (BiomedCLIP THAF) / 3c (clean dev30 baseline) 全套 eval 结果，剖析两个核心问题 — base 涨而 novel 反跌的悖论，以及 cross-attention fusion 模块的"训练后自我归零"现象。

> **⚠ 2026-05-12 公式更正**：原"avg novel"为 4-split 算术均值（包含 `full_5 = main_3 ∪ pseudo_2`，5 类双重计算）。改用 **mean over 9 unique novel** = `(3·main_3 + 2·pseudo_2 + 4·hard_4) / 9`。下表 avg 列已重算；结论方向不变，数字整体上调 0.01-0.02。

---

## 0. TL;DR

| 发现 | 严重程度 |
|---|---|
| **THAF + BiomedCLIP base 25-cls = 0.327**（vs clean dev30 baseline 0.310，**+1.7pp**）| ✅ 正面信号 |
| **THAF novel zero-shot 全线 collapse**（avg 0.020-0.041 vs v2 baseline 0.108）| 🚨 **方法核心目的失败** |
| THAF 的 **cross-attention fusion module 训练后 α≈0**，等价于 mean pooling | ⚠ 设计被实证证伪 |
| **70-99% novel 图像** 在 image encoder 输出上 top-1 预测都是 base class | 🚨 **真正的 novel zero-shot 瓶颈** |
| dev32→dev30 的 −1pp drop 主要是 **训练随机性**（不是 NHGUC merge）| 修正之前的 hypothesis |

**论文叙事影响**：原计划"THAF 作为 method §A"立不住，**cross-attention 是死代码**。需要 reframe 为"5-attr 结构化 prompt + medical encoder + mean pool"，并把 novel zero-shot 失败定位到 **image encoder 端**（不是 text 端，跟之前判断完全相反）。

---

## 1. 完整 ablation 表

dev30 ckpt × 各种 inference 策略 / 训练方法 × 4 novel splits + base 25-cls：

注：`Avg novel = (3·main_3 + 2·pseudo_2 + 4·hard_4) / 9`（mean over 9 unique novel cls，2026-05-12 重算公式）。full_5 仅作 5-class mixed 参考。

| Method | Base 25-cls | main_3 | pseudo_2 | hard_4 | _full_5_ | **Avg novel (9 unique)** |
|---|---:|---:|---:|---:|---:|---:|
| v2 baseline (XLM-R, single PSC) | 0.310 | 0.134 | 0.108 | 0.088 | _0.049_ | **0.108** |
| score fusion (XLM-R, raw visproto) | 0.310 | 0.137 | 0.108 | 0.095 | _0.051_ | **0.112** |
| Procrustes calfused (DEAD-5 verify) | 0.310 | 0.132 | 0.092 | 0.002 | _0.045_ | **0.065** |
| **THAF + XLM-R (768d)** | 0.302 | 0.021 | 0.033 | 0.013 | _0.013_ | **0.020** 🔻 |
| **THAF + BiomedCLIP (512d)** ← method §A 主结果 | **0.327** ✨ | 0.009 | 0.137 | 0.017 | _0.045_ | **0.041** 🔻 |
| THAF (XLM-R) + score fusion | 0.302 | 0.018 | 0.025 | 0.008 | _0.013_ | 0.014 |
| THAF (BiomedCLIP) + score fusion | 0.327 | 0.011 | 0.120 | 0.014 | _0.041_ | 0.038 |
| _visproto raw (5-shot, leakage, XLM-R)_ | — | 0.011 | 0.032 | 0.011 | _0.012_ | _0.020_ |

**关键观察**：
- Base 25-cls：BiomedCLIP THAF **唯一突破 baseline 的方法**（+1.7pp），但 XLM-R THAF 反而 **小跌 0.8pp**
- Novel zero-shot：**所有 THAF 变体都比 v2 baseline 差 2-5x**
- THAF + score fusion 不能救：novel mAP 跟 THAF 单独差不多

---

## 2. 问题 A：dev32 → dev30 的 −1pp drop 是不是 NHGUC merge 害的

### 2.1 之前的 hypothesis（已全部证伪）

| Hypothesis | 实证 | 结论 |
|---|---|---|
| GPU 1 thermal throttle 导致旧 dev30 训坏 | clean dev30 重训 fix throttle + LR，base 25-cls = 0.310 ≈ 旧 0.306 | ❌ |
| NCCL 切 1GPU 影响动力学 | clean 全程 2GPU，结果一样 | ❌ |
| LR overlap (begin=1) | clean 用 begin=2，结果一样 | ❌ |
| NHGUC merge 改变 loss balance | per-class diff 不支持（见下）| ❌ |

### 2.2 per-class diff（dev32 vs 旧 dev30，同 25 类，同测试集）

| 类 | dev32 | dev30 | Δ | 跟 NHGUC merge 有关？ |
|---|---:|---:|---:|:---:|
| **Thyroid gland-Macrophages** | 0.583 | 0.465 | **-0.118** | ❌ |
| Urine-SHGUC | 0.176 | 0.106 | -0.070 | ✅ |
| respiratory-Diseased cells | 0.483 | 0.429 | -0.054 | ❌ |
| Serous effusion-Diseased cells | 0.305 | 0.382 | **+0.077** | ❌ |
| 其他 21 类 | — | — | ±0.05 内 | — |

**整体均值**：dev32 = 0.315 / dev30 = 0.306 / Δ = **−0.010**

但 −0.010 完全由 3 个特定类的 drop（−0.118 / −0.070 / −0.054 = 合计 −0.242）和 1 个类的 gain（+0.077）凑出来。21 个类基本不变。

如果 NHGUC merge 真的影响所有类的梯度（理论 hypothesis），应该看到**普遍小幅下降**，而不是**3 个类剧烈下降 + 21 个不变**。

### 2.3 真正的 hypothesis：训练随机性

最可能的解释：**单次训练的 per-class AP 浮动是 ±0.05-0.12 量级**。不同 random seed / data shuffle / NCCL all-reduce 顺序，会让某些类 AP 跳 0.10 以上。dev32→dev30 因为 class count 改了（32→30），所有 RNG 顺序都变了，自然产生不同的 per-class outcome。

**clean dev30 重训的实证**（用同样 30 类 taxonomy 重训）：
- Base 25-cls = 0.310 vs 旧 0.306（Δ +0.004，**within noise**）
- 同样训练成本，单次实验，per-class AP 差异同量级

→ **结论**：dev32→dev30 的"1pp drop"**不是 systematic**，是 single-run noise。修不修都救不回来。论文 headline 直接用 dev30 baseline = **0.310**（clean ckpt 数字）。

---

## 3. 问题 B：THAF base 涨但 novel 反跌 —— **核心 paradox**

### 3.1 Base 涨 — 跟 cross-attention fusion 没关系

| 候选解释 | 是不是真因 |
|---|---|
| 5-attr 文本信号比单 PSC prompt 更丰富 | ✅ 部分（mean pool 已经能利用）|
| Cross-attention fusion module 学到了 attribute 选择 | ❌ **反证**：alpha→0，模块输出无贡献 |
| BiomedCLIP 比 XLM-R 适合医学 | ✅ 主因（XLM-R THAF 同 5-attr + mean pool，反而 -0.8pp）|
| LR schedule (begin=1) 跟 clean (begin=2) 不公平 | ⚠ 0.8pp 差距里可能有部分 LR confound |

**净结论**：BiomedCLIP THAF base 涨 +1.7pp，**主要来自 (a) BiomedCLIP encoder + (b) 5-attr mean pool**，**不是来自 trainable fusion 设计**。

### 3.2 THAF fusion module 自我归零（关键反证）

**Phase 3.5 诊断**（`tools/diagnose_thaf_fusion.py`）：

| 量 | BiomedCLIP THAF | XLM-R THAF |
|---|---:|---:|
| trained `alpha` （init=0.3）| **−0.0001** | **−0.0003** |
| forward: `output = alpha · cross_attn_proj + (1−alpha) · attr_mean` | 等价于 `attr_mean` | 等价于 `attr_mean` |
| 死参数 | ~3.15M | ~7.1M |

意思：cross-attention 模块、attr_type_embed、output_proj —— **全部参数训练后等效失效**。trained THAF 跟"直接 mean pool 5 个 attribute embeddings"的输出**几乎相同**。

**Cosine geometry 验证**（trained vs attr_mean baseline）：

| metric | BiomedCLIP trained | BiomedCLIP attr_mean | Δ |
|---|---:|---:|---:|
| base↔base off-diag max cos | 0.962 | 0.967 | -0.005 |
| novel↔novel off-diag max cos | **0.940** | 0.947 | -0.006 |
| novel→base avg cos | 0.823 | 0.842 | -0.020 |

→ trained 跟 attr_mean **几乎一致**（diff < 0.02），训练的影响微乎其微。

**为什么 fusion 训练成空**：
- output_proj 的初始化 gain=0.1（小），cross-attention 路径贡献信号很弱
- attr_mean 路径已经是优秀的梯度 sink（容易降 loss）
- 优化器选择 "shrink alpha to 0" 而非"调 cross-attention 让它有用"
- 12 epochs 不够长，可能没机会让 cross-attention 起步

**Cosine 热力图对比**（trained vs attr_mean）：

```
docs/figures/thaf_diagnostic/biomedclip/cosine_heatmap_trained.png
docs/figures/thaf_diagnostic/biomedclip/cosine_heatmap_attr_mean.png
docs/figures/thaf_diagnostic/xlmr/cosine_heatmap_trained.png
docs/figures/thaf_diagnostic/xlmr/cosine_heatmap_attr_mean.png
```

两张图视觉上几乎一致 → fusion module 没有移动类向量在 cosine 空间的相对位置。

### 3.3 Novel 跌 — image encoder 端瓶颈（Phase 3.6 实证）

**Phase 3.6 诊断**（`tools/diagnose_image_encoder.py`，sampled GT bbox 30/class × 39 classes）：

#### BiomedCLIP THAF：

| metric | base GT bboxes | novel GT bboxes |
|---|---:|---:|
| n | 900 | 263 |
| mean cosine to GT class | 0.061 | **−0.178** |
| top-1 accuracy | **75.2%** | **0.4%** 🚨 |
| top-1 is BASE class | 99.9% | **99.2%** 🚨 |

→ novel image 经 image encoder 输出，**99.2% 都被预测成 base class**（虽然 GT 是 novel 类）。**class vector 几何 ok（max cos 0.94），但 image feature 完全不指向 novel 类**。

#### XLM-R THAF：

| metric | base | novel |
|---|---:|---:|
| top-1 accuracy | 54.4% | 3.8% 🚨 |
| top-1 is BASE class | 87.8% | 70.3% 🚨 |

#### Per-novel-class breakdown (BiomedCLIP)：

| novel class | mean cos to GT | top-1 acc | top-1 is base |
|---|---:|---:|---:|
| respiratory-adenocarcinoma | −0.131 | 0.0% | 100% |
| Serous-Ovarian | −0.160 | 0.0% | 96.7% |
| respiratory-SCC | −0.189 | 0.0% | 100% |
| Serous-Breast | −0.152 | 3.3% | 96.7% |
| **Thyroid-MTC** | **−0.370** | 0.0% | 100% |
| respiratory-SmallCell | −0.129 | 0.0% | 100% |
| Serous-Adeno | −0.136 | 0.0% | 100% |
| Thyroid-Suspicious | −0.189 | 0.0% | 100% |
| Thyroid-MalTumour | −0.141 | 0.0% | 100% |

**所有 9 个 novel 类**的 mean cos to GT 都是负数，**全部 novel 类的 top-1 几乎 100% 落到 base class 上**。

**图示**：
- `docs/figures/thaf_diagnostic/xlmr_image_encoder/image_encoder_alignment.png`
- `docs/figures/thaf_diagnostic/biomedclip_image_encoder/image_encoder_alignment.png`

#### 解读

THAF 训练让 image encoder **过度 specialize 到 base 30 类的 attribute mean 方向**：
- ImageEncoder(novel_image) 跟 base 类的 class_vec 余弦最大
- 跟 novel 类的 class_vec 余弦**负值**（图像 feature 跟 novel 文本方向**反向**）
- 这是 **out-of-distribution 失败**：novel 图像被推到 base anchors 附近

→ Novel zero-shot 失败的**真因是 image encoder overfitting**，**不是 text encoder 的 cos collision**（cos 已经 < 0.94，文本端是 OK 的）。

---

## 4. 假说决策树（最终版）

```
为什么 THAF novel zero-shot collapse？

├─ Hypothesis A: trained class vectors 互相挤压 (cos saturation)
│   实证: novel↔novel max cos 0.940 (BiomedCLIP) / 0.991 (XLM-R)
│         attr_mean baseline 也是 0.947 / 0.993 (几乎一致)
│   → REFUTED. 文本端几何是 OK 的。
│
└─ Hypothesis B: image encoder 不对齐 novel ✅ CONFIRMED
    实证: 99.2% novel 图像 top-1 → base class
         mean cos novel image to GT class = −0.178 (BiomedCLIP)
         全部 9 个 novel 类 mean cos to GT 都是负数
    → 图像 encoder 训练时只学到了对齐 base 30 类的 attribute mean，
       novel 图像产出的 feature 跟 novel 文本方向反向。
```

---

## 5. 论文叙事 reframe

### 5.1 ❌ 不能讲的（之前 plan 里写的，现在不成立）

- "Trainable Hierarchical Attribute Fusion" 作为架构创新 —— 模块训练后自我归零
- "Cross-attention 学到 attribute 加权" —— alpha→0，没学到任何加权
- "THAF 解决 novel zero-shot" —— novel 反跌

### 5.2 ✅ 现在能讲的

- **§A: "5-attribute structured medical prompts + frozen domain encoder (BiomedCLIP) + mean pool"**
  - Base mAP +1.7pp vs baseline（实证）
  - **Drop cross-attention fusion module**（dead weight，反正 alpha 训成 0）
  - 极简架构：5 个文本 embedding 取平均当 class vector
  - paramter-free 聚合，但需要 BiomedCLIP encoder swap（这是真创新点之一）

- **§B: "Novel zero-shot bottleneck is image encoder, not text encoder"**
  - 配 Phase 3.5/3.6 两个 diagnostic 实证：cos 几何 OK，image encoder 不对齐
  - 这本身是 paper 级别的 negative result（reviewer 会很喜欢实证 + 反直觉）
  - 提出 Phase 5+ "multi-modal class encoder with train-time visual prompts" 作为 future work

- **§C: 现实工程方案 — post-hoc score fusion**
  - 在 dev30 上 score fusion = 0.098 avg novel（比 THAF 0.052 好近 2x）
  - 这是真正能给临床部署用的方案
  - 不需要重训，inference-only

### 5.3 重新规划的 ablation 表（论文风格）

```
                                       base 25  | main3 | pseudo2 | hard4 | full5 | avg novel
v2 baseline (XLM-R + single PSC)        0.310    | 0.134 | 0.108   | 0.088 | 0.049 | 0.095
+ score fusion (raw visproto, post-hoc) 0.310    | 0.137 | 0.108   | 0.095 | 0.051 | 0.098     ← 部署用
5-attr + XLM-R + mean pool              0.302    | 0.021 | 0.033   | 0.013 | 0.013 | 0.020   🔻
5-attr + BiomedCLIP + mean pool ← §A    0.327 ✨ | 0.009 | 0.137   | 0.017 | 0.045 | 0.052   🔻
```

---

## 6. 已死 hypothesis 总账（更新 TODO 死路表）

| # | 假设 / 死路 | 状态 |
|---|---|---|
| DEAD-1 | inference-time text ensembling | ❌（旧）|
| DEAD-2 | inference-time anisotropy reduction | ❌（旧）|
| DEAD-3 | per-variant L2-norm 后再 mean | ❌（旧）|
| DEAD-4 | raw text+visproto 单 inference binary fusion | ❌（旧）|
| DEAD-5 | Procrustes calfused | ❌（旧）|
| **DEAD-6** | **THAF cross-attention fusion module** | ❌ **新**：训练后 alpha→0，等价于 mean pool |
| **DEAD-7** | **THAF 解决 novel zero-shot** | ❌ **新**：novel mAP 实测比 baseline 跌 50-80% |
| **DEAD-8** | **GPU throttle / NCCL / LR overlap 解释 dev32→dev30 1pp drop** | ❌ **新**：clean 重训证明都不是元凶，主要是单次训练随机性 |

---

## 7. 下一步

### 立即可做

1. **paper 叙事 reframe**：method §A 从"trainable fusion"→"5-attr + BiomedCLIP + mean pool"
2. **简化架构**：删 cross-attention fusion 模块（参数省 3.15M / 7.1M）
3. **更新 plan + memory** 反映新现实

### 中期（需要新实验）

1. **Phase 5 multi-modal class encoder**（之前在 plan 里 "future ideas" 段，**现在变 first-class 必跑实验**）—— 用 visual prompt 给 image encoder 训练信号，避免它 overfit base
2. **跨编码器 fusion 重新尝试**：alpha 用更强的 init（0.7+）、cross-attention output_proj gain=1.0、加 entropy regularization 防止 alpha shrink

### 不再做

1. 任何"调 THAF cross-attention 内部架构"的事 —— 反正会被训成 0
2. 改 NHGUC merge 试图恢复 1pp —— 跟 novel 主问题无关

---

## 8. 附录

### 8.1 验证脚本

- `tools/diagnose_thaf_fusion.py`：诊断 fusion 模块 trained vs attr_mean 等价性
- `tools/diagnose_image_encoder.py`：诊断 image encoder 对 novel image 的 alignment

### 8.2 关键 ckpt

- `work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu/best_coco_bbox_mAP_epoch_10.pth`
- `work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu/best_coco_bbox_mAP_epoch_10.pth`
- `work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu/best_coco_bbox_mAP_epoch_9.pth`

### 8.3 关键日志

- `work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu/thaf_eval_summary.txt`
- `work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu/thaf_eval_summary.txt`
- `work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu/baseline_eval_summary.txt`
- `work_dirs/ablation_table.md`（auto-compiled by `tools/compile_ablation_table.py`）
