# noTHAF BiomedCLIP Ablation — Per-Class Decomposition Analysis

**日期**：2026-05-11
**目的**：完成 (encoder, text-format) × {clean dev30 / noTHAF / THAF BiomedCLIP} 的 3-点 ablation 网格，per-class 拆解 +1.7pp 的来源。

---

## 0. TL;DR

THAF BiomedCLIP base 25-cls 0.327 vs clean dev30 0.310 的 **+1.7pp** 拆分：

| 贡献来源 | 涨幅 | 占比 | 一致性 |
|---|---:|---:|---|
| **Encoder swap (XLM-R → BiomedCLIP)** | **+1.1pp** | **65%** | 19/25 类涨（76% 一致）✅ |
| **5-attr 富文本** | **+0.6pp** | 35% | 11/25 类涨（44%，**有涨有跌**）⚠ |
| **THAF cross-attention fusion** | ≈ 0 | 0% | DEAD-6 (alpha→0) |

→ **Encoder swap 是 CLEAN 主功臣**（论文 §A 强 claim）；**5-attr 是 mixed 次功臣**（论文需要 nuance 解释）。

---

## 1. Ablation 网格

| Method | Encoder | Text | Fusion | Base 25-cls |
|---|---|---|---|---:|
| clean dev30 | XLM-R 768d | 1 PSC | — | **0.310** |
| **noTHAF** ← 本次新增 | **BiomedCLIP 512d** | **1 PSC** | **—** | **0.321** (+1.1pp) |
| THAF BiomedCLIP | BiomedCLIP 512d | 5 attr | THAF (α→0) | 0.327 (+1.7pp) |
| THAF XLM-R | XLM-R 768d | 5 attr | THAF (α→0) | 0.302 (−0.8pp) |

**关键**：noTHAF 这一点填了网格的 (BiomedCLIP, 1 PSC) 缺口，使 encoder 和 5-attr 两个变量首次能**单变量分离**。

XLM-R 这条线：1 PSC=0.310 → 5 attr THAF=0.302（**反跌**），说明 XLM-R 不能 hold 5-attr 信号（per memory `feedback_xlmr_text_saturation_medical.md`）。

---

## 2. Per-class breakdown（25 classes）

### 2.1 Encoder swap effect (noTHAF − clean) — top10 / bot5

| Class | clean | noTHAF | enc Δ |
|---|---:|---:|---:|
| respiratory tract-Diseased cells | 0.399 | 0.456 | **+5.7pp** |
| Serous effusion-Diseased cells | 0.245 | 0.299 | **+5.4pp** |
| Urine-SHGUC | 0.097 | 0.146 | **+4.9pp** |
| Thyroid gland-AUC | 0.061 | 0.108 | **+4.7pp** |
| Thyroid gland-FC | 0.482 | 0.519 | +3.7pp |
| Thyroid gland-NS | 0.161 | 0.186 | +2.5pp |
| Thyroid gland-SPTC | 0.232 | 0.248 | +1.6pp |
| respiratory tract-Alveolar macrophages | 0.520 | 0.535 | +1.5pp |
| Urine-AUC | 0.073 | 0.088 | +1.5pp |
| respiratory tract-Lymphocyte | 0.184 | 0.195 | +1.1pp |
| ... (15 中间 类略) ... | | | |
| respiratory tract-Neutrophil | 0.281 | 0.270 | −1.1pp |
| Thyroid gland-PTC | 0.496 | 0.485 | −1.1pp |
| TCT_CCD-monilia | 0.127 | 0.113 | −1.4pp |
| respiratory tract-Squamous epithelial cells | 0.629 | 0.603 | −2.6pp |
| Thyroid gland-Macrophages | 0.547 | 0.516 | −3.1pp |

**Aggregate**：mean +1.1pp，std 2.3pp，**19/25 类涨（76% 一致）**。

### 2.2 5-attr effect (THAF − noTHAF) — top10 / bot5

| Class | noTHAF | THAF | 5-attr Δ |
|---|---:|---:|---:|
| **Serous effusion-Diseased cells** | 0.299 | 0.414 | **+11.5pp** ✨ |
| Thyroid gland-Macrophages | 0.516 | 0.608 | **+9.2pp** |
| respiratory tract-Squamous epithelial cells | 0.603 | 0.660 | +5.7pp |
| TCT_CCD-monilia | 0.113 | 0.133 | +2.0pp |
| respiratory tract-Diseased cells | 0.456 | 0.476 | +2.0pp |
| Thyroid gland-NS | 0.186 | 0.202 | +1.6pp |
| respiratory tract-Lymphocyte | 0.195 | 0.211 | +1.6pp |
| Urine-AUC | 0.088 | 0.101 | +1.3pp |
| Thyroid gland-PTC | 0.485 | 0.490 | +0.5pp |
| respiratory tract-Neutrophil | 0.270 | 0.274 | +0.4pp |
| ... (10 中间 类略) ... | | | |
| Thyroid gland-AUC | 0.108 | 0.084 | −2.4pp |
| respiratory tract-Alveolar macrophages | 0.535 | 0.511 | −2.4pp |
| Urine-HGUC | 0.402 | 0.378 | −2.4pp |
| Thyroid gland-SPTC | 0.248 | 0.207 | −4.1pp |
| **Thyroid gland-FC** | 0.519 | 0.452 | **−6.7pp** 🔻 |

**Aggregate**：mean +0.6pp，std 3.8pp，**11/25 类涨（仅 44% 一致）**——**有涨有跌**。

### 2.3 Per-organ aggregate

| Organ | n | enc avg | 5-attr avg | total avg |
|---|---:|---:|---:|---:|
| **Serous effusion** | 1 | +5.4pp | **+11.5pp** | **+16.9pp** ✨ |
| Urine | 3 | +2.1pp | −0.6pp | +1.5pp |
| Thyroid gland | 6 | +1.4pp | −0.3pp | +1.1pp |
| respiratory tract | 6 | +0.8pp | +1.2pp | +2.0pp |
| TCT_CCD | 9 | +0.4pp | −0.1pp | +0.3pp |

---

## 3. Interpretation

### 3.1 Encoder swap：**clean win，但偏向"病理类"**

**Top winner pattern**：
- "Diseased cells" 类（Resp +5.7pp，Serous +5.4pp）
- 罕见 / 异型 类（Urine-SHGUC +4.9pp，Thyroid-AUC +4.7pp）

**Loser pattern**：
- "正常细胞" 类（Resp-SE −2.6pp，Resp-Neutrophil −1.1pp，Thyroid-Macrophages −3.1pp）
- 大常见类（Resp-Squamous −2.6pp，Thyroid-Macrophages −3.1pp）

**Why**：BiomedCLIP 在 PubMedBERT 文本预训练时见过大量医学文献——病理 / 罕见 / 异型 类的医学描述（"adenocarcinoma"、"squamous cell carcinoma"、"PSC VI Malignant"）在 PubMed 出现频率高，encoder 学到了 fine-grained 区分；而 "正常细胞"（"squamous epithelial cells"、"neutrophil"）也在 general 语料里常见，**BiomedCLIP 没有相对优势**，反而在编码长度 / dim (512 vs 768) 上略亏。

**论文 §A 可讲**："BiomedCLIP encoder swap helps pathological classes by an average of +X.X pp while maintaining normal-cell class performance within noise."

### 3.2 5-attr：**mixed，需要 nuance**

**Top winners 显著**：
- **Serous-Diseased +11.5pp**：单一最大涨幅。5-attr 提供了**多视角描述**（形态学 + 免疫表型 + 鉴别点）来区分各种 Diseased cells，对 cytomorphology 高度异质的 Serous effusion 极其有用
- Thyroid-Macrophages +9.2pp：跟 Thyroid 其他细胞类区分（"foamy macrophages with abundant cytoplasm" vs FC / NS）
- Resp-Squamous +5.7pp：详细形态学描述（"intermediate squamous epithelial cells with vesicular nuclei"）

**Top losers**：
- **Thyroid-FC −6.7pp 🔻**：5-attr 可能让 FC 跟 SPTC 太相似（同属滤泡性病变），分类反而模糊
- Thyroid-SPTC −4.1pp：跟 FC 互相挤压
- Thyroid-AUC −2.4pp：富文本反而稀释了 AUC 的简单形态特征

**Pattern**：5-attr 帮**形态学异质 / 需要多视角区分**的类（Diseased cells、Macrophages），**伤害形态学相似的邻近类对**（Thyroid 滤泡性病变三胞胎 FC/SPTC/AUC）——多 attribute 反而**加剧 cos saturation**。

这跟 memory entry `feedback_xlmr_text_saturation_medical.md` 的发现一致：medical fine-grained 上同胞类的 cos 容易爆。**5-attr 不是 universal win，是 selective win**。

### 3.3 论文叙事 nuance

不能简单讲"5-attr 涨 base 0.6pp"——更准确的说法：

> "Switching from a single Papanicolaou system prompt to a 5-attribute structured representation (organ, diagnostic code, cytomorphology, background/immunoprofile, distinguishing feature) improves base mAP by 0.6pp on average, but the effect is class-dependent: it provides strong gains (+5-11pp) for morphologically heterogeneous classes ('Diseased cells', 'Macrophages') and incurs losses (-4 to -7pp) for nearby class pairs in the same diagnostic category (Thyroid follicular cell variants), where additional attribute detail amplifies inter-class cosine saturation."

---

## 4. 跟之前 Phase 3.5 diagnostic 的一致性

Phase 3.5 实测 trained THAF ≈ attr_mean（α→0），意味着 noTHAF (1 PSC) → THAF BiomedCLIP (5 attr + α→0) 实际上 **= 1 PSC → mean(5 attr)** 的对比。

也就是说，**第 3 节的 5-attr 效果 = mean pooling 5 attrs vs 1 PSC 的效果**，**没有任何 fusion 智能加权**。

这给用户的"四不像"质疑提供了实证素材：mean pool 5 个 attrs **在 Thyroid 滤泡变体三胞胎上确实糟糕**（FC −6.7pp，SPTC −4.1pp）。如果 Option 3a (per-class weights) 能**只在这些类用更稀疏的权重**（比如 Thyroid-FC 只重 attr 5=distinguishing），理论上能救回这部分 loss。

→ **Option 3a 的预期 base 涨点上限**：救回 Thyroid 滤泡变体（FC、SPTC、AUC）= 大约 +0.7pp 在 base 25-cls 上（如果三个类都能恢复到 noTHAF 水平）。如果 Option 3a 能 +1pp 在 base 上，说明用户直觉**完全对**。

---

## 5. 更新的实验决策

跟 Option 3a / 4 跑完的对照：

| Outcome | Interpretation |
|---|---|
| Option 3a 涨 +1pp+ on base | per-class weighting 救回了 Thyroid 三胞胎 → 论文 §A 升级为 "per-class attribute weighting" |
| Option 3a ≈ THAF BiomedCLIP 0.327 | mean pool 真的是 5-attr 上限 → 论文 §A 用 "BiomedCLIP + mean pool" |
| Option 4 (concat+proj) 涨 base | non-linear combination 学到了 attribute 选择 → 论文 §A 用 concat proj |
| Option 4 不涨 | projection 没学到东西，跟 mean pool 一样 |

**预测**：Option 3a 涨 0.5-1.5pp（救 Thyroid 三胞胎概率高）；Option 4 涨 0-1pp（projection 容量比 per-class 大但没 inductive bias）。

---

## 6. 附录

### 6.1 Eval 配置

- 测试集：`/home1/liwenjie/TCT_NGC/annotations/instances_test_base_clean_dev30.json`
- 排除类：5 个 dev30 negatives（Resp-Impurity、Serous-Neg、Thyroid-Neg、Urine-NHGUC、TCT_CCD-normal）
- noTHAF ckpt: `work_dirs/.../noTHAF_2gpu/best_coco_bbox_mAP_epoch_11.pth`
- clean ckpt: `work_dirs/.../disjoint_clean_2gpu/best_coco_bbox_mAP_epoch_9.pth`
- THAF BiomedCLIP ckpt: `work_dirs/.../thaf_biomedclip_2gpu/best_coco_bbox_mAP_epoch_10.pth`

### 6.2 关键 log

- `work_dirs/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu/eval_base_25cls.log`
- `work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu/eval_base_25cls_clean/20260510_202814/20260510_202814.log`
- `work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu/thaf_eval_summary.txt`
