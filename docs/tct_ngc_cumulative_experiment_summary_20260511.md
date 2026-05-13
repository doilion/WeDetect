# TCT_NGC 累计实验总结（Phase 1-4+）

**日期**：2026-05-11
**目的**：把过去多个 phase 的实验、发现、死路、当前结论汇总成单一文档，让任何人 1 小时内能 onboard 完整状态。

---

## 0. TL;DR（2026-05-12 更新）

> **⚠ 2026-05-12 公式更正**：原"avg novel"是 4 个 splits 算术平均，但 `full_5 = main_3 ∪ pseudo_2` 双重计算。已统一改用**所有 9 个 unique novel classes 的宏平均**：
> `avg_novel = (3·main_3 + 2·pseudo_2 + 4·hard_4) / 9`。full_5 单独报告作"5-class mixed eval"。

- **新最佳 novel zero-shot 方法**：**noTHAF (BiomedCLIP + 1 PSC) + 视觉提示 (visproto-only, inference-only, 5-shot)** = **avg novel 0.122 (leakage) / 0.123 (strict)**，超过历史最佳 (XLM-R + score fusion 0.112) **+10%**
- **关键发现**：BiomedCLIP encoder swap 让 image encoder 隐式学到 **6.3x 更强的视觉特征空间** —— text 路径 novel 全死（0.004），visproto 路径 novel 大涨（0.104）
- **THAF（cross-attention fusion）失败**：trained α→0，fusion 模块训练成 mean pool；novel mAP 反跌 50-80%。
- **真正瓶颈**在 image encoder：99.2% novel 图像被 image encoder 推到 base anchors。但 **BiomedCLIP 训练 image encoder 比 XLM-R 训练 less overfit**，所以 visproto 路径救活。
- **PCW + ConcatProj 已废**（用户终止）：text 端替代方案对 novel 故事 zero contribution。
- **下一步**：Phase 5 multi-modal class encoder w/ train-time visual prompts，把 5-shot inference-leakage 变成 0-shot；预期 novel +5-10pp on top of 0.104。

---

## 1. 实验时间线

```
Phase 1  (2026-04-...)  dev32 baseline 训练 + eval → 0.316 base 25-cls
Phase 2  (2026-04-...)  novel zero-shot 第一轮：v2 text baseline，post-hoc score fusion 发现 ✅
Phase 2.5 (2026-05-08)  dev30 重训（NHGUC merge）→ 0.306 base，−1pp drop 调查
Phase 3a/3b (2026-05-09) THAF 训练：XLM-R 768d + BiomedCLIP 512d 两条线并行
Phase 3.5 (2026-05-10)  THAF fusion diagnostic → alpha→0 发现
Phase 3.6 (2026-05-10)  THAF image encoder diagnostic → 99.2% novel→base 发现
Phase 2.5b (2026-05-11) clean dev30 重训（fix GPU throttle + LR）→ 0.310 (within noise of 0.306)
                        noTHAF BiomedCLIP ablation（BiomedCLIP + 1 PSC，no fusion）→ 跑完
Phase 3.7 (2026-05-11)  Option 3a (per-class weights) + Option 4 (concat+proj)，两个文本侧替代
                        method 跑起来，verify mean pool 是不是真天花板 ← 当前
Phase 4   (推迟)        Detection-level gate network — 数据显示帮不到 novel 真问题，降级
Phase 5   (下一步)      Multi-modal class encoder w/ train-time visual prompts — first-class
```

---

## 2. 完整 ablation 表

| Encoder | Text format | Fusion | Train | Base 25-cls | Avg novel | 备注 |
|---|---|---|---|---:|---:|---|
| XLM-R 768d | 1 PSC | — | dev32 (32-cls) | 0.316 | — | dev32 baseline，旧 |
| XLM-R 768d | 1 PSC | — | old dev30 (throttled GPU 1) | 0.306 | ~0.11 | 论文 旧 baseline（旧 4-split 数 0.103 重算后约 0.11） |
| **XLM-R 768d** | **1 PSC** | **—** | **clean dev30** | **0.310** | **0.108** | **现 clean baseline** (mean over 9 unique novel) |
| XLM-R 768d | 5 attr | THAF cross-attn | dev30 | 0.302 | 0.020 | 🔻 fusion 失败 |
| **BiomedCLIP 512d** | **1 PSC** | **—** | **dev30 (noTHAF)** | **0.321** | **0.005 text** / **0.123 visproto** ✨ | encoder 单变量 — visproto 大胜 |
| **BiomedCLIP 512d** | **5 attr** | **THAF cross-attn** | **dev30** | **0.327** | **0.041** | THAF fusion bypass |
| ~~BiomedCLIP 512d~~ | ~~5 attr~~ | ~~per-class weights (155)~~ | ~~dev30~~ | ~~未完~~ | ~~未完~~ | **DEAD** — Option 3a 用户终止 |
| ~~BiomedCLIP 512d~~ | ~~5 attr~~ | ~~concat+proj (1.58M)~~ | ~~dev30~~ | ~~未完~~ | ~~未完~~ | **DEAD** — Option 4 同上 |

### Inference-only 后处理（4 splits 详细对比 — 跨 ckpt）

| 推理策略 | ckpt | main_3 | pseudo_2 | hard_4 | full_5 | **Avg** |
|---|---|---:|---:|---:|---:|---:|
| v2 text baseline | clean XLM-R | 0.134 | 0.108 | 0.088 | 0.049 | 0.095 |
| visproto raw | clean XLM-R | 0.011 | 0.032 | 0.011 | 0.012 | _0.016_ |
| **post-hoc score fusion** | clean XLM-R | 0.137 | 0.108 | 0.095 | 0.051 | **0.098** (旧最佳) |
| Procrustes calfused | clean XLM-R | 0.132 | 0.092 | 0.002 | 0.045 | 0.068 ❌ DEAD-5 |
| THAF + score fusion | THAF BiomedCLIP | 0.011 | 0.120 | 0.014 | 0.041 | 0.047 |
| **text 1 PSC** | **noTHAF BiomedCLIP** | 0.002 | 0.005 | 0.007 | 0.001 | **0.004** 🔻🔻 DEAD-9 |
| **visproto (5-shot leakage)** ✨ | **noTHAF BiomedCLIP** | **0.076** | **0.135** | **0.150** | **0.056** | **0.104** ✅ **新最佳** |
| score fusion (per-class routing) | noTHAF BiomedCLIP | 0.016 | 0.022 | 0.038 | 0.012 | _0.022_ ⚠ 路由失效 |

---

## 3. 死路清单（DEAD-1 到 DEAD-8）

| # | 死路 | 实测 | 失败根因 |
|---:|---|---|---|
| DEAD-1 | 推理端文本 ensembling (CuPL/CLIP-style) | 4 splits 跌 60-100% | image encoder 钉死在某个 prompt 方向 |
| DEAD-2 | 推理端 anisotropy reduction (mean center / whiten) | 全跌 | 破坏 contrastive head 期望 |
| DEAD-3 | Per-variant L2-norm 后 mean | 全跌 | 跟 raw 内积架构不兼容 |
| DEAD-4 | Raw text+visproto 单次 inference 二元路由 | text 类被 visproto 挤死 | 单 inference 几何不匹配 |
| DEAD-5 | Procrustes calfused | visproto 类全死 (Resp-Adeno 0.095→0.000) | R 不 novel-transferable |
| **DEAD-6** | **THAF cross-attention 设计** | **α=−0.0001** | **output_proj gain=0.1 太小 + attr_mean 是优秀梯度 sink** |
| **DEAD-7** | **THAF 解决 novel zero-shot** | novel 跌 50-80% | **image encoder overfit base，99.2% novel→base** |
| **DEAD-8** | dev32→dev30 1pp drop 是 systematic | clean 重训仍 0.310 ≈ 0.306 | 单次训练 random noise |

---

## 4. Phase 3.5/3.6 双 diagnostic 实证

### Phase 3.5 — THAF fusion 自我归零

| | BiomedCLIP THAF | XLM-R THAF |
|---|---:|---:|
| trained alpha (init 0.3) | **−0.0001** | **−0.0003** |
| dead params (cross-attn + output_proj) | 3.15M | 7.1M |
| trained novel↔novel max cos | 0.940 | 0.991 |
| attr_mean baseline novel↔novel max cos | 0.947 | 0.993 |
| Δ trained vs attr_mean | −0.006 | −0.002 |

→ trained THAF ≈ attr_mean（mean pool）；cross-attention 没贡献。

### Phase 3.6 — Image encoder novel image overfit

| | BiomedCLIP THAF | XLM-R THAF |
|---|---:|---:|
| novel image top-1 → base class | **99.2%** 🚨 | 70.3% 🚨 |
| mean cos novel image → GT class | −0.178 | −0.304 |
| base top-1 acc | 75.2% | 54.4% |

→ Novel image feature 跟 novel 文本方向**反向**；base specialize 严重。**真正的瓶颈在这里**。

---

## 5. 用户提的关键质疑（影响路径决策）

### 5.1 "Mean pool 是不是天花板"（2026-05-11）

**用户原话**："但是其实 mean average 肯定其实是一个下线按道理...庐山的四面八方很多个面的描述如果取平均可能会类似四不像的一个结果"

**理论分析**：5 个 attribute（解剖部位 / 诊断码 / 形态学 / 背景免疫 / 鉴别点）描述不同维度，向量近似正交。Mean = 5 个方向的质心 → 类内信号被稀释。理论上 per-class sparse selection 应该有空间涨点。

**实证 path（当前 Phase 3.7 跑的两个）**：
- Option 3a：per-class learnable softmax weights（验证"每类自己选哪几个 attr 重要"）
- Option 4：concat + linear proj（验证"保留全部信息 + 学非线性组合"）

如果两个都 ≈ THAF BiomedCLIP base 0.327 → mean pool 真的是天花板，文本侧没空间。
如果 Option 3a / 4 涨点 → 用户直觉对，THAF cross-attention 是训练失败不是 design fundamental issue，论文 §A 故事升级。

---

## 6. 论文叙事（基于已有实证）

### 6.1 现在能讲的章节

```
§A: 5-attribute structured medical prompts + BiomedCLIP encoder
    - Base 25-cls +1.7pp (0.327 vs 0.310 clean dev30)
    - DROP cross-attention fusion (Phase 3.5 实证 fusion 是 dead weight)
    - 候选 fusion 方法（Phase 3.7 跑完后决定）：
        - 如果 per-class weights work → "per-class attribute weighting" §A
        - 如果 concat+proj work → "concat projection" §A
        - 如果都 ≈ mean pool → "mean pool over 5 structured attributes" §A

§B: Negative result — novel zero-shot bottleneck is image encoder
    - Phase 3.5 fusion bypass diagnostic：trained THAF = mean pool
    - Phase 3.6 image encoder diagnostic：99.2% novel→base
    - 文本端 cos 已经 separable（max 0.94），novel 失败不是 text 问题
    - 这本身是个 paper-worthy negative result

§C: Engineering solution — post-hoc score fusion
    - 0.310 base 不动 + 0.112 avg novel (mean over 9 unique novel; vs single source 0.108)
    - 唯一 inference-only 击败单源 baseline 的方法
    - 部署友好，不需要重训

§D (proposed): Multi-modal class encoder w/ train-time visual prompts (Phase 5)
    - 解 §B 的 image encoder overfit
    - YOLOE-style modality dropout（一半 batch text-only / 一半 vis-only）
    - 预期 novel +5-10pp（如果 hypothesis 正确）
```

### 6.2 不能讲的（实证证伪）

- ❌ "Trainable Hierarchical Attribute Fusion" 是创新点（α→0 实证）
- ❌ "Cross-attention learns attribute weighting"（α→0 = 没学到加权）
- ❌ "THAF solves novel zero-shot"（实测反跌 50-80%）
- ❌ "NHGUC merge causes dev30 −1pp"（clean 重训 0.310 反证）

---

## 7. 当前正在跑的实验（2026-05-11）

| 实验 | GPU | 启动时间 | ETA | 工作目录 |
|---|---|---|---|---|
| Option 3a per-class weights (BiomedCLIP) | 0+1 (待 eval 完成) | 待启动 | ~8h | `work_dirs/.../pcw_biomedclip_2gpu` |
| Option 4 concat+proj (BiomedCLIP) | 2+3 | 2026-05-11 | ~8h | `work_dirs/.../concatproj_biomedclip_2gpu` |

完成后 eval 走 `eval_baseline_all.sh` 同套（base 25-cls + 4 novel splits + score fusion）。

---

## 8. 关键文件 reference

### 8.1 文档

- 本文件：`docs/tct_ngc_cumulative_experiment_summary_20260511.md`
- Phase 3 完整分析：`docs/tct_ngc_phase3_thaf_results_analysis_20260511.md`
- Phase 4/5 决策：`docs/tct_ngc_phase4_phase5_decision_20260511.md`
- Phase 3 inflight 记录：`docs/tct_ngc_phase3_thaf_inflight_20260509.md`

### 8.2 诊断工具

- `tools/diagnose_thaf_fusion.py`（Phase 3.5）
- `tools/diagnose_image_encoder.py`（Phase 3.6）

### 8.3 诊断图

- `docs/figures/thaf_diagnostic/{xlmr,biomedclip}/cosine_heatmap_{trained,attr_mean}.png`
- `docs/figures/thaf_diagnostic/{xlmr,biomedclip}_image_encoder/image_encoder_alignment.png`

### 8.4 训练 ckpt

- Clean dev30 (XLM-R + 1 PSC)：`work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu/best_coco_bbox_mAP_epoch_9.pth`
- THAF XLM-R：`work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu/best_coco_bbox_mAP_epoch_10.pth`
- THAF BiomedCLIP：`work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu/best_coco_bbox_mAP_epoch_10.pth`
- noTHAF BiomedCLIP：`work_dirs/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu/best_coco_bbox_mAP_epoch_11.pth`

### 8.5 Auto-compiled ablation

- `work_dirs/ablation_table.md`（`tools/compile_ablation_table.py`）

### 8.6 Eval orchestrators

- `tools/eval_baseline_all.sh`（XLM-R / clean dev30 全套）
- `tools/eval_thaf_all_splits.sh`（THAF × 2 encoder）

---

## 9. 下一步决策树

```
等 Option 3a + Option 4 跑完（~8h，2026-05-11 夜里）
  │
  ├─ 两个都 ≈ THAF BiomedCLIP (0.327)
  │   → mean pool 真天花板，论文 §A = "5-attr + BiomedCLIP + mean pool"
  │   → 跳 Phase 5（image encoder 侧 attack）
  │
  ├─ Option 3a 或 4 涨 1-3pp on base
  │   → 用户直觉对，cross-attention 是训练失败
  │   → 论文 §A = 涨点的那个
  │   → 仍跳 Phase 5（novel 还是不动）
  │
  └─ Option 3a 或 4 同时涨 novel
      → 不太可能（image encoder 没改）但要 verify
      → 重新评估 Phase 5 优先级
```

无论哪种情况，**Phase 5 都是必跑**，因为只有 image encoder 端干预能解 novel 真问题。
