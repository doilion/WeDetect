# Phase 4 / Phase 5 优先级决策（基于 Phase 3 完整结果）

**日期**：2026-05-11
**依据**：`docs/tct_ngc_phase3_thaf_results_analysis_20260511.md` + `work_dirs/ablation_table.md`

---

## 决策摘要

| Phase | 之前 plan 状态 | 新决策 | 理由 |
|---|---|---|---|
| **Phase 4** Detection-level gate network | 下一步立即做 | **DEMOTE → 可选** | 不解决 novel 真因（image encoder 端）；THAF+score fusion 数据显示 fusion 思路对 novel 没帮助 |
| **Phase 5** Multi-modal class encoder w/ visual prompts | future ideas，未承诺 | **PROMOTE → first-class 必跑** | 直接 attack DEAD-7 (image encoder novel image 反向对齐) |

---

## Phase 4 (Detection-level gate) — DEMOTE 理由

### 原 Plan 的预期

```
text branch (THAF/v2)  →  detection 1 with score₁
visproto branch (raw)  →  detection 2 with score₂
gate(image_feat, score₁, score₂) → α
final = α·score₁ + (1−α)·score₂
```

预期：比 score fusion 的"硬规则 per-class 路由"涨 1-3 mAP。

### Phase 3 实证给出的判断

**注**：avg novel = mean over 9 unique novel = `(3·main_3 + 2·pseudo_2 + 4·hard_4) / 9`（2026-05-12 重算公式）。

| 实验 | avg novel mAP (9 unique) | 跟 Phase 4 gate 的关系 |
|---|---:|---|
| v2 text | 0.108 | gate 的 source 1 候选 |
| raw visproto (XLM-R) | 0.020 | gate 的 source 2 候选 |
| score fusion | 0.112 | gate 要超过这个 |
| THAF + BiomedCLIP | 0.041 | THAF 端反而比 v2 text 差 |
| THAF + BiomedCLIP + score fusion | 0.038 | **THAF + score fusion 比 score fusion 还差** ⚠ |

### 关键观察

THAF + score fusion = 0.038 < score fusion (v2) = 0.112 → **加 THAF 反而拖累**。这说明：

1. **Score fusion 已经把两个 source 的"互补部分"取出来了**——再加 gate 学路由收益边际
2. **THAF source 在 novel 上质量差**（avg 0.041）—— gate 学到 α≈0 不调用 THAF 才是最优
3. **真正的限制在 image encoder 端**（DEAD-7 Phase 3.6）—— 不管换什么 source，image encoder 都把 novel 推到 base anchors

### 推迟 Phase 4 的影响

- 论文章节结构不变：method §A (5-attr+BiomedCLIP) + §B negative result (image encoder bottleneck) + §C (score fusion 部署方案)
- §C 现在就有最强候选：post-hoc score fusion（不需要 gate network）
- 如果有时间再做 gate 作为补充 ablation，证明 "learned routing 也救不了" 加强 §B 的 negative result

### 何时再启动 Phase 4

如果 Phase 5 multi-modal encoder 训出来 image encoder 不再 overfit base，那时两个 inference source 的预测质量都 OK，gate network 可能有真正空间涨 1-3pp。**但这是 Phase 5 之后的事**。

---

## Phase 5 (Multi-modal class encoder w/ visual prompts) — PROMOTE 理由

### 原 Plan 的设计（plan 文件 Future ideas 段）

3 个工程层次：
- 层次 1: visual prototype 当 cross-attention 第 6 个 attribute（~1 天）
- 层次 2: YOLOE-style 双路并联 + modality dropout + alignment loss（~3 天 + 8h GPU）
- 层次 3: 完整 YOLOE 含 prompt-free 模式（~1-2 周）

### 为什么现在 PROMOTE

Phase 3.6 image encoder diagnostic 实证：

| metric | BiomedCLIP THAF | XLM-R THAF |
|---|---:|---:|
| novel image top-1 落到 base class 比例 | **99.2%** 🚨 | 70.3% 🚨 |
| mean cos novel image to GT novel class | −0.178 | −0.304 |

→ image encoder 训练时**只见过 base 30 类的图像**，特征空间 specialize 到 base attribute mean 方向。Novel 类图像产出的 feature **跟 novel 文本方向反向**，被推到 base anchors 附近。

**唯一解决方法是给 image encoder 训练时见过 novel-style 信号**。但 novel 类训练时不可见（zero-shot 前提）。**Multi-modal class encoder 用 visual prompt（base 类的几张 GT crop）当"虚拟 novel"训练信号**，让 image encoder 学到"对齐到 prompt 来源（不管是 text 还是 image）"，而不是"对齐到 base 类的 text mean 方向"。

### 工程层次升级路径

**层次 1（先做，~1-2 天 + 8h GPU）**：
- 把 visual prototype 作为 fusion module 第 6 个 attribute 输入
- 训练时 50% modality dropout（一半 batch 不传 vis）
- 这一步是 cheap 验证："image encoder 能不能学会兼容 text + vis 两种来源"
- 如果 novel zero-shot 涨到 0.10+ → 升级层次 2

**层次 2（如果层次 1 work，~3 天 + 8h GPU）**：
- 双路并联：text THAF + visual prompt encoder 共享 768d 空间
- alignment loss `L_align = ||text_emb(cls) − vis_emb(cls)||²` per class
- 3 模式 modality dropout（text-only / vis-only / both）
- 论文 method §B 候选（"Multi-modal class encoder w/ train-time visual prompts"）

**跟 DEAD-4/5 的本质区别**：DEAD-4/5 是 inference-only 硬混（几何不匹配）；Phase 5 是**训练时联合**，image encoder 从一开始就学跟两种 source 都对齐。

---

## 修订后的论文章节预期

```
§A: 5-attr structured prompts + BiomedCLIP + mean pool
    - 实证 base +1.7pp（0.327 vs 0.310）
    - 极简架构，无 fusion module
    - 关键 finding：trainable fusion 不必要（alpha→0 实证）

§B: Negative result — novel zero-shot bottleneck is image encoder
    - Phase 3.5 fusion bypass diagnostic
    - Phase 3.6 image encoder alignment diagnostic（99.2% novel→base）
    - 文本端 cos geometry 已经 OK（max cos < 0.94）

§C: Engineering solution — post-hoc score fusion
    - 0.310 base 不动 + 0.112 avg novel (mean over 9 unique novel)
    - 唯一 inference-only 击败单源的方法
    - 部署友好（不需要重训）

§D (proposed): Multi-modal class encoder w/ visual prompts ← Phase 5
    - 解决 §B identified 的 image encoder 端瓶颈
    - YOLOE-style 双模态 prompt 训练
    - 预期 novel +5-10pp（如果 hypothesis 对）
```

---

## 行动项

1. ✅ 写 Phase 3 results analysis doc
2. ✅ 更新 TODO 加 DEAD-6/7/8
3. ✅ 更新 memory 加 Phase 3 results entry
4. ⏸ Phase 4 detection-level gate：**降级为可选**，等 Phase 5 出结果后再评估
5. ▶ **Phase 5 启动**：先做层次 1（visual proto 当第 6 attribute + modality dropout），~1 天工程 + 8h GPU 重训
6. ▶ 更新 `/home/25_liwenjie/.claude/plans/zazzy-hugging-pearl.md` 反映新现实

---

## 数据备份

完整数字、per-class diff、所有 PNG 图都在：
- `docs/tct_ngc_phase3_thaf_results_analysis_20260511.md`
- `docs/figures/thaf_diagnostic/`
- `work_dirs/ablation_table.md`
