# Phase 5 Plan — Multi-Modal Class Encoder w/ Visual Prompts

**日期**：2026-05-11
**目的**：直接 attack DEAD-7 — image encoder 把 99.2% novel image 推到 base anchors。
**依赖**：Option 3a + Option 4 跑完 + eval（~05:40 PT 完）；不阻塞，Phase 5 plan 可以并行成型。

---

## 0. 问题陈述

Phase 3.6 实证：
- ImageEncoder(novel_image) 的特征 **跟 novel 文本方向反向**（mean cos = −0.178 / −0.304）
- **99.2% novel image top-1 → base class**（即使 GT 是 novel）
- 文本端 cos geometry 已 OK（novel↔novel max 0.94，可分）

**为什么 image encoder 这样？** 训练时只见过 base 30 类图像 → encoder specialize 到 base attribute mean 方向。Novel 类**完全没见过类似图像**，feature 落到 base anchors 附近。

**修法核心**：训练时让 image encoder 见过"对齐到 prompt 来源"这件事，而不是"对齐到 base 类 text mean 方向"。给它 **visual prompt** 作为 alternative class signal，强迫它学**通用的"feature ↔ class signal"映射**，而非"feature ↔ base text"映射。

---

## 1. 三级工程层次

### Level 1（推荐先做，1-2 day 工程 + 8h GPU）

**思路**：visual prototype 当作第 6 个 attribute，加进现有 5-attr fusion pipeline。Modality dropout 让 image encoder 学到 "5 attrs 或 5 attrs+vis 都要 work"。

**Forward 改造**：
```python
# Current (Option 3a / 4):
attr_embs: [B, C, 5, D]  ← from text per-attr cache
class_vec = fusion(attr_embs)

# Phase 5 Level 1:
attr_embs: [B, C, 5, D]   ← from text per-attr cache (frozen)
vis_embs:  [B, C, 1, D]   ← from image encoder + crop pool (training: real GT crops; inference: novel 5-shot)
all_attrs: [B, C, 6, D]  ← concat along attr axis
class_vec = fusion(all_attrs)  ← reuse PCW / ConcatProj / mean
```

**关键设计点**：

1. **训练时 visual prompt 来源**：base 30 类的 train set GT crops，每 batch 采样 1-3 个 crop per class。
2. **Modality dropout**：50% batch 用 `[5 text attrs]`（mask vis），50% batch 用 `[5 text attrs + vis]`。
   - 这是 trainability 的关键：forces image encoder to handle text-only AND text+vis paths
   - 没有 dropout 的话，模型会过度依赖 visual prompt，novel 推理时（用 5-shot from test set）泛化更差
3. **Inference for novel**：用 novel 类的 5-shot crop（test set leakage）作为 vis prompt。
4. **Inference for base**：用 train set 1-shot 作为 vis prompt（或 sentinel zero vector + uniform fusion weight）。

**预期收益**：
- Novel：+5-10pp（如果 image encoder 真的学会了通用 feature↔prompt 对齐）
- Base：不变或 +0.5pp（multi-task learning 一般不会跌）

**关键风险**：
- Image encoder forward cost 增加（要为 visual prompt 跑一次 backbone）
- Modality dropout 比例没调好 → 退化成 text-only 或 vis-only

---

### Level 2（如果 Level 1 work，3-5 day 工程 + 8h GPU）

**思路**：完整 YOLOE-style 双路 class encoder + alignment loss。

**架构**：
```
text branch:
  5 attrs → text_encoder (frozen) → text_proj (learnable, D→D) → text_cls_vec
visual branch:
  N visual prompts → image_encoder (shared with detection backbone) →
    visual_proj (learnable, D→D) → visual_cls_vec
fusion (optional):
  fused = gate * text_cls_vec + (1-gate) * visual_cls_vec
```

**Loss**：
```python
L = L_cls(text_cls_vec)  # text-only path
  + L_cls(visual_cls_vec)  # vis-only path
  + L_cls(fused_cls_vec)  # fused path
  + lambda_align * ||text_cls_vec - visual_cls_vec||²
  + lambda_orth * dispersion(visual_cls_vec)  # 防止 vis collapse
```

**Modality dropout（3 模式）**：
- 33% text-only：vis branch masked
- 33% vis-only：text branch masked
- 34% both：完整 fusion

**优势 vs Level 1**：
- text_proj 也学，文本侧不再 frozen
- 更系统化的 alignment supervision
- 论文 §B（negative result）+ §D（multi-modal method）story 更完整

---

### Level 3（论文之外，长期，1-2 周）

完整 YOLOE：
- Prompt-free fallback mode（不需要 visual prompt 也能 detect "novel object"）
- Universal proposal generator
- 不在本论文 scope，但可以作为 future work outlook

---

## 2. Level 1 详细实现 plan

### 2.1 Critical files

```
新建：
- wedetect/models/backbones/multimodal_class_backbone.py
  → PseudoMultiModalClassBackbone (用 cache + 训练时调用 image_model)
- wedetect/datasets/transforms/visual_prompt_loader.py
  → SampleVisualPrompts transform（per-batch 采样 GT crops）
- config/wedetect_tiny_tct_ngc_dev30_mm_biomedclip_2gpu.py
- tools/build_visual_prompt_pool.py
  → 预算每个 base 类的 N=10 crops，存 cache

复用：
- tools/build_visual_prototype.py（novel 5-shot prototype 建好后可作为 inference 用）
- existing per-attr text emb cache（5 attrs）
- BiomedCLIP image encoder（如果实现 dual-encoder 路径）
```

### 2.2 Modality dropout 实现细节

```python
class PseudoMultiModalClassBackbone(BaseModule):
    def __init__(self, ..., vis_dropout_p=0.5, ...):
        self.vis_dropout_p = vis_dropout_p

    def forward(self, text, vis_prompts=None):
        # text: [B][C][5 attrs]  text strings
        # vis_prompts: [B, C, N, 3, H, W]  per-class GT crops (training)
        #              or [B, C, D]  cached vis embedding (inference)

        text_embs = self._lookup_text(text)  # [B, C, 5, D]

        if self.training and vis_prompts is not None:
            # Random drop visual modality per batch (not per sample —
            # ensures each forward is consistent across all classes)
            if torch.rand(1).item() < self.vis_dropout_p:
                vis_embs = None
            else:
                vis_embs = self._encode_vis(vis_prompts)  # [B, C, 1, D]
        elif vis_prompts is not None:
            # At inference, always use vis if provided
            vis_embs = self._lookup_or_encode_vis(vis_prompts)
        else:
            vis_embs = None

        if vis_embs is not None:
            all_attrs = torch.cat([text_embs, vis_embs], dim=2)  # [B, C, 6, D]
        else:
            all_attrs = text_embs  # [B, C, 5, D]

        return self._fuse(all_attrs)
```

### 2.3 Training pipeline 改造

```python
# config train_pipeline:
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # ... existing transforms ...
    dict(
        type='HierarchicalRandomLoadText',
        num_attr_types=5,
        ...
    ),
    # NEW: sample 1-3 GT crops per sampled class as visual prompts
    dict(
        type='SampleVisualPrompts',
        n_crops_per_class=1,
        crop_pool_path='data/visual_prompts/base30_crop_pool.pth',
        # crop_pool 是 build_visual_prompt_pool.py 预产出的 dict {class_str: List[crop_tensor]}
        # 训练时 dataloader 已经选定了 sampled classes，从 pool 里随机选 N 个 crop
    ),
    dict(type='PackDetInputs', meta_keys=(..., 'vis_prompts')),
]
```

### 2.4 验收标准

完成 Level 1 训练后 eval：

| 指标 | 目标 |
|---|---|
| Base 25-cls mAP | ≥ 0.327（THAF BiomedCLIP 基线）|
| Novel zero-shot avg mAP | **> 0.10**（vs THAF 0.052，target +5-10pp）|
| Phase 3.6 diagnostic 重跑 | novel image top-1 → base class **< 80%** (vs 99.2%) |
| novel mean cos to GT class | **> 0**（vs −0.178，目标 image feature 不再跟 novel 文本反向）|

如果 Phase 3.6 diagnostic 数字没改 → image encoder 还是只学到 base 方向 → modality dropout 比例需调，或 visual prompt supervision 信号太弱。

---

## 3. 数据 / GPU 时间预算

| 任务 | 工程量 | GPU 时间 |
|---|---|---|
| Level 1 数据 prep（base 30 GT crop pool 建库）| 1 天 | 1h GPU |
| Level 1 backbone + transform 实现 + sanity | 1 天 | 0 |
| Level 1 训练 | (启动) | 8h × 2 GPU |
| Level 1 eval suite（4 splits + base + Phase 3.6 重跑）| 0.5 天 | 1h |
| **Level 1 total** | **2.5 天 + 10h GPU** | |
| | | |
| Level 2 实现 + 训练 + eval | 3-5 天 | 8h × 2 GPU |

---

## 4. 决策树（看 Option 3a/4 结果定）

```
Option 3a / 4 结果（2026-05-12 早上）
  │
  ├─ 文本侧涨 base +1-3pp（用户直觉对，per-class 或 concat 击败 mean pool）
  │   → Phase 5 仍跑，因为 novel 还是不动
  │   → 论文 §A = 涨点的 text-side method
  │   → 论文 §D = Phase 5 Level 1/2
  │
  └─ 文本侧 ≈ THAF BiomedCLIP（mean pool 真天花板）
      → Phase 5 必须 work 才有论文 method §A
      → 风险更高：如果 Phase 5 也不 work，论文只剩 negative result + score fusion
      → 这时考虑 Level 2（更强干预）而非 Level 1
```

---

## 5. 关键风险与回退

### 风险 1：image encoder 不肯学 modality 通用对齐

**症状**：训练完 Phase 3.6 diagnostic 仍 99% novel→base。
**回退**：升级到 Level 2，加 explicit alignment loss。

### 风险 2：visual prompt 给的信号"too easy"，模型学到 shortcut

**症状**：base mAP 大涨（vis prompt = positive sample 几乎是 cheat），但 novel 不涨。
**回退**：减小 vis_dropout_p（更多 batch text-only），或换 sampling 策略（vis 和 GT 来自不同样本 / 不同 augmentation）。

### 风险 3：mmengine pipeline 改造复杂度爆炸

**症状**：vis_prompts 传到 forward 的 plumbing 改得太多，引入 bug。
**回退**：先做 inference-only 实验：用现有的 novel visproto cache + concat 当 6th attribute，**不重训**，eval 看是否涨。这是更便宜的初步验证。

---

## 6. 现成的可复用资源

- `tools/build_visual_prototype.py`：已有 N-shot visproto 构建逻辑，可改造成 base 30 GT crop pool builder
- `data/texts/tct_ngc_base30_visproto_train_clean.pth`：base 30 train-set 5-shot visproto（768d 旧）
- `data/texts/tct_ngc_novel_*_visproto_emb_clean.pth`：novel 5-shot prototype (test set leakage)，inference 可直接用
- 现有 PCW / ConcatProj backbone：fusion 模块可直接复用，只需加一个 vis_embs concat 路径

---

## 7. 立即可做的预热（不需要 GPU）

1. **写 `tools/build_visual_prompt_pool.py`**：从 train set 抽 N=10 crops per base class，存 cache
2. **写 `wedetect/datasets/transforms/visual_prompt_loader.py`**：训练时 SampleVisualPrompts transform
3. **写 `wedetect/models/backbones/multimodal_class_backbone.py`**：基于 PCW / ConcatProj 改造
4. **写 config `wedetect_tiny_tct_ngc_dev30_mm_biomedclip_2gpu.py`**

工作量 ~6-8h，可以在 Option 3a/4 训练期间完成。完成后等 GPU 释放（明早）一次 launch。
