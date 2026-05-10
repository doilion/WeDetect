# Phase 3 — Trainable Hierarchical Attribute Fusion (THAF) — In-Flight Status

**Date**: 2026-05-09 11:15 PDT
**Plan**: `~/.claude/plans/zazzy-hugging-pearl.md` — approved
**Status**: Phase 3a (XLM-R) + Phase 3b (BiomedCLIP) **both training in parallel**, ETA ~9h each

---

## Code shipped

### New modules

| 路径 | 说明 |
|---|---|
| `wedetect/datasets/transformers/hierarchical_mm_transforms.py` | `HierarchicalRandomLoadText` (训练) / `HierarchicalLoadText` (推理), num_attr_types=5 |
| `wedetect/models/backbones/hierarchical_mm_backbone.py` | `HierarchicalXLMRLanguageBackbone` (encoder + fusion, 继承父类保持 ckpt 权重路径), `PseudoHierarchicalXLMRLanguageBackbone` (cache 加速版) |
| `wedetect/models/backbones/biomedclip_backbone.py` | `HierarchicalBiomedCLIPLanguageBackbone` (open_clip + fusion 512d), `PseudoHierarchicalBiomedCLIPLanguageBackbone` (cache 版) |

Fusion module 共享 `_build_fusion_module / _fuse_attr_embeds / _init_fusion_weights` helpers in `hierarchical_mm_backbone.py`：
- attr_type_embed (5, D) + fusion_query (1, 1, D) + cross_attn (8 heads) + norm1/2 + output_proj (D→4D→D) + alpha
- 参数量：XLM-R 768d **7.09M**, BiomedCLIP 512d **3.15M**

### New tools

| 路径 | 用途 |
|---|---|
| `tools/build_hierarchical_class_text.py` | dict 5-attr JSON → list-of-list training file + 4 per-split eval files |
| `tools/build_per_attr_emb_cache.py` | 预缓存 195 unique attr 字符串 embedding（XLM-R 768d / BiomedCLIP 512d）|
| `tools/build_hier_class_embeddings.py` | 训练后跑 fusion forward 一次，生成 39-class `{class_name: Tensor[D]}` 缓存供 PseudoLanguageBackbone eval 用 |
| `tools/eval_thaf_all_splits.sh` | 一键编排：build cache → base eval → 4 splits novel eval → 汇总 |

### New configs

| 路径 | 说明 |
|---|---|
| `config/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu.py` | Phase 3a：XLM-R 冻结 + 5-attr 5K cross-attention fusion (768d) |
| `config/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu.py` | Phase 3b：BiomedCLIP 冻结 + 同 fusion (512d, head 重训) |

---

## 关键发现：BiomedCLIP cache geometry 强烈优于 XLM-R

| 度量 | XLM-R 768d | BiomedCLIP 512d | Δ |
|---|---:|---:|---|
| 单 attr 字符串 pairwise cos (mean) | 0.89 | **0.55** | -0.34 ✅ |
| 单 attr pairwise cos (max) | 0.998 | 0.976 | -0.022 |
| Per-class attr_mean (5-attr 平均后 L2) pairwise cos (mean) | 0.96 | **0.82** | -0.14 ✅ |
| Per-class attr_mean (max) | 0.995 | 0.967 | -0.028 |
| Novel↔Novel (9 类) pairwise cos (mean) | 0.87 | **0.79** | -0.08 ✅ |
| Novel↔Novel max cos | 0.99 | **0.95** | -0.04 ✅ |

**解读**：
- BiomedCLIP 在医学 fine-grained vocabulary 上给的 embedding 几何远比 XLM-R 分散
- 这直接验证了 Phase 2 "XLM-R 是 encoder 瓶颈" 的假设
- BiomedCLIP novel↔novel max cos 0.95 在 train-separable 阈值附近（业界经验 0.92-0.95）；XLM-R 0.99 早就饱和

---

## 训练状态（in-flight）

### Phase 3a (THAF + XLM-R, GPU 0+1)

启动：`2026-05-09 10:54:25`
ETA：~10h

| 时间点 | iter | loss | loss_cls | loss_bbox | loss_dfl | grad_norm |
|---|---:|---:|---:|---:|---:|---:|
| ep1 [50/3238] | 50 | 368.33 | 282.87 | 39.57 | 45.89 | 8607 |
| ep1 [1100/3238] | 1100 | 188.84 | 121.54 | 30.52 | 36.77 | 2715 |

对比 dev30 baseline 同 iter：
- ep1 [50]: baseline loss=409.88 cls=322.48 → THAF -10% on cls ✅
- ep1 [100]: baseline loss=274.71 cls=203.24

THAF + XLM-R 起步比 baseline 略好；cls loss 在 ep1 后半段进入 ~120 区间，跟 baseline 进度一致。

### Phase 3b (THAF + BiomedCLIP, GPU 2+3)

启动：`2026-05-09 11:09:22`
ETA：~8h

| 时间点 | iter | loss | loss_cls | loss_bbox | loss_dfl | grad_norm |
|---|---:|---:|---:|---:|---:|---:|
| ep1 [50/3238] | 50 | 396.14 | 313.08 | 39.65 | 43.41 | 11446 |
| ep1 [250/3238] | 250 | 220.34 | 153.30 | 30.95 | 36.09 | 3438 |

预热阶段比 Phase 3a 略高（grad_norm 11446 vs 8607）但稳定，无 loss spike。每 iter 时间 0.71s vs Phase 3a 1.26s — BiomedCLIP 路径更快（512d head 计算量小）。

### 几何回顾 vs 训练初始化

随机 fusion init 时的 class vectors（all 30 base classes，pseudo backbone forward）：
- Phase 3a (XLM-R) class-class cos: mean 0.98, max 0.995 — 严重坍缩
- Phase 3b (BiomedCLIP) class-class cos: mean 0.82, max 0.97 — 自然分散

→ Phase 3b 训练应该更稳（实测：5-iter smoke 没出现 1M loss spike，Phase 3a 出现过）

---

## 实施过程中的 4 个修正点（vs 计划）

1. **LR 多项式**：原计划 fusion params 用 10× LR mult（matching YOLO-World-Medical），实测在 WeDetect-tiny 上引发反复 loss spike 到 ~1M。改为 1×（仅保留 `decay_mult=0` 给 fusion_query / alpha）。
2. **权重映射**：`HierarchicalXLMRLanguageBackbone` 改成**继承** `XLMRobertaLanguageBackbone`（不是组合），保持 `model.*` / `head.*` 参数名 → 旧 dev30 ckpt 直接 strict-load 父类参数，fusion 模块按 missing keys 留 random init ✓
3. **预缓存**：训练加速从计划值 5×/iter 提到了实测 ~5×（BiomedCLIP 0.71s/iter, XLM-R 1.26s/iter），符合预期
4. **Eval cache key 协议**：用 class_name 作 key，借助 `PseudoLanguageBackbone.forward_text` 的 `text.split("/")[0]` 行为（class name 无 "/"）— 现有 eval_novel_split.py 不需要修改

---

## 下一步（等训练完成后立即做）

1. **预期 GPU 时间**：~9-10h，预计今晚 ~21:00 PDT 完成两条线
2. **训练完成后，依次跑两个 eval**：
   ```bash
   bash tools/eval_thaf_all_splits.sh xlmr
   bash tools/eval_thaf_all_splits.sh biomedclip
   ```
3. **写 Phase 3 完整对比报告**：`docs/tct_ngc_phase3_thaf_eval_<date>.md`，含完整 ablation 表

---

## 论文级 ablation 表（Phase 3 完成后填）

```
                                         | Base mAP | Novel main_3 | pseudo_2 | hard_4 | full_5 | Avg novel
v2 baseline (XLM-R, single PSC)          |  0.283   |   0.155      |  0.138   |  0.054 |  0.065 |  0.103
score fusion (XLM-R)                     |  0.283   |   0.167      |  0.165   |  0.088 |  0.079 |  0.125
THAF + XLM-R (768d, 5-attr fusion)       |    ?     |     ?        |    ?     |    ?   |    ?   |    ?
THAF + BiomedCLIP (512d, 5-attr fusion)  |    ?     |     ?        |    ?     |    ?   |    ?   |    ?
THAF + BiomedCLIP + score fusion         |    ?     |     ?        |    ?     |    ?   |    ?   |    ?
```

预期 BiomedCLIP+THAF 是 paper headline；XLM-R+THAF 作为"换 encoder 的必要性证据"；score fusion 还可以 stack 上去。
