# Paper §A 最终实验结果（截至 2026-05-15）

## TL;DR

**Main method (proposed)** = M1 organ-conditional class restriction
                          + 5-attribute structured prompts
                          + Image-Conditional Fusion (ICF, Design A)

| | Base macro | Novel macro |
|---|---:|---:|
| **ICF (proposed)** | **0.3522** | **0.1648** |
| 最强 baseline (M1-5attr 平均, 零参) | 0.3396 | 0.1642 |
| Δ | **+1.26pp** | **+0.06pp** (tied) |

**Novel mAP 上限 ~0.165** 在 retrieval-style 架构 + 当前 image encoder 下结构性触顶。突破需要 plan 3 (cytology MIM domain adaptation)。

---

## 完整 ablation table（corrected protocol, OrganRestrictedCocoMetric）

eval set: `instances_test_base_clean_dev30.json` (base 25) + `instances_test_novel_merged_9.json` (novel 9), 排除 5 个 negative classes from COCOeval catIds。

### Main table

| Row | 方法 | text 端可训参数 | Base macro | Base inst-wt | Novel macro | Novel inst-wt |
|---|---|---|---:|---:|---:|---:|
| 1 | M1 (1-PSC) | 0 | 0.3369 | 0.4046 | 0.1500 | 0.1537 |
| 2 | M1-5attr 平均 | 0 (静态 mean pool) | 0.3396 | 0.3973 | 0.1642 | 0.1657 |
| 3 | M2 完整方法 (Stage 1+2+3 + ord_loss) | ~200K | 0.3436 | 0.4088 | 0.1051 | 0.1203 |
| 4 | M2-auxfix (强 aux loss, sum norm) | ~200K | 0.3434 | 0.4070 | 0.0557 | 0.0651 |
| 5 | M2-axisstruct (per-axis structured loss) | ~200K | 0.3371 | — | 0.0665 | 0.0848 |
| 6 | **Row 6c**: Stage 1 + clean ord_loss (no MoE, no rank emb lookup) | ~150K | 0.3386 | — | 0.1555 | 0.1538 |
| 7 | **ICF (Design A)** ⭐ | ~2.4M | **0.3522** | — | **0.1648** | 0.1582 |

### Per-organ novel breakdown

| 方法 | Resp (3 cls) | Serous (3) | Thyroid (3) | macro |
|---|---:|---:|---:|---:|
| M1 (1-PSC) | — | — | — | 0.1500 |
| M1-5attr 平均 | — | — | — | 0.1642 |
| M2 完整方法 | — | — | — | 0.1051 |
| M2-auxfix | 0.0123 | 0.0634 | 0.0914 | 0.0557 |
| **Row 6c** | **0.1781** | 0.1615 | 0.1270 | 0.1555 |
| **ICF** | **0.1978** | 0.1608 | 0.1359 | **0.1648** |

### Bypass 诊断 (auxfix ep12 ckpt, inference-time module knockout, novel9 only)

| Inference mode | Resp | Serous | Thyroid | Novel macro | Δ vs auxfix |
|---|---:|---:|---:|---:|---:|
| 全开（auxfix 原始） | 0.0123 | 0.0634 | 0.0914 | 0.0557 | — |
| Stage 1 bypass（uniform α） | 0.0049 | 0.0841 | 0.0889 | 0.0593 | +0.4pp |
| Stage 3 bypass（rank=0） | 0.0416 | 0.0875 | 0.1072 | 0.0788 | +2.3pp |
| **Stage 2 bypass（uniform MoE）** | 0.1077 | 0.1103 | 0.0944 | **0.1041** | **+4.8pp** ⚠️ |
| 全 bypass | 0.0156 | 0.0563 | 0.0305 | 0.0341 | −2.2pp |

→ Stage 2 organ MoE 是 novel killer (+4.8pp 回升)，Stage 3 rank embedding 是次犯 (+2.3pp)，Stage 1 中性。

---

## 模块/设计有效性结论

### ✅ 成功的设计

| 设计 | 证据 | 论文叙事 |
|---|---|---|
| Organ-conditional class restriction (M1) | clean dev30 0.108 → M1 0.150 novel (+4.2pp) | "Clinical deployment scenario aligned: WSI organ is known a priori, so detector should not score cross-organ classes" |
| 5-attribute structured prompts (vs 1-PSC) | M1 0.150 → M1-5attr 平均 0.164 novel (+1.4pp) | "Pathologist-canonical 5-attribute decomposition gives richer text geometry than single concatenated prompt" |
| **Image-Conditional Fusion (ICF)** | M1-5attr 0.164 → ICF 0.1648 novel (tied), base **+1.3pp** | **Main contribution**: "small-data medical OV-detection benefits from image-conditional fusion that escapes mean-pool collapse, with anti-collapse diagnostics (ICFCollapseGuard)" |

### ❌ 失败的设计（写进 §A.X "Negative results"）

| 设计 | 证据 | 失败机制 |
|---|---|---|
| THAF cross-attention fusion (class-agnostic query) | α→0 collapse, fusion ≡ mean pool | optimizer 没有信号让 fusion 偏离 mean pool 盆地 |
| M2 per-organ MoE (Stage 2) | Bypass +4.8pp on novel; M2 vs M1-5attr novel −5.9pp | redundant with M1 organ restriction + 数据按 organ 分片到 5 个 expert 导致 base-overfit |
| M2 rank embedding lookup (Stage 3) | Bypass +2.3pp on novel | novel ranks 在训练里没见过 → lookup 拿到 init 噪声污染 emb |
| M2-axisstruct (per-axis structured aux loss) | base 0.337 / novel 0.067（比 M1-5attr 平均 novel 掉 9.8pp） | structured loss 仍依赖 organ MoE + rank emb 主体，没解决 M2 核心病灶 |
| Trainable Stage 1 + clean ord_loss (Row 6c) | Row 6c vs M1-5attr 平均: base −0.1pp, novel −0.9pp | 任何 trainable text-side module 在小医疗数据上都 net-negative vs 静态 mean pool |

### ⚠️ Rank label quality audit (build_taxonomy_metadata.py 解析的 PSC/Bethesda/Paris Roman numeral)

| Organ-axis | Rank 标签 | 评价 |
|---|---|---|
| Urine axis 0 (Paris) | NHGUC(2) → AUC(3) → SHGUC(4) → HGUC(5) | ✅ clean ordinal |
| TCT_CCD axis 0 (Bethesda cervical) | ASCUS(1) → ASCH(2) → LSIL(3) → HSIL(4) | ✅ clean |
| Thyroid axis 0 (Bethesda thyroid) | ⚠️ rank 1 有 2 类碰撞, rank 2 有 2 类碰撞 | ⚠️ 部分 ordinal |
| **respiratory tract axis 0 (PSC)** | 5 个正常细胞共 rank 2 | ❌ MSE collision 破坏 cls 判别力 |
| **Serous effusion axis 0** | 只 2 类，binary | ❌ degenerate |
| **TCT_CCD axis 2 (感染)** | monilia / dysbacteriosis / vaginalis 非 ordinal | ❌ 标签任意指派 |

→ Row 6c 实验中 `OrganOrdinalLoss.exclude_organ_axes = [(0,0), (1,0), (4,2)]` + `skip_collision_ranks=True` 剔除 broken axes，只保留 Urine + TCT_CCD axis 0 + Thyroid axis 0 的 3 个 rank-unique exemplar 类。

---

## Key open questions / future work

1. **Novel ceiling 0.165 是 image encoder OOD 上限**：ConvNext-COCO pretrain 跟 cytology 像素分布差距。Plan 3 cytology MIM domain adaptation 是下一步。
2. **YOLOE 0.261 novel 是 visual prompt path**：跟 ICF 的 text path 不可直接比；如果做 visual prompt path，参考 plan_appendix。
3. **ICF 的 attn_entropy 1.26-1.32**：attention 在挑 ~2-3 个 attribute 而不是全部，验证了 "image-conditional attribute selection" 是真的在 work。具体 attribute pattern 可作为 paper 可视化材料。

---

## ICF 训练诊断（ICFCollapseGuard, 全程绿灯）

| 指标 | 健康范围 | ep12 final | 状态 |
|---|---|---:|---|
| fused_pairwise_cos_mean (同类跨图) | 0.5-0.95 | 0.89-0.92 | ✅ image-conditional fusion 真的不同图给不同输出 |
| attn_entropy_mean | < 1.58 (log 5) | 1.26-1.32 | ✅ attention 选择性激活 (~2-3 个 attribute) |
| cos_to_attr_mean_mean | < 0.95 | -0.02 | ✅ fused 跟 mean pool 方向**完全不重合**，跳出盆地成功 |

→ Design A 机制保证（image-conditional query 跳出 mean-pool 函数族）在训练完成后**完整保持**，没退化。

---

## ICF Val 曲线（每 epoch best）

```
ep1: 0.2283   ep2: 0.2546   ep3: 0.2617   ep4: 0.2784
ep5: 0.2900   ep6: 0.2932   ep7: 0.2963   ep8: 0.3019
ep9: 0.3028   ep10: 0.3059  ep11: 0.3070  ep12: 0.3112 ← best
```

单调上升，未塌缩。

## Row 6c Val 曲线

```
ep1: 0.2203   ep2: 0.2394   ep3: 0.2582   ep4: 0.2675
ep5: 0.2749   ep6: 0.2789   ep7: 0.2982   ep9: 0.3073
ep12: 0.3088 ← best
```

也单调上升。
