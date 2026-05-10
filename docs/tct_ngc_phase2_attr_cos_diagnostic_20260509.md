# Phase 2.1 cos heatmap 诊断 — 5 属性 × 4 聚合策略

**日期**：2026-05-09
**目标**：在重训之前，先用半天验证"hierarchical attribute training（组件 A）能否破解 XLM-Roberta 在 fine-grained cytology 上的 cos collision"。
**输入**：`data/texts/tct_ngc_fullnames_39_attr.json`（30 base + 9 novel × 5 属性，LLM 生成预版本）
**编码器**：XLM-Roberta（dev30 best ckpt 中提取的 text branch，未微调）

---

## TL;DR

**❌ 组件 A 在 XLM-R 上 DEAD**。4 种聚合策略 novel↔novel max cos 全部 > 0.97，远高于 0.92 的训练可分阈值。结构化 prompts 把 cos 比单 PSC prompt 降低 ~2.5 点，但**XLM-R 不是医学专用**，所有 fine-grained cytology 描述都被压扁到同一片高 cos 区域。

**下一步**：跳到 item 15（BiomedCLIP / PubMedBERT swap）。

---

## 4 种策略对比

| 策略 | 维度 | novel↔novel max cos | novel↔base max cos | 决策 |
|---|---:|---:|---:|---|
| **v2 baseline**（单 PSC/MAL-S prompt）| 768 | **0.996** | 0.967 | — (基线) |
| **concat**（5 attrs 拼接）| 3840 | 0.971 | 0.973 | ❌ |
| **sum**（5 attrs 简单求和）| 768 | 0.993 | 0.992 | ❌ |
| **weighted-sum**（distinguish 0.4 / morph 0.3 / diag 0.15 / organ 0.1 / bg 0.05）| 768 | 0.991 | 0.990 | ❌ |
| **only-distinguish**（仅 key_distinguishing_feature）| 768 | **0.971** | 0.973 | ❌ |

阈值参考：
- `<0.85` → 文本端可分，可以用现有 encoder 重训组件 A
- `0.85-0.92` → 文本端边缘，重训可能有边际收益
- `>0.92` → 文本端饱和，**必须换 encoder**

**结果**：所有策略 max cos ∈ [0.971, 0.993]，**全部超过 0.92 阈值**。

---

## 解读

### 1. 结构化 prompts 有微小收益但远不够

| 改进 | 收益 |
|---|---|
| v2 单 prompt → only-distinguish | -0.025（0.996 → 0.971）|
| v2 单 prompt → concat | -0.025（0.996 → 0.971）|
| v2 单 prompt → sum / weighted | ≈ 0 |

最好的策略（concat / only-distinguish）只把 cos 降低 ~2.5 点。这说明：
- "key_distinguishing_feature" 字段确实带了一些差异信息
- 但 XLM-R 把它跟其他字段一起编码后还是被一般化的医学 jargon 主导
- **额外 4 个属性几乎没贡献**（only-distinguish 与 concat 持平）

### 2. sum 比 weighted 还差，concat 跟 only-distinguish 持平

- **sum 0.993** > **weighted 0.991** > **concat = only-dist 0.971**
- sum 把 5 个高度相似的向量直接相加，等于把医学 jargon 累加 5 倍 → 反而更靠拢
- weighted 把 distinguish 加权 0.4，效果接近"主要看 distinguish"，但其他字段 0.05-0.30 的拉拢仍然在
- **concat 不在共享 768-dim 空间**（3840-dim），cos 值是 5 个独立子空间的均值，能看到 distinguish 子空间的差异
- **only-distinguish** 直接只用 distinguish 字段，跟 concat 在 distinguish 子空间的判别力等价

### 3. novel↔base 的 collision 模式

- v2 baseline：novel→base 最相似往往是 NILM / Negative samples（cos 0.93-0.97）
- 5 属性所有策略：novel→base 还是 NILM / 同 organ base 类（cos 0.95-0.99）
- **没有任何策略让 novel 类远离 base NILM 类**，说明 XLM-R 对"恶性 vs 良性"的语义并不敏感

---

## 失败根因

XLM-Roberta 是**通用多语言编码器**，没经过医学预训练：
1. 各种 cytology / 病理描述都用相似的临床词汇（cells / nuclei / chromatin / cytoplasm / ...）
2. 编码空间高度 anisotropic（一个共享方向占主导，所有医学描述都聚在那个方向附近）
3. 真正的判别信息藏在词汇细节（"keratinization" vs "salt-and-pepper chromatin"）—— XLM-R 不区分

这是文献里反复证明过的现象（Mu et al. 2017 anisotropy；Ethayarajh 2019 BERT geometry），DEAD-1/DEAD-2 已经在 inference 端验证过 anisotropy 救不了。

**结构化 prompts 解不了 encoder bottleneck**：你给它再好的医学描述，它都用通用的方式编码，把所有 cytology 内容拍成一坨。

---

## 决策

按 Phase 2 kill criteria：
- **所有 4 种策略 cos > 0.92 → 组件 A 在 XLM-R 上死** ✅ 命中

**Pivot**：
1. **保留 `tct_ngc_fullnames_39_attr.json`** 作为下一步 BiomedCLIP 的输入（结构化数据已经齐了）
2. **下一步走 item 15**：用 BiomedCLIP 或 PubMedBERT 重新编码同样的 5 属性，看 cos 是否能跌破 0.85
3. 如果 BiomedCLIP 上 cos < 0.85 → 走 Phase 3 重训（用 BiomedCLIP encoder + 5 属性 hierarchical training）
4. 如果 BiomedCLIP 上 cos 还是 > 0.92 → 文本端在医学 fine-grained 上**根本不可能**，只能走 visual exemplar prototype（item 13/14）+ score fusion 路线

---

## 论文向收益

虽然 Phase 2 没救活组件 A，但产出了**论文级 negative result**：
- "We show that even careful 5-attribute structured prompts (organ + diagnostic code + morphology + immunoprofile + key distinguishing feature) cannot reduce novel↔novel cos below 0.97 in XLM-R, providing direct evidence for the **encoder-level bottleneck hypothesis** of medical OVD"
- 这给后续提 BiomedCLIP swap 一个非常强的 motivation
- ablation 表里"v2 baseline / concat / sum / weighted / only-distinguish"5 行直接进论文 §A.2

---

## 产物

| 文件 | 用途 |
|---|---|
| `data/texts/tct_ngc_attr_{concat,sum,weighted-sum,only-distinguish}_emb.pth` | 4 种聚合策略的 39 类 embedding（XLM-R 版）|
| `data/texts/tct_ngc_attr_base30.json` / `tct_ngc_attr_novel9.json` | base/novel 类名 list-of-list（cos heatmap 工具输入）|
| `tools/build_attr_text_embeddings.py` | 5 属性 4 策略编码器（脚本可重用：换 checkpoint 就能换 encoder）|
| `docs/figures/dev30_taxonomy_refactor_20260508/diagnostics/attr_<strat>/` | 4 个策略各自的 heatmap PNG + 9×9 finegrained PNG + summary |

---

## 重生成命令

```bash
# 重新编码 5 attr × 4 strategies（用同一 checkpoint）
python tools/build_attr_text_embeddings.py \
    --attr-json data/texts/tct_ngc_fullnames_39_attr.json \
    --checkpoint checkpoints/wedetect_tiny.pth

# 重新画 4 strategy 的 cos heatmap
for strat in concat sum weighted-sum only-distinguish; do
    python tools/analyze_novel_prompt_cos.py \
        --base-emb data/texts/tct_ngc_attr_${strat}_emb.pth \
        --base-json data/texts/tct_ngc_attr_base30.json \
        --novel-embs data/texts/tct_ngc_attr_${strat}_emb.pth \
        --novel-jsons data/texts/tct_ngc_attr_novel9.json \
        --out-dir docs/figures/dev30_taxonomy_refactor_20260508/diagnostics/attr_${strat}
done
```
