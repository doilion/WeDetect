# TCT_NGC dev30 后续 TODO

本 TODO 由 `docs/tct_ngc_dev32_disjoint_baseline_report_20260508.md` §9 + 用户讨论汇总。每项标注前置依赖、工程量、预期收益。

---

## ⛔⛔⛔ 已实验确认的死路 — 不要再试 ⛔⛔⛔

> **这些方向我们已经实测失败，记下来避免以后重复浪费时间。**

| # | 死路 | 实测结果 | 失败根因 |
|---:|---|---|---|
| **DEAD-1** | **推理端文本 ensembling**（CuPL/CLIP-style 多 prompt 平均）| 所有 4 个 novel split mAP **跌 60-100%**（v3 / v3pure 实验） | image encoder 训练时被钉死在某个特定 prompt 方向，平均把向量拉到无人区 |
| **DEAD-2** | **推理端 anisotropy reduction**（mean centering / whitening）| 同上，全军覆没 | 破坏了 contrastive head 期望的 raw embedding scale |
| **DEAD-3** | **Per-variant L2-normalize 后再平均** | 同上 | 跟 raw 内积架构不兼容 |
| **DEAD-4** | **Raw text+visproto 单次 inference 二元路由（binary fusion）** | main_3 Breast **0.454 → 0.000**；text 类被 visproto 类系统性挤死 | text(XLM-R) 与 visproto(cls_preds) 几何不同，单 inference 内 visproto 对图像 cosine 天然高 |
| **DEAD-5** | **Procrustes 对齐后的 visproto + text 单次 inference 二元路由（calfused）** | text 类救回（main_3 Breast 0.447 ✅）但 visproto 类全死（Resp-Adeno **0.095 → 0.000**，Thyroid-Sus **0.079 → 0.009**）；4 splits **全部低于 score fusion** | **R 不是 novel-transferable**：base 类对(text↔vis) 拟合 cos −0.30 → 0.97 ✅；但 **calibrated visproto SOLO eval（main_3）mAP = 0.005**（vs raw visproto 0.042，跌 87%）—— 证实 R 旋转 novel visproto 后落到 text 空间错位置，不是单 inference 混合的问题。Procrustes 对 novel-class transfer **不 generalize** |

**核心教训**：
1. WeDetect 的 PseudoLanguageBackbone 用**冻结缓存 + raw 内积**，**任何 inference-only 后处理都救不了 novel zero-shot**。
2. **同一次 inference 内混 text + visproto 不可行**（DEAD-4/5 双向都失败）—— 几何不匹配靠 rotation 救不了；唯一 work 的是 **post-hoc score fusion**（两次独立 inference + per-class 合并预测）。

要改进必须从下面三条路之一走：

1. **post-hoc score fusion**（item 13.5，已验证 ✅，所有 4 splits 上都 ≥ 单源最优）
2. **换 text encoder**（item 15，需要重训）
3. **train-time prompt augmentation / hierarchical attribute training**（item 17/19，需要重训）

---

## 排序原则

按 ROI（不靠扩数据/扩 case 的杠杆排前）：

| 序号 | 项 | 前置 | 工程量 | 预期 |
|---:|---|---|---|---|
| 1 | Loss 加权 | dev30 已训完 | 重训 1 次 (~8h) | SHGUC/AUC +5-10% |
| 2 | Stain normalization | 无 | 1 天 + 重训 | cohort-reversal 类 +5-10% |
| 3 | Strong augmentation | 无 | 重训 1 次 | +3-5% (尤其小 cohort) |
| 4 | Test-time augmentation (TTA) | 无 | 0.5 天，零训练 | 小目标 +5%，整体 +2-3% |
| 5 | Self-supervised pretrain | 较多无标注 cytology 图 | 1-2 周 | 全类 +5-10% |
| 6 | Pseudo-labeling | 同上 | 1 周 | 取决于无标注数据规模 |
| 7 | TCT_CCD provenance 修复 | 运维 | 看运维 | 让 TCT_CCD 4 个弱类 AP 数字可信 |
| 8 | EC vs normal 合并（dev30 验收后再决定） | dev30 跑完 | 1 天 + 重训 | 待 dev30 验证 |
| 9 | Hier_v2 层级训练 | TCT_CCD provenance | 数周 | 嵌套类（ASC-H/HSIL）显著 |
| 10 | Prompt 改写（cos > 0.97 同胞对，本轮 NHGUC 之外的 5 对） | 无 | 1 周（含重训） | 单类 +5-10% |
| 11 | 双层评估 multi-organ screening | 临床指标工具完成 | 0.5 天 | 报告增强 |
| 12 | **Novel text-encoder cos heatmap** (诊断文本端是否瓶颈) | 无 | 半天 | 决定 13-15 走哪条 |
| 13 | **Visual exemplar prototype** (image-as-class-vector, 不重训) | dev30 ckpt | 1-2 天 | novel mAP +0.10-0.30 |
| 14 | **Text + visual dual-anchor fusion** (α-加权融合) | 13 完成 | 0.5 天 | base 不掉 + novel 涨 |
| 15 | **Text encoder 换 BiomedCLIP / PubMedBERT** | 架构改造 | 1-2 周 | fine-grained 类文本可分性质变 |
| 16 | **Hierarchical 2-stage detection** (粗类 OV + 细类 few-shot prototype) | 13 验证 | 2-3 周 | 临床可解释性 ↑ + novel 子型可达 |
| 17 | **Train-time prompt augmentation** (每 base 类 5-10 variants 随机采样) | 无 | 1 次重训 (~8h) | image encoder 学 prompt cloud |
| 18 | **Novel 评估按"粗类 vs 细类子型"拆分**报告 | 无 | 0.5 天 | 数字诚实，定位瓶颈 |
| 19 | **4 层次属性 hierarchical prompt training**（organ + tier + subtype + morphology）| 病理医生 curation 4-attr 卡片 | 数据 2-3 天 + 1 次重训 | 治本，跟 17 合并实现 |

> 死路警告已移到 README 顶部（DEAD-1/2/3）。

---

## 1. Loss 加权（focal / class-balanced / hard-negative mining）

**目的**：合并阴性后正负实例比仍 ~25:1，模型损失被多数类主导。引入针对稀有阳性类的训练信号增强。

**方案**：
- (a) Focal loss γ=2 替换 CrossEntropyLoss in `bbox_head.loss_cls`（最便宜）
- (b) Class-balanced sampling：dataset wrapper 让稀有类按 effective number 上采样
- (c) Hard negative mining：每 batch 显式挑 NILM/SHGUC 边界细胞做 contrastive

**Critical files**：
- `config/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu.py` — `model.bbox_head.loss_cls`
- 可能 `wedetect/datasets/wecoco.py` — 如果上 (b)

**Verification**：dev30 baseline 跟加权后 dev30 在 cost-weighted M2 metric 上对比。

---

## 2. Stain normalization

**目的**：cohort-reversal 类（Thyroid-NS val 0.388 / test 0.198 等）的 val/test 差距大概率来自染色批次差异。归一化染色后差距应收敛。

**方案**：上 Reinhard 或 Macenko stain norm 到 dataloader 的 LoadImageFromFile 后。

**Critical files**：
- `wedetect/datasets/transforms/` — 新写 StainNormalize transform
- `config/.../*disjoint*.py` 的 `train_pipeline` / `test_pipeline` 加入

**前置**：选择 reference image (canonical stain)。

---

## 3. Strong augmentation

**目的**：用同一批 case 制造"虚拟 case"，缓解数据多样性不足。

**方案**：
- Mosaic（YOLOv5 已有，dev32 配置里没启用）
- ColorJitter histo-aware（在 PhotoMetricDistortion 上扩 HSV 范围）
- GridMask / Cutout

**Critical files**：`config/.../*.py` 的 `train_pipeline`。

---

## 4. Test-time augmentation (TTA)

**目的**：零训练成本提升小目标类（Lymphocyte / Neutrophil）。

**方案**：写 `tools/infer_tta.py` 跑同一 ckpt 多次：原图 + flip + scale 0.83/1.0/1.2，detection NMS 合并。

---

## 5. Self-supervised pretrain backbone

**目的**：从无标注 cytology 图学到 cytology-specific 表征，提升所有类基础精度。

**方案**：MAE / DINO / iBOT 在大规模无标注 cytology image 上跑 SSL pretrain → fine-tune dev30。

**前置**：拿到至少 100k 张无标注 cytology 图。

---

## 6. Pseudo-labeling

**目的**：用当前模型给无标注图打高置信度伪标签，等价于扩 ann。

**方案**：confidence > 0.5 的 detection 作为伪 GT，加入训练集。第二轮训练时双过滤（confidence + consistency under augmentation）。

**前置**：同 5；以及一个稳定的 baseline（dev30）。

---

## 7. TCT_CCD provenance 修复

**目的**：当前 TCT_CCD 4 个弱类（asch / monilia / ec / vaginalis）的 AP 数字不能用于泛化分析（patient-disjoint 性质未验证）。

**方案**：拿原始数据集的 path 信息（含 WSI / case ID），重做 patient-disjoint split。

---

## 8. EC vs normal 合并

**目的**：dev30 跑完之后，如果合并 NHGUC 收益明显（验证假设），考虑同样合并 TCT_CCD-ec / normal。

**前置**：dev30 实验结果出炉 + 收益归因清晰。

---

## 9. Hier_v2 层级训练

**目的**：对"语义嵌套"类（ASC-H/HSIL/ASC-US 三层、AUC/SPTC/PTC 三层）显式建模层级。

**前置**：TCT_CCD provenance 修复（否则 ASC-H/HSIL 数字不可靠）。

---

## 10. Prompt 改写（剩余 5 对 cos > 0.97）

dev30 合并 NHGUC 解决了 SHGUC↔Negative 一对。剩余 5 对 cos > 0.97：
- Urine-AUC ↔ Urine-HGUC：0.973
- Thyroid-NS ↔ Thyroid-PTC：0.972
- respiratory-Lymphocyte ↔ respiratory-Neutrophil：0.980
- Thyroid-AUC ↔ Thyroid-NS：0.969
- TCT_CCD-asch ↔ TCT_CCD-hsil_scc_omn：0.964

**方案**：参考 PSC / Bethesda / Paris 标准，每对 prompt 重写让 cos < 0.92。重新生成 emb cache + 重训。

---

## 11. 双层评估 multi-organ screening

**目的**：当前 M1 image-level 是单 binary（是否含任何阳性）。可以分器官报：respiratory / serous / thyroid / urine / TCT_CCD 各自的 screening AUROC，对临床部署更有用。

**方案**：扩 `tools/eval_clinical_metrics.py` 加 per-organ 切分输出。

---

## 12. Novel text-encoder cos heatmap（**最优先 — 诊断**）

**目的**：在投入任何 13-17 的工程之前，先用半天确认"文本端到底是不是瓶颈"。

**方案**：把 9 个 novel 类 + 25 个 base 类的 prompt 用 XLM-Roberta 编码，算两两 cosine，画 34×34 heatmap。

**期望读数**：
- 如果 fine-grained 子型（Bethesda V vs VI、Adeno vs SCC vs Small Cell）的 cos > 0.90，说明 XLM-Roberta 在医学语义上**根本分不开**，跳到 item 15（换 BiomedCLIP）
- 如果 cos < 0.85 但 novel mAP 仍然差，说明问题在 image encoder 端，跳到 item 13（visual prompt）
- 如果 cos 在 0.85-0.95 区间，两条路都值得试

**Critical files**：新写 `tools/analyze_novel_prompt_cos.py`。

**Verification**：heatmap PNG + 一行结论"Bethesda V/VI cos = X.XX → 走 visual prompt"。

---

## 13. Visual exemplar prototype（**最优先 — 实验**）

**目的**：放弃用文本描述 fine-grained 子型；改用"几张样本图的视觉特征均值"当类向量。绕开医学文本-视觉 gap。

**方案**：
1. 从每个 novel 类的 GT 标注里**留出 5 张 bbox crop**（标记为 prototype set，剩余作 evaluation set）
2. 用 dev30 best ckpt 的 image encoder + ROI pool 对每个 crop 提 768 维特征
3. 类内取平均 → 视觉原型向量
4. 替换 PseudoLanguageBackbone 的 cache_bank 里那个类的向量
5. 跑 evaluation 对比 v2 文本 prompt baseline

**为什么会成功**（v3 失败的对偶）：
- contrastive head 是 image_feat × class_feat 内积，class_feat 来自图像还是文本它**不在乎**，只要维度匹配
- 视觉原型直接落在 image encoder 已经学到的特征流形里，**不存在"被钉住"问题**
- T-Rex2 / OWLv2 / DINO-X 在自然图像 OVD 上已经验证有效；医学场景**更适用**因为去掉了不可靠的文本桥梁

**Critical files**：
- 新写 `tools/build_visual_prototype.py`（image encoder + ROI pool）
- 新写 `tools/eval_novel_visual_prompt.py`（替换 cache_bank，跑 eval）
- 不改 `mm_backbone.py`

**Verification**：novel mAP per-organ 对比表，特别看 Resp/Thyroid 那两组（v2 文本 = 几乎 0）。

---

## 14. Text + visual dual-anchor fusion

**目的**：base 类靠 text（已学好，泛化）；novel 类靠 visual prototype（细类区分）。融合给整体最佳。

**方案**：
```
final_class_embed = α · text_embed + (1-α) · visual_prototype
```
α 可以学习（per-class），也可以按"训练见没见过"硬切：base α=1.0、novel α=0.3。

**前置**：item 13 完成，确认 visual prototype 单独能涨。

**Critical files**：扩 `PseudoLanguageBackbone` 支持 dual-source class vector。

---

## 15. Text encoder 换 BiomedCLIP / PubMedBERT

**目的**：从根上解决 XLM-Roberta 在医学语义上的判别力不足。

**方案候选**（按工程量排）：
- **PubMedBERT / BioClinicalBERT**：纯文本，dim 通常 768，可直接 drop-in 替换 XLM-Roberta（输入 tokenizer 也 HuggingFace 标准）。**不需要重新对齐图像端**，但需要重训整个 dev30（image encoder 要重新对齐到新文本空间）。
- **BiomedCLIP** (微软 PubMed 图文对训练)：text + image encoder 都换，需要把 ConvNext 替换为 BiomedCLIP image branch。架构改造较大。
- **MedCLIP / PMC-CLIP**：同 BiomedCLIP，候选。

**前置**：item 12 cos heatmap 确认是文本端瓶颈。

**预期**：fine-grained 类文本 cos 从 >0.95 跌到 0.7-0.8（reviewer 经验值）。

**Critical files**：
- `wedetect/models/backbones/mm_backbone.py` — 新加 `BiomedicalLanguageBackbone` 类
- `tools/build_text_embeddings.py` — 兼容新 tokenizer/encoder
- 新 config 链 `config/wedetect_tiny_tct_ngc_dev30_biomedbert_*.py`

---

## 16. Hierarchical 2-stage detection（粗 OV + 细 few-shot）

**目的**：承认 fine-grained 子型靠纯文本不可能 zero-shot 分开。改成临床上更稳的两阶段：
1. **Stage 1**: OVD 检测出"thyroid lesion"（粗类，文本可分）
2. **Stage 2**: 对检出 region 跑 few-shot prototype network（exemplar matching）

**为什么临床更优**：医生看到"系统检出甲状腺病灶，疑似 Beth-V/VI" 比"系统说这是 Beth-V" 更可信。可解释性 ↑，部署摩擦 ↓。

**前置**：item 13 验证 visual prototype 在 fine-grained 上有效。

**工程**：Stage 2 的 prototype head 可以写成独立模块挂在 dev30 image encoder 后面，不动主干。

---

## 17. Train-time prompt + exemplar augmentation（**重训方案**）

**目的**：让 dev30 base 训练阶段就**不是钉死在单一 prompt 方向上**。

**方案**：
- 每个 base 类预备 5-10 个 v2 风格 variants（PSC/Bethesda/形态学描述）
- `RandomLoadText` 在每个 batch 随机抽 1 个 variant 当作类向量（已支持 multi-prompt JSON 结构）
- 可选：每 N 步替换部分类的 text 向量为同类训练图的 visual prototype（exemplar augmentation），双模态训练
- 重训 1 次 dev30，~8h

**预期**：image encoder 学到的是"对齐到一片 prompt cloud" 而不是某个点。novel 端（v2 prompts）应该 +0.05-0.15，**但更重要的是 item 13/14 的 visual prototype 才能真正发挥（image encoder 不再钉死）**。

**Critical files**：
- `data/texts/tct_ngc_fullnames_30_v_aug.json` — 30 类 × 5-10 variants
- `tools/build_text_embeddings.py` — 接受 multi-variant，但训练时不平均（保留每个 variant 独立 embedding）
- `wedetect/datasets/transformers/mm_transforms.py:RandomLoadText` — 已支持，verify
- 新 config

---

## 18. Novel 评估按"粗类 vs 细类子型"拆分

**目的**：当前 4 个 novel split（main_3/pseudo_2/hard_4/full_5）混合了"组织粗类（Resp/Serous/Thyroid）" 和"细子型（Bethesda V vs VI、Adeno vs SCC）"。混合报告掩盖问题。

**方案**：报告里固定按 2 个维度拆：
- **Coarse-novel**（组织级别新类）：novel 类与 base 类不在同一 organ。预期文本 prompt 能涨。
- **Fine-novel**（细子型新类）：novel 类与 base 类同 organ，只是子型不同（Bethesda V/VI vs base Thyroid-AUC）。预期文本 prompt 几乎 0，必须靠 visual prompt。

**Critical files**：纯报告改动，无代码。

---

## 19. 4 层次属性 hierarchical prompt training（**结构化文本融合，治本方案**）

**目的**：让 image encoder 在训练阶段就学会**对齐到 4 层结构化文本**，而不是单条 flat prompt。这是 OWL-ViT / RegionCLIP / TaxCLIP 系列的 hierarchical prompt 训练标准范式，本质上是把医学诊断报告的天然层级结构注入文本编码。

**4 层定义**（每个类需要这 4 个字段）：
```
Level 1 organ:       e.g. "respiratory tract"
Level 2 tier:        e.g. "PSC Category VI: Malignant"
Level 3 subtype:     e.g. "Squamous cell carcinoma"
Level 4 morphology:  e.g. "keratinization, intercellular bridges, hyperchromatic nuclei"
```

**为啥不能 inference-only 做（已经诊断过的死路）**：
- 模型训练时见的是单条 flat prompt，image encoder 只学过"对齐到那条具体字符串编码"
- inference 时换成 4 层聚合（sum / concat / weighted）相当于全新的类向量分布，模型从未对齐过 → 同 CuPL 失败的根因
- 必须**训练时就用 4 层结构**，模型才学得会

**实现方案**（跟 item 17 合并）：
```python
# 训练时 LoadText 替换：
class HierarchicalLoadText:
    def __call__(self, results):
        for cls in classes:
            attrs = attr_db[cls]  # 4 个字段
            # Option A: 拼接成一段 ("Resp. PSC VI Malignant. SCC. With keratinization...")
            # Option B: 每层独立编码，sum/concat/weighted-avg
            # Option C: 训练时随机采样某个层级 (subtype-only / morphology-only / 全)
            ...
```

模型 forward 不用改：text_embed_dict 里直接存"融合后的 768 维向量"，PseudoLanguageBackbone 不感知。

**数据 curation（关键卡点）**：
- 30 个 base 类 × 4 层属性 = **30 张属性卡片**
- 每张卡 morphology 必须**基于教科书** / 国际诊断标准（PSC、Bethesda、WHO Tumor Classification、Robbins、Diagnostic Cytopathology by Bibbo & Wilbur）
- **不能让 LLM 直接生成**——医学术语容易幻觉，错的属性会把训练信号污染
- 病理医生或熟悉细胞学的同事过一遍，2-3 天人工
- 输出 JSON 格式：`data/texts/tct_ngc_fullnames_30_4attr.json`，schema =
  ```json
  [{"organ":"...", "tier":"...", "subtype":"...", "morphology":"..."}, ...]
  ```

**Critical files**：
- 新建 `data/texts/tct_ngc_fullnames_30_4attr.json`（人工 curation，含 30 base + 9 novel）
- 新写 `wedetect/datasets/transformers/mm_transforms.py:HierarchicalLoadText`
- 修改 `tools/build_text_embeddings.py` 支持 4 层 fusion 模式（sum/concat/weighted）
- 新 config `config/wedetect_tiny_tct_ngc_dev30_4attr_2gpu.py`

**Verification**：
1. 4 层属性 cos heatmap：fine-grained novel 对（SCC ↔ SmallCell）的 cos 应该比 v2 单 prompt 低（因为 morphology 字段引入了真正的判别信息）
2. 重训后 novel mAP 比 v2 baseline 涨 +0.05-0.15
3. **跟 visual prototype 协同**：4 层属性 + visproto 的双锚点融合应该比单边都好

**前置**：
- 数据 curation（2-3 天，需要外部协作）
- item 17 train-time prompt augmentation 框架先就绪（让多 variant prompt loader 能工作）

**预期时间**：阻塞在数据；数据齐了之后 1 天框架 + 1 次 8h 重训。

**跟其他 item 的关系**：
- 跟 item 17 合并：把 hierarchical 4-attr 当作 "augmentation" 的一种形式，统一进 train-time prompt aug 框架
- 跟 item 13/14 互补：visual prototype 解决"视觉判别"，hierarchical attr 解决"文本判别"，两者**正交**
- 跟 item 15 替代：如果换 BiomedCLIP，4 层属性的语义距离自然就拉开了；不换 encoder 时 hierarchical attr 是"用结构补 encoder 弱"

---

## 已 done（不在 TODO 范围）

- ✅ Patient-disjoint split（旧 image-CV → 新 disjoint）
- ✅ V2 国际标准 novel prompts (PSC / MAL-S / Bethesda)
- ✅ dev32 baseline 训练 + 评估 + 报告（含 §11 novel zero-shot per-class breakdown）
- ✅ 9 张 viz panel 嵌入 report 做问题分析
- ✅ Phase 1: 临床指标工具 + dev32 临床基线
- ✅ Phase 2: dev30 类合并（Urine 3-neg → NHGUC）+ 12-epoch 重训（best ep9 mAP 0.283）
- ✅ Phase 3: dev30 完整 eval + 5 plots + 16 viz panels + 中文报告
- ✅ NCCL watchdog timeout fix（dev_2gpu base + dev30 chain 都修了 env_cfg.dist_cfg.timeout）
- ✅ 3-panel loss curves（per-iter train + per-epoch val 同 x 轴 + mAP 独占）
- ❌ **Novel inference-side ensembling 实验（v3 全 norm + v3pure 仅 ensemble）—— 全部失败**，记录在 TODO 表头的"死路"段；不要再试。

---

## Phase 2.1 cos heatmap 诊断（2026-05-09，5 属性 × 4 聚合策略）

**❌ 组件 A 在 XLM-R 上 DEAD**。详见 `docs/tct_ngc_phase2_attr_cos_diagnostic_20260509.md`。

| 策略 | novel↔novel max cos | <0.92? |
|---|---:|---|
| v2 baseline (单 PSC prompt) | 0.996 | ❌ |
| concat (3840) | 0.971 | ❌ |
| sum (768) | 0.993 | ❌ |
| weighted-sum (768, dist 0.4) | 0.991 | ❌ |
| only-distinguish (768) | 0.971 | ❌ |

结构化 prompts 把 cos 降低 ~2.5 点（0.996 → 0.971），但 XLM-R 在医学 fine-grained 上是天花板。
**Pivot 决策**：跳到 item 15（BiomedCLIP / PubMedBERT），重新编码同 5 属性 JSON，验证是否能跌破 0.85。

---

## Novel zero-shot 已跑实验汇总（2026-05-09）

dev30 best ckpt (ep9 mAP 0.283) × 4 splits × 4 策略：

| 策略 | main_3 mAP | pseudo_2 mAP | hard_4 mAP | full_5 mAP | 状态 |
|---|---:|---:|---:|---:|---|
| v2 文本 baseline | 0.155 | 0.138 | 0.054 | 0.065 | ✅ Serous 强、Resp/Thyroid 死 |
| 视觉 prototype（5-shot, leakage）| 0.042 | 0.098 | 0.063 | 0.041 | ⚠ Resp/Thyroid 救活、Serous 大跌 |
| **post-hoc score fusion**（推荐）| **0.167** | **0.165** | **0.088** | **0.079** | ✅ **每个 split 都 ≥ 单源最优** |
| Raw binary fusion（DEAD-4）| - | - | - | - | ❌ Breast 0.454→0.000 |
| Procrustes calfused（DEAD-5）| 0.149 | 0.117 | 0.057 | 0.059 | ❌ visproto 类全死 |

**每类细节**（v2 / visproto / score-fuse / calfused）：

| split | class | v2 | vis | sfuse | cal |
|---|---|---:|---:|---:|---:|
| main_3 | Resp-SCC | 0.009 | 0.056 | 0.056 | 0.000 |
| main_3 | Serous-Breast | **0.447** | 0.070 | 0.447 | 0.447 |
| main_3 | Thyroid-MTC | 0.009 | 0.000 | 0.000 | 0.000 |
| pseudo_2 | Resp-Adeno | 0.042 | 0.095 | 0.095 | 0.000 |
| pseudo_2 | Serous-Ovarian | 0.234 | 0.100 | 0.234 | 0.234 |
| hard_4 | Resp-SmallCell | 0.001 | 0.052 | 0.052 | 0.000 |
| hard_4 | Serous-Adeno | 0.213 | 0.110 | 0.213 | 0.213 |
| hard_4 | Thyroid-Suspicious | 0.000 | 0.079 | 0.079 | 0.009 |
| hard_4 | Thyroid-MalTumour | 0.000 | 0.009 | 0.009 | 0.004 |
| full_5 | Resp-Adeno | 0.019 | 0.091 | 0.091 | 0.000 |
| full_5 | Serous-Ovarian | 0.203 | 0.072 | 0.203 | 0.203 |
| full_5 | Resp-SCC | 0.003 | 0.010 | 0.010 | 0.001 |
| full_5 | Serous-Breast | 0.093 | 0.030 | 0.093 | 0.093 |
| full_5 | Thyroid-MTC | 0.009 | 0.000 | 0.000 | 0.000 |

**关键观察**：
- score fusion 取**两边 max**，**每类的提升路径都被保留**（Serous 走 text、Resp/Thyroid 走 vis），无回归。
- calfused 看似能救 text 类（Breast 0.447 复活），但代价是 visproto 类全死（DEAD-5 = DEAD-4 的对偶失败）。**单 inference 双锚点路线全死**。
- Procrustes 数学正确（base 类 cos −0.296 → 0.971），但 R 在 base 拟合，novel visproto 旋转后落到错位置。

**论文 method 路径决策**：
- 组件 B 不再以"calibrated binary fusion in single inference"形式呈现；改成 **score fusion + per-class confidence calibration**（两次独立 inference）
- 组件 A（hierarchical attribute training）仍是论文主创新，等病理校对后启动
- 组件 C（learned gating）需要从 embedding-level 改成 **detection-level gating**（输入 = bbox feature + cos score from each branch）

产物：
- `data/texts/tct_ngc_novel_<split>_visproto_calibrated_emb.pth` × 4（保留作 ablation）
- `data/texts/tct_ngc_novel_<split>_calfused_emb.pth` × 4（保留作 ablation，但 method 不用）
- `data/texts/procrustes_R.pth`（768×768 旋转矩阵 + base common_keys）
- `tools/procrustes_text_visual.py`（Procrustes 对齐工具）
- `tools/fuse_novel_predictions.py`（score fusion 工具，已验证 4 splits）
- `data/texts/_archive/`（DEAD-4 raw binary fusion 已归档）
