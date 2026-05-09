# TCT_NGC dev30 后续 TODO

本 TODO 由 `docs/tct_ngc_dev32_disjoint_baseline_report_20260508.md` §9 + 用户讨论汇总。每项标注前置依赖、工程量、预期收益。**当前已在做的两件事**（P0/P1 临床指标 + P2 dev30 重训）见 `/home/25_liwenjie/.claude/plans/zazzy-hugging-pearl.md`，本 TODO 记录之外的待办。

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

> **本会话已确认的死路（不要再试）**：
> - 推理端文本 ensembling（CuPL-style 多 prompt 平均）—— v3/v3pure 实验显示**全部更差**，比 v2 baseline 跌 60-100%。
> - 推理端 anisotropy reduction (mean centering / whitening) —— 同上，破坏了 contrastive head 期望的 raw embedding scale。
> - 原因：WeDetect 的 PseudoLanguageBackbone 用**冻结缓存 + raw 内积**，image encoder 在训练时被钉死在某个特定 prompt 方向；任何 inference 后处理都是在挪一个被钉住的锚点。
> - 结论：**text 端不重训就别再折腾了**。要么走 visual prompt（item 13-14），要么换 text encoder（item 15），要么重训（item 17）。

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
