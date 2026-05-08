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

## 已 done（不在 TODO 范围）

- ✅ Patient-disjoint split（旧 image-CV → 新 disjoint）
- ✅ V2 国际标准 novel prompts (PSC / MAL-S / Bethesda)
- ✅ dev32 baseline 训练 + 评估 + 报告
- ✅ 9 张 viz panel 嵌入 report 做问题分析
- ⏳ Phase 1: 临床指标工具 + dev32 临床基线（**进行中**）
- ⏳ Phase 2: dev30 类合并 + 重训（**等 Phase 1 通过**）
- ⏳ Phase 3: dev32 vs dev30 对比报告
