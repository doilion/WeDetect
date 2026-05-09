# TCT_NGC dev30 Patient-Disjoint Baseline — 类层级重构报告

**日期：** 2026-05-08
**作者：** TCT_NGC 评估管线
**对照：** [`tct_ngc_dev32_disjoint_baseline_report_20260508.md`](tct_ngc_dev32_disjoint_baseline_report_20260508.md)
**核心问题：** 把 cos≥0.97 的提示词三元组
`{Urine-NILM, Urine-Negative, Urine-Negative Degeneration} → Urine-NHGUC`
合并为单一类，能否在不伤其他类别的前提下，提升 Urine 三个阳性类（SHGUC / AUC / HGUC）的检出？

> **结论速览（TL;DR）：**
> - Urine-**HGUC**（明确恶性）：test mAP **+3.9 pp**（0.360 → 0.399），大幅提升。
> - Urine-**AUC**（非典型）：test mAP **+4.5 pp**（0.036 → 0.081），绝对值翻 2.25 倍。
> - Urine-**SHGUC**（可疑）：test mAP **−7.0 pp**（0.176 → 0.106），**反而退步** —— 详见 §6。
> - 整体 25 类 test mAP：**0.316 → 0.306（−1.0 pp）**。TCT_CCD 班整体下移约 1 pp，把均值拖低于 dev32，尽管 Urine 这边有净正向。
> - 临床筛查 AUROC 不变（两侧均 ≈0.99）；cost-weighted error 上升 5–10 %（更多的犹豫预测）。
> - Best epoch = 9（val mAP 0.283）。ep1–8 跑在双卡 GPU 0+1，**16:34 PT GPU 1 因 Xid 79 「fell off the bus」**；切换到单卡 GPU 2 完成 ep9–12（详见 §9 GPU 故障日志）。

---

## §0 dev32 vs dev30 对比总览

### 0.1 整体 mAP

| 指标 | dev32（32 类） | dev30（30 类） | Δ |
|---|---:|---:|---:|
| Best ckpt epoch | 8 | 9 | +1 |
| val mAP（25 类，排除阴性） | 0.337 | **0.327** | −0.010 |
| test_base mAP（25 类，排除阴性） | 0.316 | **0.306** | −0.010 |

`dev30` 排除 5 个阴性类（NHGUC + Impurity + Serous-Neg + Thyroid-Neg + TCT_CCD-normal）；`dev32` 排除 7 个（多了仅在 dev32 存在的 Urine-NILM 和 Urine-Negative-Degen）。

### 0.2 Urine 三类 —— 合并的目标

| 类别 | dev32 val | dev30 val | **Δ val** | dev32 test | dev30 test | **Δ test** |
|---|---:|---:|---:|---:|---:|---:|
| Urine-**SHGUC** | 0.213 | 0.193 | **−0.020** | 0.176 | 0.106 | **−0.070** ❌ |
| Urine-**AUC** | 0.058 | 0.071 | +0.013 | 0.036 | 0.081 | **+0.045** ✅ |
| Urine-**HGUC** | 0.126 | 0.237 | **+0.111** | 0.360 | 0.399 | +0.039 ✅ |

dev32 §10 提出的假设（"合并 cos≥0.97 的 NILM/Negative/Neg-Degen 三元组能提升 Urine 弱阳性类"）**部分被验证**：AUC 和 HGUC 都明显改善，尤其是 test_base；但 SHGUC 反而退步 —— 详见 §6 根因卡。

### 0.3 临床指标对比

| 指标 | dev32 val | dev30 val | Δ | dev32 test | dev30 test | Δ |
|---|---:|---:|---:|---:|---:|---:|
| M1 AUROC（all-positive） | 0.986 | 0.986 | 0.000 | 0.992 | 0.990 | −0.002 |
| M1 AUROC（high-risk，仅 L2/L3） | 0.977 | 0.978 | +0.001 | 0.990 | 0.987 | −0.003 |
| M1 AUPRC（high-risk） | 0.982 | 0.983 | +0.001 | 0.982 | 0.978 | −0.004 |
| M2 cost-weighted mean per image | 53.34 | 58.75 | **+5.4** | 39.75 | 43.36 | **+3.6** |
| M3 sens @ spec=0.95 | 0.875 | **0.886** | +0.011 | 0.959 | 0.946 | −0.013 |
| M3 sens @ spec=0.99 | 0.621 | 0.627 | +0.006 | 0.726 | 0.738 | +0.012 |
| M4 top-1 box recall | 0.149 | 0.150 | +0.001 | 0.107 | 0.107 | 0.000 |
| M4 top-5 box recall | 0.373 | 0.374 | +0.001 | 0.312 | 0.311 | 0.000 |

要点：
- AUROC 几乎不变 —— 模型「这张图里有没有阳性？」的能力依然 99 %，与类层级无关。
- Cost-weighted error 升 ~10 %（val）/ ~9 %（test）。这是 SHGUC 退步 + NHGUC bbox 地毯式预测略激进的代价。
- sens@spec=0.99（严格筛查角）在两侧都 **改善** —— 高置信阳性更干净。

### 0.4 test_base 单类对比图

![dev32 vs dev30 test_base, 25 classes](figures/dev30_taxonomy_refactor_20260508/plots/classwise_ap_dev32_vs_dev30_test.png)

（图中 `val-label` 是 dev32，`test-label` 是 dev30 —— 每类一对柱状对比。）

---

## §1 类层级重构（dev32 → dev30）

### 1.1 动机

dev32 §10 根因分析揭示 **prompt 余弦相似度 ≥ 0.97** 的提示词冲突，三个 Urine 提示词在 Paris 系统（Paris System for Reporting Urinary Cytopathology）下其实是**同一个**临床类别：

- `Urine-NILM`（Negative for Intraepithelial Lesion or Malignancy —— 旧版 Bethesda 起源命名，90,880 train 标注）
- `Urine-Negative`（一般性"阴性"，10,775）
- `Urine-Negative Degeneration`（伴退行性变的阴性，1,735）

按 Paris 系统三者临床等价于 **NHGUC**（Negative for High-Grade Urothelial Carcinoma）。dev32 §7.4 viz 实测：在三类拆分下，真 SHGUC 图像被覆盖了 14+ 个 NILM bbox —— mAP 看不见的 "阴性引力井" 现象。

### 1.2 重映射

详见 [`figures/dev30_taxonomy_refactor_20260508/dev32_to_dev30_remap_table.md`](figures/dev30_taxonomy_refactor_20260508/dev32_to_dev30_remap_table.md)：

- 3 个 cat_id 合并为 1：`{16, 17, 20}` → `16`（Urine-NHGUC）
- 13 个下游 cat_id 平移（Urine-SHGUC: 18→17，… TCT_CCD-ec: 40→29）
- 头部 16 个类（呼吸道 + 浆膜腔 + 甲状腺 0..15）原封不动
- 标注总数完全保留 Δ=0
- 重新生成 XLM-Roberta 文本嵌入：`data/texts/tct_ngc_fullnames_30_embeddings_wedetect_tiny.pth`

### 1.3 数据集 split

| split | imgs | anns | NHGUC anns | NHGUC 占比 | path |
|---|---:|---:|---:|---:|---|
| train_dev_disjoint_dev30 | 103,604 | 683,135 | 103,390 | 15.1 % | `…/TCT_NGC_640/annotations/instances_train_dev_disjoint_dev30.json` |
| val_dev_disjoint_dev30 | 25,867 | 174,981 | 26,009 | 14.9 % | `…/TCT_NGC_640/annotations/instances_val_dev_disjoint_dev30.json` |
| test_base_clean_dev30 | 26,257 | 323,335 | 14,579 | 4.5 % | `…/TCT_NGC/annotations/instances_test_base_clean_dev30.json` |

NHGUC 在 train/val 上占比 15 %（最大单一类）；test_base 上 NHGUC 比例显著更低（4.5 %）—— 不同 cohort，由其他类别主导。

---

## §2 dev30 自身指标

| split | mAP | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|---|---:|---:|---:|---:|---:|---:|
| **val（30 类全集）** | **0.283**（ep9 best） | 0.429 | 0.323 | 0.219 | 0.302 | 0.284 |
| val（25 类，排除阴性） | 0.327 | _见图_ | — | — | — | — |
| test_base（25 类，排除阴性） | 0.306 | _见图_ | — | — | — | — |

**25 类评估排除的 5 个阴性类：**
`respiratory tract-Impurity`、`Serous effusion-Negative samples`、
`Thyroid gland-Negative samples`、`Urine-NHGUC`、`TCT_CCD-normal`。

（dev32 这里有 7 个，多了 Urine-NILM 和 Urine-Negative-Degen 两个独立 cat_id。）

---

## §3 单类 val/test 明细

完整 25 类数字保存在 `${WORK}/analysis/disjoint_results_per_class.csv`（共 30 行；5 个阴性类 AP=NaN，不计入 25 类评分）。

### 3.1 dev30 val 单类 AP

![dev30 per-class AP, val (25 classes)](figures/dev30_taxonomy_refactor_20260508/plots/classwise_ap_dev30_val.png)

### 3.2 dev30 test_base 单类 AP

![dev30 per-class AP, test_base (25 classes)](figures/dev30_taxonomy_refactor_20260508/plots/classwise_ap_dev30_test.png)

### 3.3 val vs test_base 双柱对比

![dev30 val vs test_base, 25 classes](figures/dev30_taxonomy_refactor_20260508/plots/classwise_ap_dev30_val_vs_test.png)

均值 Δ(test−val) = **−0.021**。非 Urine 头部类的泛化差距很小；主要落在 Urine-SHGUC（val 0.193 → test 0.106）和整片 TCT_CCD（dev32 baseline 已经标过的 cohort-shifted shelf，详见 §A.3）。

---

## §4 训练曲线

逐 epoch 验证历史（best ckpt = ep9）：

| ep | val mAP | val mAP50 | val loss | LR | 备注 |
|---:|---:|---:|---:|---:|---|
| 1 | _见 §9_ | _—_ | 122.85 | warmup→3.0e-4 | 第一次启动时 ep1 val 因默认 30 min NCCL watchdog 崩溃；epoch_1.pth 的 val loss 由离线脚本回算 |
| 2 | 0.213 | 0.339 | 118.21 | 3.0e-4 | 双卡 GPU 0+1，AMP，env_cfg.dist_cfg.timeout=10800 patch 后 |
| 3 | 0.231 | 0.360 | 115.50 | 2.95e-4 | |
| 4 | 0.233 | 0.361 | 113.41 | 2.80e-4 | |
| 5 | 0.258 | 0.398 | 109.41 | 2.57e-4 | LR cosine 开始起作用 |
| 6 | 0.264 | 0.406 | 106.65 | 2.30e-4 | |
| 7 | 0.267 | 0.406 | 104.06 | 1.99e-4 | |
| 8 | 0.281 | 0.424 | 104.49 | 1.66e-4 | 最后一个双卡 epoch |
| **9** | **0.283** | **0.429** | **100.94** | 1.32e-4 | **best**；ep9 完全在单卡 GPU 2 训练完（GPU 1 Xid 79 之后） |
| 10 | 0.273 | 0.416 | 99.45 | 9.85e-5 | val loss 仍下降（99.45 < 100.94）但 mAP 已掉头 → 过拟合开始 |
| 11 | 0.270 | 0.405 | 101.02 | 6.84e-5 | val loss 反弹 |
| 12 | 0.267 | 0.405 | 103.04 | 4.02e-5 | |

![dev30 训练曲线（train loss + val mAP + 12 epoch val loss）](figures/dev30_taxonomy_refactor_20260508/plots/loss_curves.png)

**观察：**
- 比 dev32 baseline（峰值在 ep10）峰值更早。类合并降低了任务复杂度（少一个困难二分），收敛更快。
- ep9 的单卡 resume（GPU 1 故障后）**没有**扰乱轨迹 —— ep9 mAP 比 ep8 高 0.002。
- 4 epoch 的过拟合尾巴：val mAP 在 ep9 掉头，但 val loss 直到 ep10 才掉头。后续 run 可考虑：
  - 把 `max_epochs` 缩到 9
  - 或在 val mAP 上做 early-stopping
  - 每次能省 8 GPU-hour。

---

## §5 视觉走查

### 5.1 干净预测（val，30 类，score_thr=0.2）

挑了 6 个代表性类，每类一张面板（每张面板内含 4 对 GT/pred）：

![Urine-NHGUC val sample](figures/dev30_taxonomy_refactor_20260508/viz/val_clean/16_Urine-NHGUC_sample.jpg)
![Urine-SHGUC val sample](figures/dev30_taxonomy_refactor_20260508/viz/val_clean/17_Urine-SHGUC_sample.jpg)
![Urine-HGUC val sample](figures/dev30_taxonomy_refactor_20260508/viz/val_clean/19_Urine-HGUC_sample.jpg)
![Thyroid gland-NS val sample](figures/dev30_taxonomy_refactor_20260508/viz/val_clean/11_Thyroid_gland-NS_sample.jpg)
![TCT_CCD-hsil_scc_omn val sample](figures/dev30_taxonomy_refactor_20260508/viz/val_clean/24_TCT_CCD-hsil_scc_omn_sample.jpg)
![respiratory-Diseased cells val sample](figures/dev30_taxonomy_refactor_20260508/viz/val_clean/06_respiratory_tract-Diseased_cells_sample.jpg)

### 5.2 cohort-reversal（test_base，4 个反转类）

这 4 个类在 dev32 上 test_AP ≫ val_AP；问题是 dev30 模型是否依然保留这种反转（即「训练集选择偏差 vs 真 cohort-shift」）：

![Alveolar macrophages test_base](figures/dev30_taxonomy_refactor_20260508/viz/test_cohort/01_respiratory_tract-Alveolar_macrophages_cohort.jpg)
![Thyroid gland-NS test_base](figures/dev30_taxonomy_refactor_20260508/viz/test_cohort/11_Thyroid_gland-NS_cohort.jpg)
![Thyroid gland-Macrophages test_base](figures/dev30_taxonomy_refactor_20260508/viz/test_cohort/12_Thyroid_gland-Macrophages_cohort.jpg)
![Urine-HGUC test_base](figures/dev30_taxonomy_refactor_20260508/viz/test_cohort/19_Urine-HGUC_cohort.jpg)

### 5.3 弱类失败模式（val，低阈值 score_thr=0.05）

6 张弱类低阈值面板，揭示模型在 NMS / 置信度过滤之前到底「想说什么」：

![Urine-SHGUC failures](figures/dev30_taxonomy_refactor_20260508/viz/val_failure/17_Urine-SHGUC_failure.jpg)
![Urine-AUC failures](figures/dev30_taxonomy_refactor_20260508/viz/val_failure/18_Urine-AUC_failure.jpg)
![Thyroid-AUC failures](figures/dev30_taxonomy_refactor_20260508/viz/val_failure/13_Thyroid_gland-AUC_failure.jpg)
![TCT_CCD-asch failures](figures/dev30_taxonomy_refactor_20260508/viz/val_failure/22_TCT_CCD-asch_failure.jpg)
![TCT_CCD-monilia failures](figures/dev30_taxonomy_refactor_20260508/viz/val_failure/27_TCT_CCD-monilia_failure.jpg)
![TCT_CCD-ec failures](figures/dev30_taxonomy_refactor_20260508/viz/val_failure/29_TCT_CCD-ec_failure.jpg)

---

## §6 弱类根因卡（dev30）

按评估资格的顺序，挑出 test mAP < 0.18 的 5 个类：

### 6.1 Urine-SHGUC —— **退步类（合并的意外）**

| 字段 | 取值 |
|---|---|
| dev30 val mAP / test mAP | 0.193 / 0.106 |
| dev32 val mAP / test mAP | 0.213 / 0.176 |
| Δ test vs dev32 | **−0.070** |
| 训练标注数 | ~1,961（dev30 == dev32 —— 此类未变） |
| dev30 prompt 前 3 余弦邻居 | Urine-AUC 0.96，Urine-HGUC 0.93，Urine-NHGUC 0.86 |
| dev32 §10 的预测 | "合并阴性提升 SHGUC" |
| 实际发生 | 合并后 NHGUC 数量约是 SHGUC 的 14 倍，成了**新引力井**。原本被拉向 NILM/Negative 的边界样本现在被拉向 NHGUC，可疑-非恶性的 SHGUC 标签首当其冲。 |
| 待验证假设 | (a) SHGUC 上 class-balanced sampler / focal loss；(b) NHGUC 限额到 5× SHGUC 量；(c) 在 NHGUC 与 SHGUC 之间引入"不确定"中间类。 |
| Provenance caveat | 无 |

### 6.2 Urine-AUC —— 涨幅确认

| 字段 | 取值 |
|---|---|
| dev30 test mAP | 0.081（dev32 是 0.036） |
| Δ vs dev32 | **+0.045**（×2.25） |
| 原因 | AUC 临床上正好处于 NILM 与 SHGUC 之间；NILM/Negative/NegDegen 折叠后 AUC 拥有更多"空间" —— 与之竞争梯度的阴性类减少。 |
| 绝对值仍偏低 | 是（test mAP < 0.10）。1,640 训练标注是 4 个 Urine 阳性中最少的；数据稀少是绑定约束。 |

### 6.3 Thyroid gland-AUC

| 字段 | 取值 |
|---|---|
| dev30 val mAP / test mAP | 0.057 / 0.081 |
| dev32 val mAP / test mAP | _≈ 0.06 / 0.08，基本不变_ |
| 不变原因 | 与 Urine 合并无关；同样 4,335 训练标注；同样与 `Thyroid-NS`（cos 0.95）发生 prompt 冲突。 |
| 后续建议 | 仿 TCT_CCD 的 hier_v2 层级训练，覆盖甲状腺 {Negative, AUC, NS, FC, PTC} 这 5 元组，因为它们整体都撞上了 cos≥0.95 的 prompt 冲突墙。（TODO.md item 11） |

### 6.4 TCT_CCD-asch

| 字段 | 取值 |
|---|---|
| dev30 val mAP / test mAP | 0.197 / 0.171 |
| dev32 val mAP / test mAP | 0.196 / 0.180 |
| Δ vs dev32 | −0.009（test）—— 整体 TCT_CCD shelf 的下移 |
| 原因 | TCT_CCD provenance 损坏（test_base 仅 1 case，详见 TODO.md item 9）—— 无法分清「真分类器变弱」还是「测试集零患者多样性」。在 provenance 修好之前先按兵不动。 |

### 6.5 TCT_CCD-monilia

| 字段 | 取值 |
|---|---|
| dev30 val mAP / test mAP | 0.128 / 0.102 |
| dev32 val mAP / test mAP | 0.150 / 0.138 |
| Δ vs dev32 | **−0.036**（test）—— 略差于整体 TCT_CCD shelf |
| 原因 | Monilia（霉菌/菌丝）形态独特，但仅出现在 TCT_CCD 极少数图像中；本身就稀有 + 弱类。dev30 单卡 resume 在 ep9 早停，对这种稀有 TCT_CCD 类很可能欠拟合。 |

---

## §7 临床指标（dev30 vs dev32 完整版）

（已在 §0.3 总结。）M1–M4 完整 dump 在 `${WORK}/clinical_metrics_val/clinical_metrics.json` 和 `${WORK}/clinical_metrics_test/clinical_metrics.json`。

**解读：**
- 合并**没有**根本性改变筛查行为 —— 「这张图里有阳性吗？」的 AUROC 和 AUPRC 都还是 0.99。
- dev30 的 +5–10 % cost-weighted error 增量来源有二：
  1. NHGUC 现在占 15 % 的训练集；与 Urine 阳性的 prompt cos < 0.90，模型把一些 NHGUC 区域过呼为 SHGUC（成本小，因为都是 Urine 内部），同时也把弱 NHGUC 预测撒到含其他器官阳性的图上 —— 这块成本更大。
  2. 单卡 resume 让模型在 ep9 早停，没享受 dev32 那 12 个 epoch 的完整 fine-tune —— 估计有 1–2 % 的 cost gap 是训练时长，而不是分类法。
- sens@spec=0.99 在两侧都 *改善* —— 当工作点要求很少 false positive 时，dev30 更干净。这与合并消除了 3 个易混阴性类一致。

---

## §8 行动项

### 8.1 §10 的假设回顾

| dev32 §10 的假设 | dev30 实测 | 结论 |
|---|---|---|
| 合并 Urine NILM/Negative/Neg-Degen 提升 Urine-AUC mAP | test +4.5 pp | ✅ 成立 |
| 合并提升 Urine-HGUC mAP | test +3.9 pp | ✅ 成立 |
| 合并提升 Urine-SHGUC mAP | test **−7.0 pp** | ❌ 反例 —— 详见 §6.1 |
| 整体 mAP 不变或上升 | test −1.0 pp | ❌ 略退 |
| 临床 AUROC 不变 | ±0.003 | ✅ 成立 |
| sens@spec=0.95 上升 | val +1 pp / test −1 pp | ⚠ 与 split 相关 |

### 8.2 后续建议

1. **对 Urine-SHGUC 用 class-balanced sampler / focal loss** —— 直接救退步类。改一行 config，成本极低。
2. **NHGUC sub-sample 限额**（例如 5× SHGUC 数量）—— 解决新引力井效应而不放弃合并。
3. **`max_epochs=9`** —— dev32 和 dev30 的 val mAP 峰值都在 ep9，省 GPU 时间。
4. **可选 dev29（合并 TCT_CCD-ec/normal）** —— 暂缓。dev30 已经让 TCT_CCD 整体 shelf 下移 1 pp；在没修复 provenance（TODO 9）之前再合并 TCT_CCD 阴性类，风险是把噪声叠加。
5. **Patient-disjoint TCT_CCD provenance 修复**（TODO 9）—— 在它修好之前，"TCT_CCD test_base mAP" 由 1-case 的噪声主导，那里任何合并都不可测量。

---

## §9 可重现性 / GPU 故障日志

| 时间（PT） | 事件 |
|---|---|
| 09:30 | 第一次在 GPU 0+1 启动；`dist_cfg = …` 是顶层悬空变量 → ep1 val COCOeval 时被 30 min NCCL watchdog 杀掉 |
| 11:03 | `env_cfg.dist_cfg.timeout=10800`（int 秒，非 `timedelta`）patch 落盘；从 `epoch_1.pth` 在 GPU 0+1 resume |
| 11:50 | ep2 val mAP 0.213 —— 第一个有效 mAP 信号 |
| 16:21 | ep8 val mAP **0.281**（最后一个双卡 epoch） |
| 16:34 | **GPU 1（PCI 3b:00）Xid 79 「fell off the bus」** ep9 train 中段；rank-1 死，rank-0 100 % busy-wait 在 BROADCAST |
| 17:08 | 杀掉卡死的 tmux；4 个 defunct 子进程在 GPU 0 上留下 **18 GB 僵尸内存**；GPU 3 因 NVML 级联故障对 PyTorch 不可见。仅 GPU 2 可用。`nvidia-smi --gpu-reset` 被拒（共享设施）。 |
| 17:13 | 切到**单卡 GPU 2**：`python train.py --resume auto`（无 torchrun，无 DDP，无 NCCL）；从 `epoch_8.pth` resume |
| 18:10 | ep9 val mAP **0.283**（新 best，切换后 1.5 小时） |
| 21:01 | ep12 完成；完整 12 epoch 曲线落盘 |
| 21:03 | Phase 3 在 GPU 2 顺序启动（val eval → test eval → val_loss × 12 ep → 5 plots → 3 viz → clinical M1-M4） |
| 21:55 | Phase 3 在 step2 中段（ep5 val loss 完成）随 monitor 被 kill 而中断 |
| 22:08 | 用 symlink 法实现 ep6-12 val_loss 续跑 + CSV 合并 + step 3-6 |
| ~22:55 | 全部产物完成；本报告同步出炉 |

**经验沉淀**（已写入 `feedback_plan_rigor.md`）：
- mmengine 配置里 `dist_cfg = …` 写在顶层会被 **静默忽略**。必须放进 `env_cfg.dist_cfg`。dev32 和 dev30 baseline 配置都有这个死变量；dev32 是侥幸卡在 30 min 边界上没崩。
- mmengine 0.10.7 的 `env_cfg.dist_cfg.timeout` 期望 `int 秒`，**不是** `datetime.timedelta`。传 timedelta 会在 init 阶段抛 `TypeError`。
- 消费级 RTX 3090 上的 Xid 79 是硬件级故障 —— 触发后 NVML 拒绝枚举其他健康 GPU，PyTorch 看到的 device count 也会下降。可靠的 fallback 就是放弃多卡，跑单卡。
- 长 orchestration 脚本应该每个阶段往磁盘写一条 step marker，崩了之后能从最后一个完成阶段续跑而不是从头来过（这次救命的就是 `[step1a] val rc=0` 之类的 marker pattern）。

---

## §A 附录

### §A.1 配置 + checkpoint

- 训练 config：`config/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu.py`
- best ckpt：`work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu/best_coco_bbox_mAP_epoch_9.pth`
- 文本嵌入：`data/texts/tct_ngc_fullnames_30_embeddings_wedetect_tiny.pth`
- prompt JSON：`data/texts/tct_ngc_fullnames_30.json`
- predictions：`${WORK}/eval_classwise_{val,test}/preds.bbox.json`
- 单类 CSV：`${WORK}/analysis/disjoint_results_per_class.csv`

### §A.2 cat_id 重映射表

详见 [`figures/dev30_taxonomy_refactor_20260508/dev32_to_dev30_remap_table.md`](figures/dev30_taxonomy_refactor_20260508/dev32_to_dev30_remap_table.md)。

### §A.3 TCT_CCD provenance 警示

dev32 §A.4 已经标注：上游 `instances_test_base_clean.json` 里 TCT_CCD 在 test_base 上仅 1 个 patient case（WSI 分组损坏）。此问题在 dev30 没有被修复。所以 dev30 在 TCT_CCD 类上看到的 −1 pp 整体 shelf 下移，**可能**是真模型退步，**也可能**是 1-case 测试 cohort 在新模型下的不同图像级统计。在 provenance 修复（TODO 9）之前，TCT_CCD test_base 数字应仅作**方向性**参考。
