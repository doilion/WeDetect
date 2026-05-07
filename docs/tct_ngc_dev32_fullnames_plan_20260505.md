# TCT_NGC dev32 full-name baseline 实验计划（2026-05-05）

## 背景

- 当前 TCT_NGC dev split 切到 32 类（在 31 类基础上加入 `Urine-HGUC`，index 21）。
- 新建 `data/texts/tct_ngc_fullnames_32.json`（Bethesda 风格 full name）和对应的 cached embedding `tct_ngc_fullnames_32_embeddings_wedetect_tiny.pth`。
- 训练沿用 `/home1/liwenjie/TCT_NGC_640/` 的 letterbox 缓存数据集，`PseudoLanguageBackbone` 直接读 cached embedding。

校验结论（已通过）：
- prompts 数=32，embedding 数=32×768，无 slash/无重复，dict keys 与 JSON 1:1 对齐。
- 原始与缓存的 `instances_train_dev.json` / `instances_val_dev.json` 均含 32 个 categories；类目顺序与 config `base_classes` 完全一致；HGUC 在 train 2046 / val 250 anns。
- 缓存图片随机抽样 50 张全部存在；cache 标注 0 个零面积 / 0 个越界 bbox；仅 4 个 < 2px 的 tiny bbox（772k 中）可忽略。

已知风险（重要性递减）：
1. **Prompt embedding 存在多对 cos > 0.97**（详见下文 v2 表）；尤其 `Urine-Negative ↔ Urine-Negative Degeneration` cos=0.9895。
2. **缓存只覆盖 train_dev / val_dev**；跑 novel/base test 必须切 `data_root` 回 `/home1/liwenjie/TCT_NGC/`。
3. **cache640 与原图运行 `AP_S/M/L` 不可直接对比**（letterbox 改变了像素面积分桶）。

## 已落地的改动

### `config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_4gpu.py`
- `base_lr`：6.0e-4 → **4.5e-4**（sub-linear，避免 fine-tune 大 batch 大 LR 在高 cos prompt 下震荡）
- `max_keep_ckpts`：12 → **5**

2gpu config 不变。

## 执行顺序

### Step 1 — 启动 2GPU baseline（首选）

```bash
PORT=29644 bash dist_train.sh \
  config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu.py 2 --amp
```

预期：
- effective batch 32，base_lr 3.0e-4（per-sample LR ≈ 1.5× pretrain，安全区）
- iters/epoch ≈ 3641，warmup 1500 iter ≈ 0.41 epoch
- 12 epoch 总 ~43.7k iter，rare 类（FC 394 / Thy-Neg 652）能见到足够多次实例
- 走 `--amp` 估计 12-16h on 2× RTX 3090

### Step 2 — epoch 4 ~ 6 中期回看

到第 4-6 个 epoch best ckpt 出来后，跑一次单卡评估拉 classwise AP：

```bash
PYTHONPATH=. python test_exclude_negative.py \
  --config config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu.py \
  --checkpoint work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu/best_*.pth
```

重点关注：
- **Urine 4 类相互混淆**：NILM(16) / NHGUC(17) / NHGUC-Degen(20) / HGUC(21) / SHGUC(18) 的 confusion 是否集中
- **Thyroid PTC(9) ↔ SPTC(10)** 是否互吃 logit
- **Respiratory Neutrophil(0) ↔ Lymphocyte(3)、Ciliated(2) ↔ Squamous(5)** 是否互混
- **rare 类绝对 AP**：Thyroid-FC(15, 43 val anns)、Thyroid-Negative samples(14, 62)、Urine-AUC(19, 218)

### Step 3 — 决策点

依据 Step 2 的 per-class AP 决定下一步：

- **如果 Urine/Thyroid 高 cos pair 互相混淆显著**（confusion 集中、rare 类 AP < 0.1） → 启动 v2 prompt 重写（见下文）
- **如果整体 mAP 合理且 rare 类还在抬升** → 让 baseline 跑完 12 epoch，记录到 `docs/EXPERIMENTS_TABLE.md`

### Step 4 — 4GPU 跑（可选）

baseline 跑通后再考虑用改过的 4GPU 配置（base_lr 4.5e-4）做对照：

```bash
PORT=29648 bash dist_train.sh \
  config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_4gpu.py 4 --amp
```

## v2 prompt 设计（暂不动手，留作 plan B）

触发条件：Step 3 判定 prompt 是瓶颈。

### 设计三原则
1. **每个 prompt 句首词必须唯一**（不再以 `<Organ> cytology -` 起头）。
2. **Negative 类不允许出现被否定的疾病名**（NHGUC 不写 "high-grade urothelial carcinoma"）。
3. **rare 类（< 1000 anns）写 ≥ 2 个互不重叠的形态学特征**。

### 改写表（v2，针对 cos > 0.97 的 13 个类）

| idx | 旧短名 | 当前 prompt | v2 prompt |
| --- | --- | --- | --- |
| 0 | respiratory-Neutrophil | Respiratory tract cytology - Neutrophil | **Polymorphonuclear leukocyte** with multilobed segmented nucleus, acute inflammatory cell in airway sample |
| 2 | respiratory-Ciliated | Respiratory tract cytology - Ciliated respiratory epithelial cell | **Tall columnar epithelial cell** with apical cilia and basal nucleus from bronchial mucosa |
| 3 | respiratory-Lymphocyte | Respiratory tract cytology - Lymphocyte | **Small mononuclear leukocyte** with dense round nucleus and scant cytoplasm in airway sample |
| 5 | respiratory-Squamous | Respiratory tract cytology - Squamous epithelial cell | **Polygonal squamous epithelial cell** with abundant cytoplasm and small pyknotic nucleus from upper airway |
| 9 | Thyroid-PTC | Thyroid - PTC (Bethesda VI: Malignant) | **Papillary thyroid carcinoma cells** with intranuclear pseudoinclusions, nuclear grooves and powdery chromatin |
| 10 | Thyroid-SPTC | Thyroid - SPTC (Bethesda V) | **Thyroid follicular cells** with partial nuclear features of papillary carcinoma, suspicious but not definitive |
| 16 | Urine-NILM | Urinary cytology - NILM | **Benign urinary smear**: clean background with mature urothelial and squamous cells, no abnormality |
| 17 | Urine-Negative | Urinary cytology - Negative for HGUC (NHGUC) | **Bland urothelial cells** with low N:C ratio and uniformly small round nuclei in urine sediment |
| 18 | Urine-SHGUC | Urinary cytology - Suspicious for HGUC (SHGUC) | **Atypical urothelial cells** with elevated N:C ratio and hyperchromasia, features insufficient for definitive carcinoma |
| 19 | Urine-AUC | Urinary cytology - Atypical urothelial cells (AUC) | **Mildly atypical urothelial cells** with subtle nuclear enlargement and mild irregularity, of uncertain significance |
| 20 | Urine-Negative Degeneration | Urinary cytology - NHGUC, with degenerative changes | **Degenerated urothelial cells** with cytoplasmic vacuolation, smudged chromatin and pyknotic nuclei but no atypia |
| 21 | Urine-HGUC | Urinary cytology - High-grade urothelial carcinoma (HGUC) | **Pleomorphic urothelial cells** with high N:C ratio, hyperchromasia, coarse irregular chromatin and angulated nuclear membranes |
| 22 | TCT_CCD-normal | Cervical cytology - NILM | **Benign cervical Pap smear**: mature squamous cells with small uniform nuclei, no intraepithelial lesion |

剩余 19 类先保留原 prompt（cos < 0.95 的安全 pair）。

### v2 落地步骤
1. 写 `data/texts/tct_ngc_fullnames_32_v2_disambig.json`
2. `PYTHONPATH=. python tools/build_text_embeddings.py --texts data/texts/tct_ngc_fullnames_32_v2_disambig.json --out data/texts/tct_ngc_fullnames_32_v2_disambig_embeddings_wedetect_tiny.pth`
3. cos sanity check：所有 pair 必须 < 0.95，最坏 pair < 0.97
4. 复制 2gpu config，只改两行：`train_class_text_path` 和 `text_embed_path`
5. 用同一份 cache640 数据，复用 baseline ckpt 作为 `load_from`（节省 warmup）

## 待验证的假设

`Urine-Negative` 和 `Urine-Negative Degeneration` 在显微图像上是否真有可分形态学特征？  
跑 baseline 时盲采 5-10 张各类 crop 给一线看片人盲判，如果连人也分不开则两类应合并。

## 触发改 plan 的信号

| 现象 | 应对 |
| --- | --- |
| 2gpu 训练 epoch 1 后整体 mAP < 0.05 | 检查 LR/数据加载，不是 prompt 问题 |
| 中期 Urine 4 类 confusion 集中（pred 全跑去 NILM 或全跑去 HGUC） | 启动 v2 prompt |
| Thyroid-FC / Thyroid-Negative AP 12 epoch 仍 < 0.05 | 不是 prompt 能救的，需考虑类合并或加权采样 |
| 跑 novel/base test 报 image-not-found | 切 `data_root` 回 `/home1/liwenjie/TCT_NGC/` |
| 与 31 类非 cache 对比时 AP_S/M/L 不一致 | 预期行为，只比 mAP/AP50/AP75 |
