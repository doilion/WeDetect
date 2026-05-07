# TCT_NGC dev32 full-name baseline 实验报告（2026-05-06）

## 1. 概况

- Config: `config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu.py`
- Checkpoint: `work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu/best_coco_bbox_mAP_epoch_12.pth`（12 epoch 末就是 best）
- 训练时长：约 9h（2× RTX 3090 + AMP）
- 训练健康度：grad_norm nan 仅 2 次（早期 AMP 初始化伪报，后续无）

## 2. 训练曲线（val on cache640 train_dev/val_dev）

| Epoch | mAP | mAP_50 | mAP_75 | AP_s | AP_m | AP_l |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.206 | 0.329 | 0.229 | 0.154 | 0.233 | – |
| 5 | 0.312 | 0.477 | 0.354 | 0.253 | 0.344 | 0.313 |
| 8 | 0.339 | 0.507 | 0.390 | 0.285 | 0.373 | 0.332 |
| 9 | 0.342 | 0.520 | 0.389 | 0.288 | 0.381 | 0.354 |
| 10 | 0.350 | 0.519 | 0.401 | 0.306 | 0.382 | – |
| 11 | 0.349 | 0.520 | 0.400 | 0.308 | 0.377 | – |
| **12** | **0.352** | **0.522** | **0.405** | **0.312** | **0.380** | **0.355** |

cosine 末段 epoch 9→12 仍 +0.010，没有过拟合迹象。

## 3. Classwise AP（exclude 7 个 negative 类后，25 类）

总体：mAP=**0.413** / mAP_50=**0.601** / mAP_75=**0.477** / AP_s=0.362 / AP_m=0.433 / AP_l=0.399。
（exclude 后比 full 32 类的 0.352 高 +0.061，符合预期：负样本类拖低均值。）

排序后：

| 档 | 类别 | mAP | mAP_50 | 备注 |
| --- | --- | --- | --- | --- |
| 强 (>0.6) | respiratory-Squamous | **0.756** | 0.878 | top |
|  | Thyroid-PTC | **0.698** | 0.883 | 主要恶性 |
|  | respiratory-Alveolar macrophages | 0.666 | 0.794 |  |
|  | respiratory-Diseased cells | 0.644 | 0.875 |  |
|  | Thyroid-Macrophages | 0.607 | 0.814 |  |
| 中 (0.4-0.6) | Thyroid-FC | **0.548** | 0.827 | rare 类（394 train ann），意外地好 |
|  | TCT-dysbacteriosis_herpes_act | 0.541 | 0.772 |  |
|  | Serous-Diseased cells | 0.507 | 0.774 |  |
|  | respiratory-Ciliated | 0.501 | 0.713 |  |
|  | Thyroid-SPTC | 0.420 | 0.562 | Bethesda V，与 PTC 同源 |
|  | respiratory-Lymphocyte | 0.418 | 0.593 |  |
|  | TCT-agc_adenocarcinoma_em | 0.411 | 0.678 |  |
|  | Urine-HGUC | 0.409 | 0.564 | 主要恶性，没塌陷 |
| 边缘 (0.3-0.4) | respiratory-Neutrophil | 0.378 | 0.520 |  |
|  | Thyroid-NS | 0.350 | 0.516 |  |
|  | TCT-lsil | 0.325 | 0.555 |  |
|  | TCT-hsil_scc_omn | 0.322 | 0.576 | 主要恶性 |
|  | TCT-ascus | 0.313 | 0.530 |  |
| 弱 (<0.3) | **Thyroid-AUC** | 0.272 | 0.367 | Bethesda III，医学模糊 |
|  | **Urine-SHGUC** | 0.256 | 0.356 | 高 cos prompt 嫌疑 |
|  | **TCT-vaginalis** | 0.227 | 0.471 | 寄生虫，少样本 |
|  | **TCT-asch** | 0.215 | 0.406 | ASC-H，与 ascus 高 cos |
|  | **TCT-ec** | 0.213 | 0.374 | endocervical |
|  | **Urine-AUC** | **0.191** | 0.275 | 高 cos prompt 嫌疑，全场最低恶性预警类 |
|  | **TCT-monilia** | 0.145 | 0.351 | 真菌，少样本 |

## 4. 高 cos prompt pair 实际表现 vs plan 预测

Plan 的担忧：cos>0.97 的 13 个类会出现互吃 logit / 塌陷。实际看：

| pair | plan 预测 | 实际 | 是否塌陷 |
| --- | --- | --- | --- |
| Thyroid PTC ↔ SPTC | 互吃 | 0.698 / 0.420 | **否**，PTC 强、SPTC 中等（医学差异本就大，prompt 不是主因） |
| Urine HGUC ↔ SHGUC ↔ AUC | 互吃 | 0.409 / 0.256 / **0.191** | **部分**，HGUC 健康，SHGUC/AUC 弱 |
| respiratory 4 类（Neu/Lymph/Cili/Squam） | 互混 | 0.378 / 0.418 / 0.501 / 0.756 | **否**，无类塌到 < 0.3 |
| TCT ascus ↔ asc-h ↔ lsil ↔ hsil | 互混 | 0.313 / 0.215 / 0.325 / 0.322 | **轻微**，asc-h 弱 |

**结论**：plan 的"Urine 4 类 confusion 集中（pred 全跑去 NILM 或 HGUC）"信号未触发，但 **Urine-SHGUC=0.256、Urine-AUC=0.191** 确实偏低，且 cos 相似度最高的 pair 集中在这里。

Plan 触发条件对照：

| 触发条件 | 实测 | 是否触发 |
| --- | --- | --- |
| epoch 1 mAP < 0.05 | 0.206 | ✗ 不触发 |
| Urine 4 类 confusion 集中 | HGUC 健康，AUC/SHGUC 弱但非全军覆没 | ✗ 部分触发 |
| Thyroid-FC / Thy-Negative AP < 0.05 | FC=**0.548** | ✗ 不触发 |

## 5. 决策建议（接 plan Step 3）

按 plan，**baseline 整体合理 + rare 类未塌陷 → 不立即切 v2 prompt**；记录到对比表，作为后续实验的对照基线。

但有 3 类需后续单独优化（按性价比排）：

1. **Urine-AUC 0.191 / Urine-SHGUC 0.256**：在 cos>0.98 pair 群里，prompt 可能确实在害事；最值得做局部 v2（仅改 Urine 6 类的 prompt + 复用 cache640 数据 + 复用 baseline ckpt 作 `load_from`）。
2. **TCT-monilia 0.145 / TCT-vaginalis 0.227**：rare + 形态特殊；先确认是否 train ann 数太少（< 200），如确实样本不足，prompt 改写收益有限，应改为加权采样或类合并。
3. **TCT-asch 0.215**：与 ascus 高 cos；可在 Urine v2 同批改写 Cervical 4 类 prompt。

## 6. 后续动作（按优先级）

- **立刻**：把这次 baseline 写入 `docs/EXPERIMENTS_TABLE.md`，记录 mAP=0.352 / exclude-neg mAP=0.413（如表不存在则跳过）。
- **下一轮（可选）**：v2-mini，仅改 Urine 6 类 + Cervical asch/ascus prompt（约 8 行 JSON）；保留 Respiratory / Thyroid / 其他 prompt 不动。这样隔离 prompt 改动的因变量。
- **不建议立刻做**：4GPU 跑（4.5e-4 LR 配置）—— 当前 2GPU 已经收敛得不错，4GPU 换的是吞吐不是分数，等 v2-mini 出来再用 4GPU 重训更划算。

## 7. 已产出物

- `work_dirs/.../best_coco_bbox_mAP_epoch_12.pth` (159 MB, 已剥离 optim)
- `work_dirs/.../epoch_{10,11,12}.pth`（保留近 3 个 raw ckpt，459 MB 含 optim）
- `work_dirs/.../eval_classwise_e12/20260506_090505/20260506_090505.log` (full classwise dump)
- `docs/tct_ngc_dev32_fullnames_plan_20260505.md`（原计划）
- 本 report
