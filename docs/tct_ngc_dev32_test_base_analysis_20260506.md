# TCT_NGC dev32 baseline — val→test 泛化分析（2026-05-06，修订）

## TL;DR

- best_e12 在 `val_dev` 上 mAP **0.413**，在 `test_base_clean` 上 mAP **0.323**，gap **-0.090**。
- **这个 gap 不是泄漏 bug，是预期行为**。审计文档已确认：dev split (train_dev / val_dev) 是 image-level 9:1 随机切分，**dev 内部同病人不同 crop 分到 train 和 val，是 by design**。`test_base_clean` 才是真正的 patient-level hold-out cohort。
- 因此 **0.323 (test) 才是 baseline 头条数字**，0.413 (val) 只是 model selection 用的辅助指标。
- 进一步过滤 TCT_CCD（**路径里根本没有 WSI/case 信息**，是 `TCT_CCD/images/train30000/...` 这种 dataset shard 命名，无法验证 patient-level 分开）和其他 organ 中 test_cases < 5 的类后，**真泛化 mAP = 0.315 (13 类)**，val→test gap **-0.143**。

---

## 1. Split 设计回顾

| split | imgs | unique cases | 与 train_dev 重合 cases |
| --- | --- | --- | --- |
| train_dev | 116524 | 568 | – |
| val_dev | 12947 | **533** | **533 (100%)** |
| test_base_clean | 26257 | **89** | 3 (3.4%) |

- train_dev / val_dev 文件级 0 重合，但同一 case 的多张 crop 9:1 进 train 和 val（典型样本：`VI.LB2103543-...PTC` 在 train 143 张 / val 17 张）
- 这是 **CV-on-crops within shared cohort** 的设计，目的是给 model selection 一个低方差信号
- **patient-to-patient generalization 用 test_base_clean 衡量** —— 89 个新 case 与 train 几乎完全 disjoint
- 如果排除路径中没有 WSI/case 信息的 TCT_CCD，case 统计变为 train/val/test = 566/531/87，test 与 train 仅重合 1 个 case。

---

## 2. 真泛化数字（去掉统计不可信的类）

### 不应进入头条的类

| 排除原因 | 类 | test_cases |
| --- | --- | --- |
| **TCT_CCD 路径无 WSI 信息**，无法验证 patient-level hold-out | 9 个 TCT_CCD-* 类 | 不可知 |
| test_cases < 5（统计噪声） | Serous-Diseased (1)、Thy-FC (1)、Thy-Macrophages (2) | 1-2 |

### 真 hold-out 13 类

| 类 | val | test | Δ | test_cases |
| --- | --- | --- | --- | --- |
| respiratory-Squamous | 0.756 | 0.551 | -0.205 | 18 |
| Thyroid-PTC | 0.698 | 0.504 | -0.194 | 23 |
| respiratory-Alveolar | 0.666 | 0.529 | -0.137 | 21 |
| respiratory-Diseased | 0.644 | 0.479 | -0.165 | 6 |
| respiratory-Ciliated | 0.501 | 0.407 | -0.094 | 19 |
| Thyroid-SPTC | 0.420 | 0.270 | -0.150 | 19 |
| respiratory-Lymphocyte | 0.418 | 0.213 | -0.205 | 19 |
| Urine-HGUC | 0.409 | **0.391** | **-0.018** ← 最稳 | 7 |
| respiratory-Neutrophil | 0.378 | 0.270 | -0.108 | 22 |
| Thyroid-NS | 0.350 | 0.162 | -0.188 | 10 |
| Thyroid-AUC | 0.272 | **0.051** | **-0.221** ← 最差 | 8 |
| Urine-SHGUC | 0.256 | 0.162 | -0.094 | 17 |
| Urine-AUC | 0.191 | 0.112 | -0.079 | 13 |

**头条 mAP** = 13 类均值 = **0.315**（val 对应 0.458，gap -0.143）

---

## 3. 为什么有的类掉得多，有的掉得少？

### 模式 A：形态学独特 → 泛化稳

| 类 | gap | 形态特征 |
| --- | --- | --- |
| Urine-HGUC | -0.018 | 高级别尿路上皮癌：核大、核浆比高、染色质粗 → 跨患者特征稳定 |
| TCT-agc_em | +0.038 | 腺癌细胞：核明显异型、腺样排列 → 跨患者还能涨 |

→ 这些类不需要动 prompt，已经在用形态学锚点。

### 模式 B：医学定义模糊 → 泛化崩

| 类 | gap | Bethesda |
| --- | --- | --- |
| Thy-AUC | **-0.221** | III（Atypia of Undetermined Significance）"我说不准" |
| Thy-NS | -0.188 | I（Nondiagnostic）"标本不可诊断" |
| Thy-PTC | -0.194 | VI（已有 Macrophages -0.057 反例，但 PTC 本身因细胞数多类内方差大） |
| Thy-SPTC | -0.150 | V（Suspicious for malignancy）"疑似恶性" |

→ **整个甲状腺的"灰色地带"类对新病人都崩**。同家族**形态独特**的 Macrophages -0.057 / FC -0.063 几乎不掉，证明不是甲状腺数据本身有问题，而是**模糊定义类**天然患者间方差大。

### 模式 C：训练集严重不足 → val 都低，test 更低

| 类 | train | val | test | val AP | test AP |
| --- | --- | --- | --- | --- | --- |
| Urine-SHGUC | 2274 | 256 | 357 | 0.256 | 0.162 |
| Urine-AUC | 1820 | 218 | 299 | 0.191 | 0.112 |
| Thy-AUC | 7762 | 839 | 1287 | 0.272 | 0.051 |

→ Urine-SHGUC / Urine-AUC 训练 ann 数都 < 2500，外加 prompt cos > 0.97 多家齐聚（NILM/Negative/Negative-Degen/HGUC/SHGUC/AUC 一片高相似度），双重打击。

### 模式 D：呼吸 4 类（Squamous/Lymphocyte 等）-0.20 跌幅

呼吸-Squamous 0.756 → 0.551 / 呼吸-Lymphocyte 0.418 → 0.213 这种单类大跌主要是：
- 呼吸细胞跨患者变异大（同样是 Squamous，不同患者炎症程度、染色情况差异大）
- 训练 ann 数其实够（30k-95k），不是数据稀缺问题
- prompt cos 高邻居（呼吸 4 类一片同义）也是部分原因

---

## 4. TCT_CCD 的"完美泛化"是数据缺失伪象

| TCT_CCD 类 | val AP | test AP | gap | path 结构 |
| --- | --- | --- | --- | --- |
| 全部 9 个 | 0.213 ~ 0.541 | 0.127 ~ 0.529 | -0.038 ~ +0.038 | `TCT_CCD/images/{train30000,val}/img.jpg` |

**TCT_CCD 路径中根本不带 WSI / case 信息**：所有图都在 `TCT_CCD/images/train30000/` 或 `TCT_CCD/images/val/` 两个 shard 文件夹底下，**没有任何标识告诉我们这张图来自哪个病人**。这跟其他 organ（呼吸 / Thyroid / Urine / Serous）每张图都带 `Annotated...session/CASE-ID/...jpg` 的完整 WSI 命名形成鲜明对比。

后果：
1. 我们**无法验证**这次 train_dev / val_dev / test_base_clean 的 TCT_CCD 切分是不是 patient-level disjoint
2. 也**无法判断** TCT_CCD 的 -0.011 gap 是真泛化稳定 还是 split 内泄漏
3. **任何 TCT_CCD 数字都不应进入头条**，至少在 WSI 标注补全前

audit doc 已经在 "high-risk findings" 里指出过 Urine-SHGUC train=181 vs test=2706 的问题。**TCT_CCD 缺 WSI 标注应该作为下一个 audit findings 加进去**。

---

## 5. 行动项

### 不需要动的

- ❌ 不动 dev/val 切分逻辑 —— 当前 image-level CV 是 by design，对 model selection 有用
- ❌ 不立即重训 —— 0.323 是合理的 baseline 起点

### 应该做的

1. **TCT_CCD 补 WSI / case 标注** —— 下次数据更新最优先；当前路径完全没有 WSI 信息（仅 `images/{train30000,val}` 两个 shard 占位），让 TCT_CCD 在所有评估里都不可信。补完后才能验证 patient-level disjoint。
2. **Thy-AUC + Thy-NS prompt v2 重写** —— 写形态学描述（核增大、染色质粗、但不到恶性标准），不写 Bethesda 编号或"undetermined significance"，模型从图像看不出文字标签。
3. **Thy-AUC + Thy-NS + Thy-Negative 合并训练**（医学合规允许的话）—— 三类灰色地带相互吃 logit，合并成"non-malignant indeterminate Thyroid" 训，下游再用规则细分。这是降模型负担最快的方法。
4. **Urine v2-mini prompt** —— 之前 plan 已经设计好的 6 个 Urine prompt 重写值得一试，AUC 0.112 + SHGUC 0.162 在新病人上都低，prompt 嫌疑被 test 集二次确认。

### 监控

5. **headline 数字用 13 类 filtered test mAP = 0.315，不要用 25 类 test mAP = 0.323**（后者含 TCT_CCD 9 类，在 WSI 信息补全前不应进入头条）。
6. **新实验 model selection 仍用 val_dev**（作为快速信号），但**最终对外报告必须用 test_base_clean**。
7. baseline_report 里 "exclude-negative mAP 0.413" 应该改成 "val_dev (selection set) mAP 0.413 / test_base_clean (held-out) mAP 0.323 / 13-class hold-out mAP 0.315"。

---

## 6. 关键 artifact 路径

- val 评估 log: `work_dirs/.../eval_classwise_e12/20260506_090505/`
- test 评估 log: `work_dirs/.../eval_test_base_e12/20260506_210717/`
- val/test 对照图: `work_dirs/.../analysis/classwise_ap_val_vs_test_e12.png`
- val 单图: `work_dirs/.../analysis/classwise_ap_e12_v2.png`
- test 单图: `work_dirs/.../analysis/classwise_ap_test_base_e12_v2.png`
- 训练曲线: `work_dirs/.../analysis/loss_curves.png`
- audit (split): `docs/tct_ngc_split_audit_20260429.md`
- audit (dataset issues): `docs/tct_ngc_dataset_issue_audit_20260429_zh.md`
