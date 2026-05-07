# TCT_NGC dev32 fullnames 2gpu — baseline 综合实验结果（2026-05-06）

> 详细 base 端原因分析另见：`docs/tct_ngc_dev32_test_base_analysis_20260506.md`
> 训练计划与 prompt 设计：`docs/tct_ngc_dev32_fullnames_plan_20260505.md`

## 0. 实验配置

| 项 | 值 |
| --- | --- |
| Config | `config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu.py` |
| Checkpoint | `best_coco_bbox_mAP_epoch_12.pth` |
| 训练数据 | `instances_train_dev.json`（116k imgs / 568 cases / 32 类）|
| Selection 数据 | `instances_val_dev.json`（13k imgs / 533 cases，与 train 同 cohort image-level 9:1 切分）|
| 训练时长 | ~9 小时（2× RTX 3090 + AMP） |
| Effective batch | 32（per-GPU 16 × 2）|
| Optimizer | AdamW，base_lr=3.0e-4，cosine to 0.01×，warmup 1500 iter |
| Epochs | 12（best 即 final） |

---

## 1. Headline 数字

| 评估集 | 类数 | 数据规模 | mAP | mAP_50 | mAP_75 | 性质 |
| --- | --- | --- | --- | --- | --- | --- |
| val_dev (selection) | 25 (exclude-neg) | 13k imgs / 533 cases | **0.413** | 0.601 | 0.477 | image-level CV，乐观 |
| **test_base_clean** | 25 (exclude-neg) | 26k imgs / 89 cases | **0.323** | 0.488 | 0.371 | patient-level hold-out |
| test_base_clean filtered | 13 (test_cases≥5 且非 TCT_CCD) | – | **0.315** | – | – | **真泛化头条**（剔 TCT_CCD 因无 WSI 信息无法验证 hold-out） |
| test_main_novel | 3 | 4030 imgs / 12 cases | **0.012** | 0.018 | 0.014 | zero-shot，prompts 未定稿 |
| test_novel | 5 | 9232 imgs / 34 cases | **0.024** | 0.037 | 0.027 | zero-shot，prompts 未定稿 |
| test_pseudo_novel | 2 | 5202 imgs / 22 cases | **0.061** | 0.094 | 0.067 | zero-shot，prompts 未定稿 |
| hard_test | 4 | 2230 imgs / 16 cases | **0.108** | 0.146 | 0.135 | zero-shot，prompts 未定稿 |

**关键告警：novel 系列的 prompts 是临时编的（`data/texts/tct_ngc_novel_*.json`），没走 base 那种 v2 vetted 流程。novel 数字属于 preliminary，不应作为最终报告。**

---

## 2. Base 端结果（val_dev / test_base_clean）

### 2.1 家族级聚合

| 家族 | n | val mAP | test mAP | gap | test_cases | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| TCT_CCD | 9 | 0.301 | 0.290 | -0.011 | N/A（无 WSI 信息）| 路径无 case 标注，无法验证 patient-level hold-out |
| Urine | 3 | 0.285 | 0.222 | -0.064 | 37 | HGUC 拉稳，AUC/SHGUC 拖低 |
| Thyroid | 6 | 0.482 | 0.337 | -0.145 | 83 | AUC/NS 是塌陷主力 |
| 呼吸 | 6 | 0.561 | 0.408 | -0.152 | 105 | Squamous/Lymphocyte 跌 -0.20 |
| Serous | 1 | 0.507 | 0.337 | -0.170 | 1 | 仅 1 个 case，统计噪声 |

### 2.2 Per-class 完整表（按 test mAP 升序）

| 类 | train | val | test | test_cases | val AP | test AP | Δ | 模式 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Thyroid-AUC | 7762 | 839 | 1287 | 8 | 0.272 | **0.051** | **-0.221** | B 模糊 |
| Urine-AUC | 1820 | 218 | 299 | 13 | 0.191 | 0.112 | -0.079 | C 训练量少 |
| TCT-monilia | 2651 | 264 | 325 | 2 | 0.145 | 0.127 | -0.018 | TCT 噪声 |
| Urine-SHGUC | 2274 | 256 | 357 | 17 | 0.256 | 0.162 | -0.094 | C 训练量少 |
| Thyroid-NS | 9531 | 1114 | 1195 | 10 | 0.350 | 0.162 | -0.188 | B 模糊 |
| TCT-asch | 13866 | 1600 | 1729 | 2 | 0.215 | 0.195 | -0.020 | TCT 噪声 |
| TCT-ec | 9404 | 1220 | 1181 | 2 | 0.213 | 0.195 | -0.018 | TCT 噪声 |
| TCT-vaginalis | 10309 | 1170 | 1335 | 2 | 0.227 | 0.204 | -0.023 | TCT 噪声 |
| 呼吸-Lymphocyte | 56988 | 6451 | 53109 | 19 | 0.418 | 0.213 | -0.205 | D 大类波动 |
| 呼吸-Neutrophil | 81807 | 9174 | 60492 | 22 | 0.378 | 0.270 | -0.108 | D 大类波动 |
| Thyroid-SPTC | 26301 | 2981 | 4487 | 19 | 0.420 | 0.270 | -0.150 | B 模糊 |
| TCT-lsil | 8405 | 960 | 1062 | 2 | 0.325 | 0.298 | -0.027 | TCT 噪声 |
| TCT-ascus | 25140 | 2766 | 3108 | 2 | 0.313 | 0.302 | -0.011 | TCT 噪声 |
| TCT-hsil_scc_omn | 10640 | 1305 | 1338 | 2 | 0.322 | 0.310 | -0.012 | TCT 噪声 |
| Serous-Diseased | 9388 | 971 | 9346 | 1 | 0.507 | 0.337 | -0.170 | 仅 1 case |
| Urine-HGUC | 2046 | 250 | 890 | 7 | 0.409 | **0.391** | **-0.018** | A 形态稳 |
| 呼吸-Ciliated | 59164 | 6667 | 48988 | 19 | 0.501 | 0.407 | -0.094 | D 大类 |
| TCT-agc_em | 10985 | 1330 | 1381 | 2 | 0.411 | **0.449** | **+0.038** | A 形态稳（唯一升） |
| 呼吸-Diseased | 9461 | 1008 | 9636 | 6 | 0.644 | 0.479 | -0.165 | D 大类 |
| Thyroid-FC | 394 | 43 | 407 | 1 | 0.548 | 0.485 | -0.063 | A 形态稳 |
| Thyroid-PTC | 75611 | 8253 | 9336 | 23 | 0.698 | 0.504 | -0.194 | D 大类 |
| 呼吸-Alveolar | 95294 | 10431 | 44439 | 21 | 0.666 | 0.529 | -0.137 | D 大类 |
| TCT-dysbacteriosis | 10128 | 1054 | 1277 | 2 | 0.541 | 0.529 | -0.012 | TCT 噪声 |
| Thyroid-Macrophages | 8858 | 910 | 1703 | 2 | 0.607 | 0.550 | -0.057 | A 形态稳 |
| 呼吸-Squamous | 30347 | 3501 | 24363 | 18 | 0.756 | 0.551 | -0.205 | D 大类 |

**模式图例**：A 形态独特 / B 医学定义模糊 / C 训练量不足 / D 大类跨患者波动 / TCT 噪声（test 仅 2 cases）

### 2.3 Base 端问题分析（精简版）

#### Q1：为什么 val 0.413 → test 0.323，gap -0.090？

不是 bug，是预期。`train_dev` 与 `val_dev` 是 image-level 9:1 随机切分，**同患者的多张 crop 分到 train 和 val**（参见 `docs/tct_ngc_split_audit_20260429.md`）。`test_base_clean` 才是真正的 patient-level hold-out（89 个新病人，与 train 几乎完全 disjoint）。所以 val 的 0.413 系 in-distribution 数字，test 的 0.323 才是真泛化。

#### Q2：为什么 TCT_CCD 家族泛化"完美"（gap -0.011）？

**数据缺失伪象。** TCT_CCD 路径里**根本没有 WSI / case 信息** —— 所有图都在 `TCT_CCD/images/train30000/...` 或 `TCT_CCD/images/val/...` 两个 shard 占位文件夹下，完全没有"哪个病人"的标识。其他 organ（呼吸/Thyroid/Urine/Serous）每张图都带 `Annotated.../CASE-ID/...jpg` 的完整 WSI 命名。

后果：**无法验证** TCT_CCD 的 train/val/test 切分是 patient-level disjoint 还是 image-level random。-0.011 gap 可能是真泛化稳，也可能是 split 内同病人泄漏。**在 WSI 标注补全前，TCT_CCD 任何数字都不应进入头条**。

#### Q3：为什么 Thyroid-AUC -0.221（最大塌陷）？

三层叠加：
1. **医学定义本身模糊**：AUC = Bethesda III "Atypia of Undetermined Significance"，"我说不准"的桶。同家族 Thy-NS（Bethesda I "标本不可诊断"）也塌 -0.188。
2. **bbox 尺寸漂移**：val 中位面积 51983 px²，test 39346 px²，test 的 AUC 框小了 24%，detector 漏检。
3. **val 仅 839 ann + image-level CV**：模型直接过拟合 val_dev 的特定 AUC 分布；test 给到 8 个新 case 的不同形态变异就废。

反例：Thy-Macrophages -0.057 / Thy-FC -0.063 几乎不掉。**形态学独特的类不需要管，定义模糊的类才崩**。

#### Q4：为什么呼吸-Squamous / Lymphocyte 跌幅 -0.20？

不是数据稀缺（train 30k+ / 56k+），是**跨患者染色与形态变异**。同样是 squamous 细胞，不同患者染色程度、炎症程度差异大。Prompt cos > 0.97（呼吸 4 类一片同义）也是部分原因。

#### Q5：为什么 Urine-HGUC 几乎不掉（-0.018），但同家族 SHGUC/AUC 在 test 上还是最低？

HGUC 形态学锚点强（核大、核浆比高、染色质粗）→ prompt 即使 cos 高也能凭视觉特征锁定。
SHGUC/AUC 都是"灰色地带"类，**train 量本身就少**（2274/1820），加上 prompt cos > 0.97 的多家族同质化（NILM/Negative/Negative-Degen/HGUC/SHGUC/AUC 一片高相似度）→ 双重打击。

---

## 3. Novel 端结果（zero-shot，preliminary）

### 3.1 4 个 split 总览

| split | imgs | cases | classes | mAP | mAP_50 | mAP_75 | 唯一类 cat_ids |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test_main_novel | 4030 | 12 | 3 | **0.012** | 0.018 | 0.014 | 24, 25, 26 |
| test_novel (full) | 9232 | 34 | 5 | **0.024** | 0.037 | 0.027 | 21, 22, 24, 25, 26 |
| test_pseudo_novel | 5202 | 22 | 2 | **0.061** | 0.094 | 0.067 | 21, 22 |
| hard_test | 2230 | 16 | 4 | **0.108** | 0.146 | 0.135 | 27, 28, 29, 30 |

### 3.2 Per-class 表（按 split 内 mAP 排）

| split | cat_id | 类 | mAP | mAP_50 | 备注 |
| --- | --- | --- | --- | --- | --- |
| main_3 | 24 | 呼吸-Squamous cell carcinoma | 0.000 | 0.000 | 完全检不出 |
| main_3 | 26 | Thyroid-MTC | 0.000 | 0.000 | 完全检不出 |
| main_3 | 25 | Serous-Breast cancer | 0.036 | 0.053 | 主类竞争少 |
| full_5 | 24 | 呼吸-Squamous cell carcinoma | 0.000 | 0.000 | 同 main_3 |
| full_5 | 26 | Thyroid-MTC | 0.000 | 0.000 | 同 main_3 |
| full_5 | 21 | 呼吸-adenocarcinoma | 0.002 | 0.003 | 几乎检不出 |
| full_5 | 25 | Serous-Breast cancer | 0.003 | 0.003 | 比 main_3 低（被 Ovarian 吃 logit） |
| full_5 | 22 | Serous-Ovarian cancer | 0.116 | 0.181 | Serous 系最稳 |
| pseudo_2 | 21 | 呼吸-adenocarcinoma | 0.004 | 0.005 | 接近 full_5 |
| pseudo_2 | 22 | Serous-Ovarian cancer | 0.118 | 0.184 | 接近 full_5 |
| hard_4 | 30 | Thyroid-Malignant tumour | 0.000 | 0.000 | 完全检不出 |
| hard_4 | 27 | 呼吸-Small cell carcinoma | 0.000 | 0.000 | 完全检不出 |
| hard_4 | 29 | Thyroid-Suspicious for Malignancy | 0.009 | 0.012 | 几乎检不出 |
| hard_4 | 28 | **Serous-adenocarcinoma** | **0.421** | **0.574** | **outlier，存疑** |

### 3.3 Novel 端问题分析

#### Q6：为什么大部分 novel mAP < 0.05？

**符合预期。** dev32 是 32 类闭集训练（fixed prompts + cached XLM-R embeddings + PseudoLanguageBackbone），**不是 open-vocab 训练**。模型从未学过"把 prompt 文字 → 视觉特征"的泛化能力，只学了"32 个固定 prompt embedding 各自对应的视觉锚点"。

零样本要求模型把没见过的 prompt embedding（"Squamous cell carcinoma"）映射到没见过的视觉形态。**当前架构基本不具备这个能力**。要拿到真零样本能力需要：
- 训练时随机 sample prompts（YOLO-World 原版做法）
- 或者解冻 text encoder 一起训
- 或者训练时见到更多类别 + 更多 prompt 变体

#### Q7：为什么 Serous-Ovarian cancer 0.116-0.118、Serous-adenocarcinoma 0.421 异常高？

**Logit 蹭。** 模型训练里有 **Serous effusion-Diseased cells**（base 类，9388 train ann）。这个 base 类对应的 prompt 是 "Serous fluid cytology - Atypical, suspicious, or malignant cells, NOS"，跟 novel 类的 "Serous fluid cytology - Metastatic ovarian carcinoma" / "Serous fluid cytology - Adenocarcinoma NOS" 在 XLM-R embedding 空间距离很近（同 organ + 都含 cancer 关键词）。

模型实际预测的还是"Serous Diseased cells"那个视觉概念，但因为 novel prompt 的 cosine 相似度最高，AP 算到了 novel 类头上。

**Serous-adenocarcinoma 0.421 尤其要打折看**：hard_test 这个 split 里，Serous 类只有 1 个 novel 类，没有竞争 → "Serous Diseased cells" 视觉特征整箱倒进 adenocarcinoma novel 头上。**这不是零样本检测能力，是 Serous 内部 base→novel 标签替换的副作用**。

#### Q8：为什么同一个类（Serous-Breast cancer）在 main_3 和 full_5 数字不同？（0.036 vs 0.003）

**竞争 prompt 越多，零样本 AP 越低**。
- main_3 split：prompts = {Squamous CC, Breast cancer, MTC}，三个 prompt 分布在三个不同器官（呼吸/Serous/Thyroid），Breast 没真正的"对手"
- full_5 split：prompts = {adenocarcinoma_resp, **Ovarian**, Squamous CC, **Breast**, MTC}，Ovarian 和 Breast 都是 Serous 类，且 Ovarian 距离 Serous-Diseased base 类的 embedding 更近 → 把 Breast 的 logit 全吃了

这说明当前数字**对 prompt 集合极度敏感**，不能简单解读为"模型对 Breast cancer 检测能力 X 分"。

#### Q9：为什么呼吸 / Thyroid 系几乎全 0？

呼吸的 novel 类（adenocarcinoma 0.002、Squamous CC 0.000、Small cell 0.000）和 Thyroid novel 类（MTC 0.000、Malignant 0.000、Suspicious 0.009）都基本归零。

原因：
1. 这两个 organ 的 base 类多是细胞分类（Neutrophil、Lymphocyte、Squamous epithelial、PTC 等），novel 是恶性肿瘤类。视觉差距大。
2. base 的 PTC（甲状腺乳头状癌）跟 novel 的 MTC（髓样癌）虽然都是甲状腺恶性，但形态学**完全不同**（MTC 细胞分散、神经内分泌、淀粉样物质；PTC 是核内假包涵体、毛玻璃核），文字 prompt 距离近但视觉锚点不通用。
3. 比 Serous 系蹭分严重的还要差，因为 Serous-Diseased 是个 catch-all 类（什么 atypical 细胞都装），呼吸/Thyroid 的 base 类语义更窄。

#### Q10：当前数字能用吗？

| 用途 | 能否用 |
| --- | --- |
| 论文 / 对外报告 | ❌ 不能。prompts 未 vetted，且基本反映"模型零样本能力近 0" |
| 内部 baseline 备忘 | ✅ 可以，标"preliminary, prompts pending" |
| 后续 v2 prompt 实验对照 | ✅ 可以，作为 prompt 改写收益的下界对比 |
| 决策 "要不要 open-vocab 训练" | ✅ 强信号 —— 当前架构没零样本能力，要做就得改训练范式 |

---

## 4. 行动项（按优先级）

| # | 项 | 状态 | 触发条件 |
| --- | --- | --- | --- |
| 1 | TCT_CCD 补 WSI / case 标注 | 待数据更新 | 当前路径仅 `images/{train30000,val}` 占位，无法验证 patient hold-out |
| 2 | Novel prompt v2 vetted | 你来定 | 重写后才能跑可信 novel eval |
| 3 | Thy-AUC + Thy-NS prompt v2（写形态特征） | 未做 | 1 天能验证 |
| 4 | Urine v2-mini（6 个 prompt 重写） | 未做 | 1 天能验证 |
| 5 | open-vocab 训练范式（prompt 随机化 + 解冻 text encoder） | 未做 | 真零样本能力的根本路径 |
| 6 | 类合并训练（Thy-AUC + NS + Negative） | 未做 | 备选，难度中 |

---

## 5. 关键 artifact 路径

```
work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu/
├── best_coco_bbox_mAP_epoch_12.pth
├── analysis/
│   ├── loss_curves.png                          训练曲线 + val mAP
│   ├── classwise_ap_e12_v2.png                  val_dev classwise（25 类）
│   ├── classwise_ap_test_base_e12_v2.png        test_base classwise（25 类）
│   ├── classwise_ap_val_vs_test_e12.png         val ↔ test 对照
│   ├── classwise_ap_novel_main_3.png            novel main 3 类
│   ├── classwise_ap_novel_full_5.png            novel full 5 类
│   ├── classwise_ap_novel_pseudo_2.png          novel pseudo 2 类
│   └── classwise_ap_novel_hard_4.png            novel hard 4 类
├── eval_classwise_e12/                          val_dev eval log
├── eval_test_base_e12/                          test_base eval log
└── eval_novel_{main_3,full_5,pseudo_2,hard_4}/  4 个 novel eval log
```

```
data/texts/
├── tct_ngc_fullnames_32.json                   ✅ vetted base prompts
├── tct_ngc_fullnames_32_embeddings_wedetect_tiny.pth
├── tct_ngc_novel_{main_3,full_5,pseudo_2,hard_4}.json    ⚠️ placeholder, 待重写
└── tct_ngc_novel_{...}_emb.pth
```

```
docs/
├── tct_ngc_dev32_fullnames_plan_20260505.md           v2 prompt 设计 + 训练计划
├── tct_ngc_dev32_fullnames_baseline_report_20260506.md  最初的 baseline 报告（val 端）
├── tct_ngc_dev32_test_base_analysis_20260506.md       test_base 深度分析
├── tct_ngc_dev32_baseline_full_results_20260506.md    本文件（综合表 + 两端分析）
├── tct_ngc_split_audit_20260429.md                    split 审计
└── tct_ngc_dataset_issue_audit_20260429_zh.md         数据集问题审计
```
