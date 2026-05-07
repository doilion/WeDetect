# TCT_NGC 数据集划分问题审计

- 数据集根目录：`/home1/liwenjie/TCT_NGC`
- 审计时间：`2026-04-29 10:15 PDT`
- 主要输入：`annotations/*.json`、`metadata/*.json`、`source_maps/*.json`
- 详细表格：
  - `work_dirs/tct_ngc_dataset_audit/split_audit_per_class.csv`
  - `work_dirs/tct_ngc_dataset_audit/test_base_category_by_wsi_role.csv`
  - `work_dirs/tct_ngc_dataset_audit/metadata_count_mismatch.csv`
  - `work_dirs/tct_ngc_dataset_audit/train_val_case_overlap_by_source.csv`

## 总结

当前 `train_dev` 和 `val_dev` 本身没有明显 9:1 切分错误：所有 base 类都满足 `train_dev + val_dev = train`，每类 `val_dev` 基本是原始 train 的十分之一。

主要问题不在开发集 9:1 切分，而在更上游的类别角色和 test_base 构造方式：

1. `Urine-SHGUC` 是最严重的问题。它被设为 base 类，但 train 只有 181 个标注，val_dev 只有 18 个标注，而 test_base 有 2706 个标注，test/train 比例达到 14.95。
2. `Urine-HGUC` 是确定高级别尿路上皮癌，却被放在 `pseudo_novel`；同时较弱的 `Urine-SHGUC` 放在 base。除非明确做“从 suspicious high-grade 到 definite HGUC 的零样本迁移”，否则这个划分在诊断语义上不合理。
3. `test_base` 不是纯 base 病例测试集。它是“base 类别标注”的测试集，里面大量图片来自 `pseudo_novel` 或 `main_novel` 病例。对尿液类影响尤其大：`Urine-SHGUC` 在 test_base 的 2706 个标注中，2621 个来自 `pseudo_novel` 病例，只有 85 个来自 base 病例。
4. `Urine-AUC` 有类似但稍轻的问题：train 765、val_dev 76、test_base 1572，其中 test_base 1432 个标注来自 `pseudo_novel` 病例。
5. NGC 的 `train_dev` 和 `val_dev` 是图片级切分，不是病例级切分。图片没有重复，但 `case_sig` 在 train 和 val 之间有重叠；如果用 val 做早停或模型选择，指标会偏乐观。
6. `metadata/label_map_v2.json` 的 `count` 和实际 COCO JSON 中可训练或可评估的标注数不是同一口径。训练和评估统计应以 `annotations/*.json` 实际计数为准。

## 高风险类别

| 类别 | role | ontology | train | val_dev | test_base | test/train | 问题 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `Urine-SHGUC` | base | leaf | 181 | 18 | 2706 | 14.95 | base 训练极少，测试极多，且 test_base 主要来自 HGUC 相关 novel 病例 |
| `Urine-HGUC` | pseudo_novel | leaf | 0 | 0 | 0 | - | 确定 HGUC 被放到 novel，而 SHGUC 放在 base，语义边界不自然 |
| `Urine-AUC` | base | leaf | 765 | 76 | 1572 | 2.05 | val 样本低于 100，test_base 大部分来自 novel 病例 |
| `Thyroid gland-FC` | base | leaf | 436 | 44 | 408 | 0.94 | val 样本低于 100，单类 AP 波动会很大 |
| `Thyroid gland-Negative samples` | base | negative | 189 | 19 | 700 | 3.70 | negative 类，本身应从主要诊断 AP 中排除，但 split 数量很不均衡 |

## test_base 中来自非 base 病例的 base 标注

`test_base` 里有 37867 张图片，其中图片级 `wsi_role` 分布如下：

| wsi_role | 图片数 |
| --- | ---: |
| base | 24273 |
| pseudo_novel | 10232 |
| main_novel | 3362 |

按标注数看，`test_base` 共 255937 个 base 类标注，其中 46724 个来自 `pseudo_novel` 病例，12459 个来自 `main_novel` 病例。

非 base 病例占比高的类别如下：

| 类别 | test_base 标注 | base 病例标注 | pseudo_novel 病例标注 | main_novel 病例标注 | 非 base 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Urine-SHGUC` | 2706 | 85 | 2621 | 0 | 96.9% |
| `Urine-AUC` | 1572 | 140 | 1432 | 0 | 91.1% |
| `Serous effusion-Negative samples` | 19268 | 4883 | 5105 | 9280 | 74.7% |
| `Urine-Negative` | 9199 | 2572 | 6627 | 0 | 72.0% |
| `respiratory tract-Impurity` | 15746 | 6538 | 6136 | 3072 | 58.5% |
| `Urine-NILM` | 46782 | 24451 | 22331 | 0 | 47.7% |
| `Urine-Negative Degeneration` | 585 | 416 | 169 | 0 | 28.9% |

这个设计可以解释为“在 novel 病例背景中同时评估 base 类识别能力”，但不能把它理解成纯 base 病例分布。尤其 `Urine-SHGUC` 和 `Urine-AUC` 是诊断性类别，不只是背景或阴性类别，因此会明显影响 base AP。

## 语义层级检查

### 尿液

尿液部分存在真正的语义划分问题：

- `Urine-AUC` 和 `Urine-SHGUC` 在 base。
- `Urine-HGUC` 在 `pseudo_novel`。
- 但 `SHGUC` 是 suspicious for high-grade urothelial carcinoma，和 `HGUC` 是同一高级别诊断链条，不像普通阴性背景类。

如果目标是闭集诊断检测，建议把 `HGUC` 放入 base 并重新平衡 `AUC`、`SHGUC`、`HGUC`。如果目标是开放词表或零样本实验，建议明确命名为 high-grade transfer，并且单独报告 `Urine-HGUC` novel 结果，不要让 `Urine-SHGUC` 的 base AP 混进主结论。

### 呼吸道和浆膜腔

呼吸道和浆膜腔也有“umbrella base，具体癌种 novel”的结构：

- `respiratory tract-Diseased cells` 是 base，具体 `adenocarcinoma`、`Squamous cell carcinoma`、`Small cell carcinoma` 是 novel 或 hard_novel。
- `Serous effusion-Diseased cells` 是 base，具体 `Ovarian cancer`、`Breast cancer`、`adenocarcinoma` 是 novel 或 hard_novel。

这更像预期的层级开放词表设计，不是和尿液完全同类的问题。但报告时需要说明：base 的 diseased umbrella 类不能直接等价于具体癌种的闭集训练。

### 甲状腺

甲状腺存在层级重叠，但风险低于尿液：

- `Thyroid gland-PTC` 和 `Thyroid gland-SPTC` 都在 base。
- `Thyroid gland-MTC` 在 `main_novel`。
- `Thyroid gland-Suspicious for Malignancy` 和 `Thyroid gland-Malignant tumour` 在 `hard_novel`，属于 umbrella 或 hard 测试类。

这可以作为 hard novel 评估，但如果做闭集诊断，`Suspicious for Malignancy` 和 `Malignant tumour` 这类 umbrella 标签会和具体肿瘤类发生语义重叠，需要单独解释。

### 宫颈

宫颈 10 类都在 base，没有 base/novel 划分冲突。需要注意的是类别本身有合并标签：

- `TCT_CCD-hsil_scc_omn` 合并 HSIL 和 SCC。
- `TCT_CCD-agc_adenocarcinoma_em` 合并 AGC、AIS、腺癌和内膜来源。
- `TCT_CCD-dysbacteriosis_herpes_act` 合并 BV、HSV、Actinomyces。

这些不是 split 错误，而是标签粒度问题。训练时可以用，但解释 AP 时不能当成单一细胞形态类别。

## val_dev 低样本类别

当前 filtered evaluation 规则排除 negative、normal、NILM、impurity，以及 val 标注数低于 100 的类别。按当前数据，非 negative 且 `val_dev < 100` 的 base 类是：

| 类别 | train | val_dev | test_base | 建议 |
| --- | ---: | ---: | ---: | --- |
| `Thyroid gland-FC` | 436 | 44 | 408 | 主 AP 中排除或只作参考 |
| `Urine-AUC` | 765 | 76 | 1572 | 主 AP 中排除，等重划分后再纳入 |
| `Urine-SHGUC` | 181 | 18 | 2706 | 主 AP 中排除，优先修复 split |

## train_dev 和 val_dev 的病例重叠

图片级别没有发现 train_dev 和 val_dev 重复，但 NGC 病例签名 `case_sig` 有重叠：

| source_category | train_cases | val_cases | overlap_cases |
| --- | ---: | ---: | ---: |
| `Serous_effusion` | 9 | 9 | 9 |
| `Thyroid_gland` | 197 | 190 | 190 |
| `Urine` | 104 | 75 | 74 |
| `respiratory_tract` | 124 | 118 | 117 |
| `TCT_CCD` | 27000 | 3000 | 0 |

这符合 README 里写的“NGC 是 image-level split，case_sig 只审计不强制”。它适合做训练过程监控，但如果要用 val_dev 做严肃模型选择，最好补一个病例级验证集。

## metadata count 口径问题

`label_map_v2.json` 里的 `count` 不应直接当成训练或评估标注数。以下类别差异较大：

| 类别 | metadata count | 实际 disjoint split sum | 差异 |
| --- | ---: | ---: | ---: |
| `Thyroid gland-Negative samples` | 9424 | 889 | 8535 |
| `respiratory tract-Impurity` | 63687 | 62713 | 974 |
| `Serous effusion-Negative samples` | 26650 | 25895 | 755 |

这不一定代表图片缺失，更可能是导出、过滤、小框删除或统计口径不同造成的。后续所有训练和评估分析应读取 `annotations/*.json` 的实际标注数。

## 当前建议

1. 当前正在跑的实验可以继续，用于保留连续性和对比。
2. 主结果应使用 filtered AP：排除 negative、normal、NILM、impurity，以及 val 标注数低于 100 的类。
3. 当前版本里建议额外谨慎看待 `Urine-AUC` 和 `Urine-SHGUC`，因为它们不只是低样本，还存在 test_base 大量来自 HGUC novel 病例的问题。
4. 下一版数据集优先修复尿液：
   - 闭集方案：把 `Urine-HGUC` 放入 base，重新平衡 `AUC`、`SHGUC`、`HGUC` 的 train、val、test。
   - 开放词表方案：保留 `HGUC` 为 novel，但从 `test_base` 中单独剥离来自 HGUC novel 病例的 `AUC` 和 `SHGUC` 标注，或至少单独报告“novel 病例背景下的 base 标注 AP”。
   - 简化方案：如果 `SHGUC` 和 `HGUC` 视觉边界太弱，可以考虑合并为一个 high-grade urine 类，再重新划分。
5. 如果 val_dev 会用于选 checkpoint，建议另外构建病例级 dev 验证集，避免同一病例签名同时出现在 train 和 val。

