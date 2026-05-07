# TCT_NGC Split Audit

- Dataset root: `/home1/liwenjie/TCT_NGC`
- Generated: `2026-04-29 10:12`
- Full per-class CSV: `work_dirs/tct_ngc_dataset_audit/split_audit_per_class.csv`

## Executive Summary

The `train_dev` and `val_dev` split is mostly behaving like a 9:1 split of `instances_train.json`. The main issue is upstream: several base classes have very different distributions between `instances_train.json` and `instances_test_base.json`.

The most important problem is `Urine-SHGUC`: it is marked as a base class, but training has only 181 annotations while `test_base` has 2706 annotations. This makes base evaluation for that class unfair and unstable. Separately, `Urine-HGUC` is a high-grade malignant category but is assigned to `pseudo_novel`, while the weaker `Urine-SHGUC` category is assigned to `base`; this is clinically and experimentally questionable unless the intended task is explicitly zero-shot transfer from suspicious high-grade to definite high-grade carcinoma.

## High-Risk Findings

| class                          | role         | ontology | train | val_dev | test_base | novel | test/train | flags                                                                                                                                                                                        |
| ------------------------------ | ------------ | -------- | ----- | ------- | --------- | ----- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Thyroid gland-Negative samples | base         | negative | 189   | 19      | 700       | 0     | 3.70       | base_train_lt_300_with_test_base_ge_100, test_base_over_train_ge_3                                                                                                                           |
| Urine-HGUC                     | pseudo_novel | leaf     | 0     | 0       | 0         | 3181  | -          | semantic_review_hguc_is_high_grade_but_not_base                                                                                                                                              |
| Urine-SHGUC                    | base         | leaf     | 181   | 18      | 2706      | 0     | 14.95      | base_train_lt_300_with_test_base_ge_100, val_dev_lt_100_non_negative, test_base_over_train_ge_3, test_base_over_train_ge_1.5_large_non_negative, semantic_review_shguc_base_but_train_sparse |

## Base Classes With Highest Test To Train Ratio

This table highlights base classes whose `test_base` annotation count is large relative to `train`. Large ratios are not always fatal for negative or background classes, but they are problematic for diagnostic disease classes.

| class                                                | ontology | train | val_dev | test_base | test/train |
| ---------------------------------------------------- | -------- | ----- | ------- | --------- | ---------- |
| Urine-SHGUC                                          | leaf     | 181   | 18      | 2706      | 14.95      |
| Thyroid gland-Negative samples                       | negative | 189   | 19      | 700       | 3.70       |
| Serous effusion-Negative samples                     | negative | 6627  | 663     | 19268     | 2.91       |
| Urine-AUC                                            | leaf     | 765   | 76      | 1572      | 2.05       |
| Urine-Negative                                       | negative | 6165  | 616     | 9199      | 1.49       |
| Thyroid gland-FC                                     | leaf     | 436   | 44      | 408       | 0.94       |
| Urine-NILM                                           | negative | 79503 | 7950    | 46782     | 0.59       |
| Thyroid gland-Macrophages                            | leaf     | 7709  | 771     | 3762      | 0.49       |
| respiratory tract-Ciliated columnar epithelial cells | negative | 69962 | 6996    | 25258     | 0.36       |
| respiratory tract-Impurity                           | negative | 46967 | 4697    | 15746     | 0.34       |
| Urine-Negative Degeneration                          | negative | 1745  | 174     | 585       | 0.34       |
| Serous effusion-Diseased cells                       | umbrella | 15046 | 1505    | 4661      | 0.31       |

## Validation Classes Below 100 Annotations

These classes make per-class AP estimates noisy on `val_dev`. The current filtered evaluation rule excludes classes with fewer than 100 validation annotations, plus negative, normal, NILM, and impurity classes.

| class            | ontology | train | val_dev | test_base | reason        |
| ---------------- | -------- | ----- | ------- | --------- | ------------- |
| Thyroid gland-FC | leaf     | 436   | 44      | 408       | val_dev < 100 |
| Urine-AUC        | leaf     | 765   | 76      | 1572      | val_dev < 100 |
| Urine-SHGUC      | leaf     | 181   | 18      | 2706      | val_dev < 100 |

## Urine Category Detail

| class                       | role         | ontology | meta_count | train | val_dev | test_base | test_novel |
| --------------------------- | ------------ | -------- | ---------- | ----- | ------- | --------- | ---------- |
| Urine-NILM                  | base         | negative | 126295     | 79503 | 7950    | 46782     | 0          |
| Urine-Negative              | base         | negative | 15364      | 6165  | 616     | 9199      | 0          |
| Urine-SHGUC                 | base         | leaf     | 2887       | 181   | 18      | 2706      | 0          |
| Urine-AUC                   | base         | leaf     | 2337       | 765   | 76      | 1572      | 0          |
| Urine-Negative Degeneration | base         | negative | 2330       | 1745  | 174     | 585       | 0          |
| Urine-HGUC                  | pseudo_novel | leaf     | 3186       | 0     | 0       | 0         | 3181       |

## Interpretation

- `Urine-SHGUC` is the clearest split defect among diagnostic base classes. It should not be a major contributor to the main base metric unless the train/test split is rebuilt.
- `Urine-AUC` is less extreme but still has only 76 validation annotations and a `test_base/train` ratio above 2, so its validation AP will also be noisy.
- `Thyroid gland-Negative samples` has a high test/train ratio, but it is a negative category and is already excluded from the filtered diagnostic metric.
- `Urine-HGUC` is semantically high-grade disease but is currently only in novel splits. This can be valid only if the experiment is explicitly an open-vocabulary or zero-shot high-grade transfer task. For a closed-set diagnostic detector, it should be moved into base training or merged with a high-grade urine class.

## Recommended Actions

1. Keep the current running experiment for continuity, but report the filtered metric as the main diagnostic metric.
2. For the next dataset revision, rebuild the base train, val, and test split so that diagnostic base classes have adequate train and validation support. `Urine-SHGUC` is the first class to fix.
3. Decide the urine label design explicitly:
   - closed-set option: include `Urine-HGUC` in base train and validation;
   - hierarchy option: keep `AUC`, `SHGUC`, and `HGUC` separate but balance all three;
   - pragmatic option: merge `SHGUC` and `HGUC` into one high-grade urine category if visual separation is too weak or sample counts are limited.
4. Continue excluding negative, normal, NILM, impurity, and validation-count-below-100 classes from the headline filtered AP until the split is rebuilt.
