# dev32 → dev30 cat_id remap (train_dev_disjoint, 683,135 anns total)

## Urine 3-way merge (verified Δ=0)

| dev32 cat | dev32 name | anns | → | dev30 cat | dev30 name | anns |
|---:|---|---:|---|---:|---|---:|
| 16 | Urine-NILM | 90,880 | → | 16 | Urine-NHGUC | 103,390 |
| 17 | Urine-Negative | 10,775 | → | 16 | (merged) | |
| 20 | Urine-Negative Degeneration | 1,735 | → | 16 | (merged) | |
| | **sum** | **103,390** | | | **103,390** | **Δ=0** |

## Shifted cat_ids (post-merge, ann counts preserved)

| dev30 cat | dev32 cat | name | train anns |
|---:|---:|---|---:|
| 0–15 | 0–15 | (unchanged: respiratory tract, serous effusion, thyroid 0..15) | — |
| 16 | 16+17+20 | Urine-NHGUC | 103,390 |
| 17 | 18 | Urine-SHGUC | 1,961 |
| 18 | 19 | Urine-AUC | 1,640 |
| 19 | 23 | Urine-HGUC | 1,957 |
| 20 | 31 | TCT_CCD-normal | 31,808 |
| 21 | 32 | TCT_CCD-ascus | 22,325 |
| 22 | 33 | TCT_CCD-asch | 12,373 |
| 23 | 34 | TCT_CCD-lsil | 7,492 |
| 24 | 35 | TCT_CCD-hsil_scc_omn | 9,556 |
| 25 | 36 | TCT_CCD-agc_adenocarcinoma_em | 9,852 |
| 26 | 37 | TCT_CCD-vaginalis | 9,183 |
| 27 | 38 | TCT_CCD-monilia | 2,332 |
| 28 | 39 | TCT_CCD-dysbacteriosis_herpes_act | 8,946 |
| 29 | 40 | TCT_CCD-ec | 8,499 |

## Counts (parity check)

| split | imgs | anns | cats |
|---|---:|---:|---:|
| dev32 train_dev_disjoint | 103,604 | 683,135 | 32 |
| dev30 train_dev_disjoint | 103,604 | 683,135 | 30 |

Same images, same annotation count — only `category_id` field changes (and merged labels collapse).

## NHGUC dominance

NHGUC (id 16) holds **103,390 / 683,135 = 15.1%** of all training annotations,
making it the **largest single class** in dev30 (Urine-Negative dominance was
already a flag in the dev32 baseline §10 root-cause analysis; merging
preserves that imbalance, but eliminates the cos≥0.97 prompt collisions
between the 3 originally-distinct Urine negative wordings).
