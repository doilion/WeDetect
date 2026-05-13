# Phase 3.6 image encoder diagnostic — biomedclip (512d)

Checkpoint: `work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu/best_coco_bbox_mAP_epoch_10.pth`
Sampled GT bboxes: 900 base + 263 novel (target n_per_class=30)

## Aggregate stats (cosine between image feature at GT bbox center and class vectors)

| metric | base GT bboxes | novel GT bboxes |
|---|---:|---:|
| n | 900 | 263 |
| mean cosine to GT class | 0.061 | -0.178 |
| mean cosine to top-1 class | 0.082 | -0.024 |
| top-1 accuracy (predict correct class) | 75.2% | 0.4% |
| top-1 is BASE class | 99.9% | 99.2% |

## Per-novel-class breakdown

| novel class | n | mean cos to GT | top-1 acc | top-1 is base |
|---|---:|---:|---:|---:|
| respiratory tract-adenocarcinoma | 29 | -0.131 | 0.0% | 100.0% |
| Serous effusion-Ovarian cancer | 30 | -0.160 | 0.0% | 96.7% |
| respiratory tract-Squamous cell carcinoma | 30 | -0.189 | 0.0% | 100.0% |
| Serous effusion-Breast cancer | 30 | -0.152 | 3.3% | 96.7% |
| Thyroid gland-MTC | 30 | -0.370 | 0.0% | 100.0% |
| respiratory tract-Small cell carcinoma | 28 | -0.129 | 0.0% | 100.0% |
| Serous effusion-adenocarcinoma | 28 | -0.136 | 0.0% | 100.0% |
| Thyroid gland-Suspicious for Malignancy | 28 | -0.189 | 0.0% | 100.0% |
| Thyroid gland-Malignant tumour | 30 | -0.141 | 0.0% | 100.0% |

## Decision tree hits

- **B confirmed (image encoder pulls novel toward base)**: 99.2% of novel images' top-1 prediction is a BASE class (out of 39 candidates), despite the GT being novel.
- **B partial: novel image features less aligned with GT class** (novel gt-cos mean -0.178 < base gt-cos mean 0.061, Δ=+0.239)
