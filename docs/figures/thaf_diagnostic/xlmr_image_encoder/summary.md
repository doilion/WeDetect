# Phase 3.6 image encoder diagnostic — xlmr (768d)

Checkpoint: `work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu/best_coco_bbox_mAP_epoch_10.pth`
Sampled GT bboxes: 900 base + 263 novel (target n_per_class=30)

## Aggregate stats (cosine between image feature at GT bbox center and class vectors)

| metric | base GT bboxes | novel GT bboxes |
|---|---:|---:|
| n | 900 | 263 |
| mean cosine to GT class | -0.285 | -0.304 |
| mean cosine to top-1 class | -0.281 | -0.291 |
| top-1 accuracy (predict correct class) | 54.4% | 3.8% |
| top-1 is BASE class | 87.8% | 70.3% |

## Per-novel-class breakdown

| novel class | n | mean cos to GT | top-1 acc | top-1 is base |
|---|---:|---:|---:|---:|
| respiratory tract-adenocarcinoma | 29 | -0.299 | 0.0% | 93.1% |
| Serous effusion-Ovarian cancer | 30 | -0.304 | 6.7% | 80.0% |
| respiratory tract-Squamous cell carcinoma | 30 | -0.288 | 26.7% | 73.3% |
| Serous effusion-Breast cancer | 30 | -0.302 | 0.0% | 50.0% |
| Thyroid gland-MTC | 30 | -0.332 | 0.0% | 60.0% |
| respiratory tract-Small cell carcinoma | 28 | -0.292 | 0.0% | 96.4% |
| Serous effusion-adenocarcinoma | 28 | -0.295 | 0.0% | 78.6% |
| Thyroid gland-Suspicious for Malignancy | 28 | -0.308 | 0.0% | 57.1% |
| Thyroid gland-Malignant tumour | 30 | -0.317 | 0.0% | 46.7% |

## Decision tree hits

- **B confirmed (image encoder pulls novel toward base)**: 70.3% of novel images' top-1 prediction is a BASE class (out of 39 candidates), despite the GT being novel.
- **B partial: novel image features less aligned with GT class** (novel gt-cos mean -0.304 < base gt-cos mean -0.285, Δ=+0.020)
