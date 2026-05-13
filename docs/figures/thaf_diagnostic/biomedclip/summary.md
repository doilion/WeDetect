# THAF fusion diagnostic — biomedclip (512d)

Checkpoint: `work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu/best_coco_bbox_mAP_epoch_10.pth`
Classes analyzed: 30 base + 9 novel = 39

## Trained fusion stats

- alpha (learnable residual weight): **-0.0001** (init=0.3)
- attr_type_embed L2 norms (5 channels in order organ_specimen, diagnostic_code, cytomorphology, background_and_immunoprofile, key_distinguishing_feature): [0.3654, 0.3709, 0.3646, 0.3729, 0.3496]

| metric | trained fusion | attr-mean baseline | Δ |
|---|---:|---:|---:|
| base↔base off-diag mean cos | 0.815 | 0.836 | -0.021 |
| base↔base off-diag max  cos | 0.962 | 0.967 | -0.005 |
| novel↔novel off-diag mean cos | 0.877 | 0.891 | -0.013 |
| novel↔novel off-diag max  cos | **0.940** | 0.947 | -0.006 |
| novel→base avg cos | 0.823 | 0.842 | -0.020 |

## Phase 2 reference (Phase 2.1 cos heatmap, single-encoder static aggregation)

| method | novel↔novel max cos |
|---|---:|
| v2_psc_single_prompt | 0.996 |
| 5attr_static_sum | 0.993 |
| 5attr_static_weighted | 0.991 |
| 5attr_static_concat | 0.971 |
| 5attr_static_only_distinguish | 0.971 |

## Decision tree hits

- **A refuted (fusion separates novel better)**: trained novel↔novel max 0.940 < attr_mean 0.947
- alpha=-0.000 ≈ init — fusion light, attr_mean dominant

## Plots

- `cosine_heatmap_trained.png` — 39×39 cosine, red lines split base/novel
- `cosine_heatmap_attr_mean.png` — alpha=0 equivalent (untrained fusion)
