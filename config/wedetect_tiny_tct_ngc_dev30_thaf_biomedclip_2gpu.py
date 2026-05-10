_base_ = ["./wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu.py"]

# Phase 3b — Trainable Hierarchical Attribute Fusion (THAF) on a frozen
# BiomedCLIP-PubMedBERT encoder. Same chain as Phase 3a (THAF + XLM-R) but
# with embed_dim 768 → 512 to match BiomedCLIP's native output. Required
# changes vs Phase 3a:
#   1. text_model: PseudoHierarchicalXLMRLanguageBackbone (768d) →
#                  PseudoHierarchicalBiomedCLIPLanguageBackbone (512d)
#   2. attr_cache_path: tct_ngc_attr_xlmr_per_attr.pth (768d) →
#                       tct_ngc_attr_biomedclip_per_attr.pth (512d)
#   3. text_channels 768 → 512 (drives bbox_head.head_module.embed_dims)
#   4. load_from: keep wedetect_tiny.pth — image encoder + neck weights load
#      cleanly; bbox_head.head_module.cls_preds.* (768d) get a strict=False
#      shape mismatch and remain randomly initialized for 512d output. This
#      is intentional: the head must be retrained for the new dim regardless.
#
# Per-attr cache geometry (vs XLM-R):
#   per-attr pairwise cos:   XLM-R 0.89 → BiomedCLIP **0.55**
#   per-class attr_mean cos: XLM-R 0.96 → BiomedCLIP **0.82**
#   novel↔novel max cos:     XLM-R 0.99 → BiomedCLIP **0.95**
# BiomedCLIP is substantially more discriminative on medical fine-grained
# vocabulary, supporting the encoder-swap hypothesis.

class_text_path = "data/texts/tct_ngc_fullnames_30_attr_train.json"
attr_cache_path = "data/texts/tct_ngc_attr_biomedclip_per_attr.pth"

text_channels = 512
num_attr_types = 5

# 1. Swap text branch + adjust head dim to 512
model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoHierarchicalBiomedCLIPLanguageBackbone",
            attr_emb_cache_path=attr_cache_path,
            num_attr_types=num_attr_types,
            embed_dim=text_channels,
            num_heads=8,
            dropout=0.1,
            residual_alpha=0.3,
        ),
    ),
    bbox_head=dict(
        head_module=dict(embed_dims=text_channels),
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10,
    ),
    dict(type="WeDetectKeepRatioResize", scale=(640, 640)),
    dict(
        type="WeDetectLetterResize",
        scale=(640, 640),
        allow_scale_up=True,
        pad_val=dict(img=114),
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(
        type="HierarchicalRandomLoadText",
        num_attr_types=num_attr_types,
        num_neg_samples=(80, 80),
        max_num_samples=80,
        padding_to_max=False,
    ),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pad_param",
            "texts",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="WeDetectKeepRatioResize", scale=(640, 640)),
    dict(
        type="WeDetectLetterResize",
        scale=(640, 640),
        allow_scale_up=False,
        pad_val=dict(img=114),
    ),
    dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
    dict(type="HierarchicalLoadText", num_attr_types=num_attr_types),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pad_param",
            "texts",
        ),
    ),
]

train_dataloader = dict(
    dataset=dict(
        class_text_path=class_text_path,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    dataset=dict(
        class_text_path=class_text_path,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

# Same paramwise_cfg rationale as Phase 3a (no LR multiplier on fusion);
# fusion_query and alpha are scalar/embedding params, no weight decay.
optim_wrapper = dict(
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            "backbone.text_model.fusion_query": dict(decay_mult=0.0),
            "backbone.text_model.alpha": dict(decay_mult=0.0),
        },
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu"
