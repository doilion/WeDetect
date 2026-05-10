_base_ = ["./wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu.py"]

# Phase 3a — Trainable Hierarchical Attribute Fusion (THAF) on a frozen
# XLM-Roberta encoder. Inherits the dev30 disjoint-2gpu chain (full image
# pipeline, 12 epochs, 640×640 cache) and only swaps:
#   1. text_model: PseudoLanguageBackbone (single PSC prompt, frozen) →
#                  PseudoHierarchicalXLMRLanguageBackbone (per-attr cache +
#                  trainable cross-attention fusion module)
#   2. train/test pipeline LoadText → HierarchicalRandomLoadText / HierarchicalLoadText
#   3. class_text_path: tct_ngc_fullnames_30.json (single prompt) →
#                       tct_ngc_fullnames_30_attr_train.json (5-attr list-of-list)
#   4. optim paramwise_cfg adds 10× LR multiplier on fusion submodules
#      (encoder is frozen and not in the optimizer's param group anyway —
#      Pseudo backbone has no encoder params, only fusion module params).

class_text_path = "data/texts/tct_ngc_fullnames_30_attr_train.json"
attr_cache_path = "data/texts/tct_ngc_attr_xlmr_per_attr.pth"

text_channels = 768
num_attr_types = 5

# 1. Swap the text branch to the cache-backed THAF backbone
model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoHierarchicalXLMRLanguageBackbone",
            attr_emb_cache_path=attr_cache_path,
            num_attr_types=num_attr_types,
            embed_dim=text_channels,
            num_heads=8,
            dropout=0.1,
            residual_alpha=0.3,
        ),
    ),
)

# 2. Train pipeline: replace LoadText with HierarchicalRandomLoadText
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

# 3. dataloaders re-point at the new 5-attr class_text_path
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

# 4. paramwise_cfg: keep parent's bias/norm decay rules; no LR multiplier on
# fusion params. We initially tried 10× LR mult (matching YOLO-World-Medical)
# but it caused multi-thousand-fold loss spikes in the first ~10 iters
# because fusion outputs are highly collapsed at random init (class-class
# cos ≈ 0.98) — high LR amplifies bad gradient direction. With 1× mult the
# trajectory still has occasional early spikes (steps 3-5 saw 47K then
# recovered to 40), but converges into the dev30-baseline-equivalent range
# by ~iter 10. Keep paramwise_cfg minimal; rely on warmup (1500 iters) and
# clip_grad max_norm=10 to handle the rest.
optim_wrapper = dict(
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            # Disable weight decay on the two scalar/embedding fusion params
            "backbone.text_model.fusion_query": dict(decay_mult=0.0),
            "backbone.text_model.alpha": dict(decay_mult=0.0),
        },
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu"
