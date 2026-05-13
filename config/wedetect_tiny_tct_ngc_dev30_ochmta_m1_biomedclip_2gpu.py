_base_ = ["./wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py"]

# OC-HMTA Module 1 only — Organ-Conditional class loss masking.
#
# Identical to noTHAF (BiomedCLIP + 1-PSC) in every other respect:
#   - text encoder frozen (PseudoLanguageBackbone, cached 512d BiomedCLIP emb)
#   - image encoder = ConvNext-tiny + FPN, trained
#   - load_from = checkpoints/wedetect_tiny.pth (same start)
#   - 30 class text broadcast, head outputs 30-channel cls_pred
#   - LR + epochs + batch sizes inherited (begin=2 cosine bug fix from clean dev30)
#
# Differences:
#   - YOLOWorldHead.organ_loss_mask = True
#   - YOLOWorldHead.organ_mask_path = .../tct_ngc_class_organ_mask_base30.pt
#     → per-sample cross-organ class BCE is zeroed in loss_by_feat,
#       so the model is never penalized for low score on cross-organ classes
#       (and never rewarded either). image encoder is freed from cross-organ
#       disambiguation (DEAD-7 partial fix).
#   - Both train_pipeline and test_pipeline insert `OrganExtractor` before
#     PackDetInputs, and PackDetInputs.meta_keys includes 'organ_id'/'organ_name'
#     so head sees organ via batch_img_metas[b]['organ_id'].
#
# This is row 3 of the ablation table (M1 only). Row 4 will add Module 2
# (diag-code text adapter), row 5 will add Module 3 (organ aux head if it helps).

organ_mask_path = "data/texts/tct_ngc_class_organ_mask_base30.pt"
taxonomy_path = "data/texts/tct_ngc_taxonomy.json"

model = dict(
    bbox_head=dict(
        type="YOLOWorldHead",
        organ_loss_mask=True,
        organ_mask_path=organ_mask_path,
    ),
)

# ── Pipeline overrides ──
# Inject OrganExtractor before PackDetInputs in train + test pipelines, and
# extend PackDetInputs.meta_keys to carry organ_id through to head.
img_scale = (640, 640)

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
    dict(type="WeDetectKeepRatioResize", scale=img_scale),
    dict(
        type="WeDetectLetterResize",
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114),
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="LoadText"),
    dict(type="OrganExtractor", taxonomy_path=taxonomy_path, strict=True),
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
            "organ_id",
            "organ_name",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="WeDetectKeepRatioResize", scale=img_scale),
    dict(
        type="WeDetectLetterResize",
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114),
    ),
    dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
    dict(type="LoadText"),
    dict(type="OrganExtractor", taxonomy_path=taxonomy_path, strict=True),
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
            "organ_id",
            "organ_name",
        ),
    ),
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline)
)
test_dataloader = val_dataloader

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu"
