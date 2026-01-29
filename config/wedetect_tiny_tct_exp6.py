_base_ = ["default_runtime.py"]

# ======================= 实验6: 强数据增强 =======================
# 目的: 验证强数据增强(颜色抖动+垂直翻转)对检测性能的影响 (对比 exp3)
# 基础配置: exp3 (Loss调整: cls=1.0, bbox=5.0)

data_root = "/home1/liwenjie/TCT_NGC/"
train_class_text_path = "data/texts/tct_ngc_v2_base_class_texts.json"
test_class_text_path = "data/texts/tct_ngc_v2_class_texts.json"

base_classes = (
    'normal', 'ascus', 'asch', 'lsil', 'agc_adenocarcinoma_em',
    'vaginalis', 'dysbacteriosis_herpes_act', 'ec',
    'Serous effusion-Negative samples', 'Serous effusion-Diseased cells',
    'Serous effusion-Breast cancer', 'Thyroid gland-Papillary cancer',
    'Thyroid gland-Negative samples', 'Thyroid gland-Suspicious for Malignancy',
    'Urine-Negative', 'Urine-SHGUC', 'Urine-AUC',
    'respiratory tract-Negative samples', 'respiratory tract-Diseased cells',
    'respiratory tract-adenocarcinoma',
)

novel_classes = (
    'hsil_scc', 'monilia', 'Serous effusion-Ovarian cancer',
    'Serous effusion-Adenocarcinoma', 'Thyroid gland-Suspicious for Papillary Cancer',
    'Thyroid gland-Atypia of Undetermined Significance', 'Thyroid gland-Malignant',
    'Thyroid gland-Nondiagnostic or Unsatisfactory', 'Urine-HGUC',
    'respiratory tract-squamous carcinoma', 'respiratory tract-small cell carcinoma',
)

all_classes = base_classes + novel_classes
dataset_metainfo = dict(classes=base_classes)

# ======================= 实验6 超参数 =======================
num_classes = 31
num_training_classes = 20
max_epochs = 12
close_mosaic_epochs = 2
save_epoch_intervals = 1
text_channels = 768
neck_embed_channels = [128, 256, 512]
neck_num_heads = [4, 8, 16]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 10

find_unused_parameters = True

model_test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.7),
    max_per_img=300)

tal_topk = 10
tal_alpha = 0.5
tal_beta = 2.0

# 保持 exp3 的 Loss 权重
loss_cls_weight = 1.0
loss_bbox_weight = 5.0
loss_dfl_weight = 1.5 / 4

custom_imports = dict(imports=["wedetect"], allow_failed_imports=False)

# ======================= 模型配置 =======================
model = dict(
    type="YOLOWorldDetector",
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type="YOLOWDetDataPreprocessor",
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type="MultiModalYOLOBackbone",
        image_model=dict(
            type="ConvNextVisionBackbone",
            model_name="tiny",
            frozen_modules=[],
        ),
        text_model=dict(
            type="XLMRobertaLanguageBackbone",
            model_name="./xlm-roberta-base/",
            model_size="tiny",
            frozen_modules=[],
        ),
    ),
    neck=dict(
        type="CSPRepBiFPANNeck",
        model_size='tiny',
    ),
    bbox_head=dict(
        type="YOLOWorldHead",
        head_module=dict(
            type="YOLOWorldHeadModule",
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes,
            model_size='tiny',
            in_channels=[96, 192, 384],
        ),
        prior_generator=dict(
            type='MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='WeDetectDistancePointBBoxCoder'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='mmyoloIoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_training_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)),
    test_cfg=model_test_cfg)

img_scale = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    # 实验6: 强数据增强
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2),
         hue_delta=10),
    dict(type='WeDetectKeepRatioResize', scale=img_scale),
    dict(
        type='WeDetectLetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),  # 新增垂直翻转
    dict(type="LoadText"),
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
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='WeDetectKeepRatioResize', scale=img_scale),
    dict(
        type='WeDetectLetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type="LoadText"),
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

train_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="WeCocoDataset",
        data_root=data_root,
        test_mode=False,
        ann_file="annotations/train_base_v2.json",
        data_prefix=dict(img=""),
        filter_cfg=None,
        metainfo=dataset_metainfo,
    ),
    class_text_path=train_class_text_path,
    pipeline=train_pipeline,
)

test_base_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="WeCocoDataset",
        data_root=data_root,
        test_mode=True,
        ann_file="annotations/test_base_v2.json",
        data_prefix=dict(img=""),
        batch_shapes_cfg=None,
        metainfo=dataset_metainfo,
    ),
    class_text_path=train_class_text_path,
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolow_collate'),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_base_dataset,
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/test_base_v2.json",
    metric="bbox",
)

test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
        }),
    clip_grad=dict(max_norm=10.0),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

dist_cfg = dict(backend="nccl", timeout=10800)
work_dir = './work_dirs/wedetect_tiny_tct_exp6'
load_from = 'checkpoints/wedetect_tiny.pth'
resume = False
