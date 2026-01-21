_base_ = ['./wedetect_base.py']

# ========== 训练 Pipeline 配置 ==========
# 减小图像尺寸以节省显存 (原始 1280x1280)
img_scale = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='WeDetectKeepRatioResize', scale=img_scale),
    dict(type='WeDetectLetterResize',
         scale=img_scale,
         allow_scale_up=True,
         pad_val=dict(img=114)),
    dict(type='RandomFlip', prob=0.5, _scope_='mmdet'),
    dict(type='RandomLoadText',
         num_neg_samples=(20, 20),
         max_num_samples=20,
         padding_to_max=True,
         padding_value=''),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='WeDetectKeepRatioResize', scale=img_scale),
    dict(type='WeDetectLetterResize',
         scale=img_scale,
         allow_scale_up=False,
         pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts')),
]

# ========== TCT_NGC 数据集配置 ==========
num_classes = 20
num_training_classes = 20
data_root = '/home1/liwenjie/TCT_NGC/'
class_text_path = '/home1/liwenjie/TCT_NGC/texts/tct_class_texts.json'

# 定义 20 个 TCT 类别（必须与标注文件中的 categories 名称完全一致）
tct_classes = (
    'normal', 'ascus', 'asch', 'lsil', 'agc_adenocarcinoma_em',
    'vaginalis', 'dysbacteriosis_herpes_act', 'ec',
    'Serous effusion-Negative samples', 'Serous effusion-Diseased cells',
    'Serous effusion-Breast cancer', 'Thyroid gland-Papillary cancer',
    'Thyroid gland-Negative samples', 'Thyroid gland-Suspicious for Malignancy',
    'Urine-Negative', 'Urine-SHGUC', 'Urine-AUC',
    'respiratory tract-Negative samples', 'respiratory tract-Diseased cells', 'respiratory tract-adenocarcinoma'
)
metainfo = dict(classes=tct_classes)

# ========== 训练超参数（单卡 + Large 模型优化） ==========
max_epochs = 30                    # 大数据集 30-50 轮足够
base_lr = 2e-4                     # 大数据集可用稍大学习率
train_batch_size_per_gpu = 1       # 降低batch_size以节省显存

# ========== 梯度累积（等效更大 batch_size）==========
# 累积 16 次 = 等效 batch_size 16
accumulative_counts = 16

# ========== 模型配置修改 ==========
model = dict(
    num_train_classes=num_classes,
    num_test_classes=num_classes,
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)))

# ========== 数据加载器配置 ==========
# 注意：标注文件中 file_name 已包含 images/train/ 前缀，所以 data_prefix 设为空
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='yolow_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='WeCocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            ann_file='annotations/train_base_v2.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=32)),
        class_text_path=class_text_path,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='WeCocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            test_mode=True,
            ann_file='annotations/test_base_v2.json',
            data_prefix=dict(img='')),
        class_text_path=class_text_path,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# ========== 评估器配置 ==========
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test_base_v2.json',
    metric='bbox')
test_evaluator = val_evaluator

# ========== 优化器配置（使用梯度累积） ==========
optim_wrapper = dict(
    type='AmpOptimWrapper',          # 混合精度训练，节省显存
    accumulative_counts=accumulative_counts,
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05))

# ========== 学习率调度 ==========
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=max_epochs, eta_min=base_lr * 0.01, by_epoch=True)
]

# ========== 加载预训练权重 ==========
load_from = 'checkpoints/wedetect_base.pth'

# ========== Runner 训练配置 ==========
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(interval=50))

# ========== 日志配置 ==========
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
