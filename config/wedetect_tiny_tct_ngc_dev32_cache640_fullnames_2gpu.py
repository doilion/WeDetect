_base_ = ["./wedetect_tiny_tct_ngc_dev32_fullnames_1gpu.py"]

# Two RTX 3090 full-name baseline on the refreshed 640x640 cached dataset.
data_root = "/home1/liwenjie/TCT_NGC_640/"
train_ann_file = "annotations/instances_train_dev.json"
val_ann_file = "annotations/instances_val_dev.json"

train_batch_size_per_gpu = 16
base_lr = 3.0e-4
max_epochs = 12
warmup_iters = 1500

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=8,
    dataset=dict(dataset=dict(data_root=data_root)),
)
val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    dataset=dict(dataset=dict(data_root=data_root)),
)
test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(lr=base_lr))

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=warmup_iters,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.01,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
    ),
]

val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu"
