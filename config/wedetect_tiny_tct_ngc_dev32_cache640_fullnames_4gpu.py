_base_ = ["./wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu.py"]

# Four RTX 3090 full-name baseline on the refreshed 640x640 cached dataset.
# Sub-linear LR scaling vs the 2gpu baseline (3.0e-4 -> 4.5e-4 instead of 6.0e-4):
# fine-tuning from wedetect_tiny.pth at effective batch 64 prefers a more
# conservative LR than strict linear scaling, especially with several prompt
# pairs at cosine similarity > 0.98 in the contrastive head.
train_batch_size_per_gpu = 16
base_lr = 4.5e-4
max_epochs = 12
warmup_iters = 1500

train_dataloader = dict(batch_size=train_batch_size_per_gpu, num_workers=8)
val_dataloader = dict(batch_size=train_batch_size_per_gpu, num_workers=4)
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

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=5,
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_4gpu"
