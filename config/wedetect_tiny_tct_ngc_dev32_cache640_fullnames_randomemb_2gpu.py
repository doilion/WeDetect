_base_ = ["./wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu.py"]

# Two RTX 3090 ablation for the refreshed 32-class full-name setup.
# The class text strings are unchanged, but PseudoLanguageBackbone receives
# fixed random normalized vectors instead of XLM-R cached text embeddings.
text_embed_path = "data/texts/tct_ngc_fullnames_32_random_embeddings_seed20260506.pth"

model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoLanguageBackbone",
            text_embed_path=text_embed_path,
        ),
    ),
)

train_batch_size_per_gpu = 16
base_lr = 3.0e-4
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

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_randomemb_2gpu"
