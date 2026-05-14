_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py"]

# OC-HMTA Module 2 — full hierarchical text adapter on top of Module 1.
#
# Adds to row 3 (M1) baseline:
#   - PseudoMultiAttrLanguageBackbone: loads 5-attr BiomedCLIP cache +
#     per-class metadata tensor (organ_id, axis_id, rank_along_axis).
#   - HierarchicalTextAdapter (Stage 1+2+3):
#       Stage 1: per-attribute proj + content-aware attention pool
#       Stage 2: per-organ soft MoE with prior bias (anti-collapse: prior
#                strongly favors class.organ at init, gate learns nuance)
#       Stage 3: rank embedding additive bypass (cervical 4-axis)
#   - OrganOrdinalLoss: within-organ rank ordinal regression + monotonicity
#   - AdapterCollapseGuard hook: monitors stage1 alpha entropy, stage2 gate
#     entropy, stage3 rank norm; logs every 500 iter + before each val
#
# Anti-collapse design (DEAD-6 prevention) lives in the adapter init:
#   - Orthogonal init scale 0.5 (proj_a avoids identity)
#   - Per-attribute init bias favoring discriminative attrs (a=2,3,4)
#   - Stage 2 prior_bias +5 strength (organ dominance at init ~95%)
#   - Stage 3 rank_emb init σ=0.05 (non-zero start)
#
# All Module 1 components (organ_loss_mask, organ_class_mask, OrganExtractor
# in pipeline) are inherited unchanged.
#
# Train data + schedule: same as row 3 (dev30, 12 ep, lr 3e-4 begin=2).
# Start weights: checkpoints/wedetect_tiny.pth (COCO pretrained), same as
# row 3. NOT loaded from row 3 ckpt — fair comparison.

class_metadata_path = "data/texts/tct_ngc_class_metadata_base30.pt"
text_channels = 512

model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoMultiAttrLanguageBackbone",
            class_metadata_path=class_metadata_path,
            adapter=dict(
                type="HierarchicalTextAdapter",
                embed_dim=text_channels,
                num_attrs=5,
                attr_hidden=128,
                num_organs=5,
                organ_hidden=128,
                max_axes_per_organ=4,
                max_rank_per_axis=7,
                attn_init_bias=(-0.5, -0.5, 0.5, 0.5, 0.5),
                gate_prior_strength=5.0,
                rank_emb_init_std=0.05,
                lambda_pool_entropy=0.02,
                lambda_proj_drift=0.001,
                lambda_gate_entropy=0.02,
                lambda_rank_norm=0.01,
                rank_norm_eps=0.05,
            ),
        ),
    ),
    ordinal_loss=dict(
        type="OrganOrdinalLoss",
        embed_dim=text_channels,
        num_organs=5,
        max_axes_per_organ=4,
        loss_weight=0.3,
        monotonicity_weight=0.5,
    ),
)

# AdapterCollapseGuard: log every 500 iter, log before each val. Halt on
# collapse only after epoch 1 (allow warmup to settle).
custom_hooks = [
    dict(
        type="AdapterCollapseGuard",
        check_interval=500,
        check_at_val=True,
        halt_on_red=False,  # log warnings; manual halt if needed
        alpha_entropy_max=1.5,
        proj_drift_min=0.05,
        gate_entropy_max=1.5,
        organ_dominance_min=0.5,
        rank_norm_min=0.05,
    ),
]

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m2_biomedclip_2gpu"
