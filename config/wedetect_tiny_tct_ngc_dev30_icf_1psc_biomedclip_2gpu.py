_base_ = ["./wedetect_tiny_tct_ngc_dev30_icf_biomedclip_2gpu.py"]

# ICF + 1-PSC ablation — proves the 5-attribute structured prompts (TBS)
# contribute beyond just "richer text representation".
#
# Compared to the ICF + 5-attr main run, this run swaps the per-class text
# input from 5 attribute embeddings to a single concatenated PSC prompt
# (the same 1-PSC text used by the M1 baseline row), then feeds it through
# the ICF fusion module as a [B, C, A=1, D] tensor.
#
# Architectural consequence (important — surfaces a clean ablation story):
#   With num_attrs=1, cross-attention has a single-token K/V sequence, so
#   softmax(Q · K^T / sqrt(d)) is identically [1.0] regardless of Q. The
#   image-conditional query has ZERO mathematical effect on the attention
#   output, and ICF degenerates to a fixed text-side projection
#       fused = output_proj( V_proj( attr_expert[0](text_emb) + pe[0] ) )
#   that is image-invariant by construction. image_proj parameters receive
#   no gradient.
#
# This is the cleanest possible demonstration that the 5-attribute structure
# is a necessary input for ICF — not a marginal improvement, but a
# precondition for the image-conditional mechanism to function.
#
# Expected outcomes (2x2 ablation matrix):
#   (a) M1 + 1-PSC + mean pool       : base 0.337 / novel 0.150 (Row 1)
#   (b) M1 + 5-attr + mean pool      : base 0.340 / novel 0.164 (Row 2)
#   (c) M1 + 1-PSC + ICF (THIS RUN)  : TBD — expect close to (a)
#   (d) M1 + 5-attr + ICF            : base 0.352 / novel 0.165 (Row 7)
# If (c) ≈ (a), the +1.2pp base gap from (b)→(d) is fully attributable to
# the synergy between ICF and 5-attr (TBS).

class_metadata_path = "data/texts/tct_ngc_class_metadata_base30_1psc.pt"

model = dict(
    backbone=dict(
        text_model=dict(
            class_metadata_path=class_metadata_path,
        ),
        cross_modal_fusion=dict(
            num_attrs=1,
        ),
    ),
)

# ICFCollapseGuard: when num_attrs=1, entropy over 1 attention key is
# identically 0 (log(1) = 0), so attn_entropy_max < 1.58 is trivially
# satisfied. fused_pairwise_cos will be 1.0 (image-invariant by
# construction) — relax thresholds so the hook does not warn-spam during
# training. Diagnostics still logged for inspection.
custom_hooks = [
    dict(
        type="ICFCollapseGuard",
        check_interval=500,
        check_at_val=True,
        halt_on_red=False,
        pairwise_cos_max=1.01,   # disable cos guard (1-attr is always 1.0)
        attn_entropy_max=0.01,   # disable entropy guard (1-attr is always 0)
        cos_to_mean_max=1.01,    # disable cos-to-mean guard
    ),
]

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_icf_1psc_biomedclip_2gpu"
