_base_ = ["./wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu.py"]

# Option 4 — Concat (5D) + Linear projection (5D→D) fusion on BiomedCLIP.
#
# Replaces THAF's cross-attention fusion (3.15M params, alpha→0 collapse)
# with a plain MLP projection:
#   class_vec = output_proj(LayerNorm(concat(attr_embs)))
#   output_proj: Linear(5D→D) -> GELU -> Dropout -> Linear(D→D)
#
# Rationale (counter to mean pool's "blurred composite"):
#   - Mean averaging discards per-attribute structure
#   - Concat preserves all 5 attributes' information
#   - Projection learns the optimal non-linear combination, with NO
#     residual / alpha gate (avoiding alpha→0 failure mode)
#   - Gain=1.0 init (vs THAF cross-attention's gain=0.1) so projection
#     has real magnitude from step 0
#
# Param count (D=512):
#   input_norm:          2 × 5D       = 5120
#   proj.Linear(5D→D):   5D*D + D     = 1,311,232
#   proj.Linear(D→D):    D*D + D      = 262,656
#   total                              ≈ 1.58M
# Smaller than THAF (3.15M), larger than per-class weights (155).
#
# Transfers to novel naturally: same projection applied regardless of
# class identity (unlike per-class weights which fall back to uniform).
#
# Other settings inherited from THAF biomedclip:
#   - 5-attr HierarchicalRandomLoadText pipeline
#   - tct_ngc_attr_biomedclip_per_attr.pth (512d)
#   - LR begin=1, T_max=12 (matches THAF, controls for schedule)

class_text_path = "data/texts/tct_ngc_fullnames_30_attr_train.json"
attr_cache_path = "data/texts/tct_ngc_attr_biomedclip_per_attr.pth"

text_channels = 512
num_attr_types = 5

model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoConcatProjBiomedCLIPLanguageBackbone",
            attr_emb_cache_path=attr_cache_path,
            num_attr_types=num_attr_types,
            embed_dim=text_channels,
            dropout=0.1,
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_concatproj_biomedclip_2gpu"
