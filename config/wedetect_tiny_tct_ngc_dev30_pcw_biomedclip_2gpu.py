_base_ = ["./wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu.py"]

# Option 3a — Per-class learnable attribute weighting on BiomedCLIP.
#
# Replaces THAF's cross-attention fusion (3.15M params, alpha→0 collapse)
# with a tiny per-class weight matrix: nn.Parameter(num_classes+1, 5)
# softmax → weighted sum of 5 per-attr embeddings.
#
# Hypothesis (user-flagged 2026-05-11):
#   "Mean of 5 attribute embeddings is a LOWER BOUND, not the ceiling.
#    Like describing Mt. Lushan from 4 directions — averaging gives a
#    blurred composite. Per-class selective weighting should beat mean."
#
# Param count: (30+1) × 5 = 155 scalars total (vs 3.15M for THAF).
# Class identification: diagnostic_code (attr idx 1) — unique per class.
# Novel zero-shot fallback: unknown classes default to softmax(0) = 1/5
# uniform weights (= mean pool, neutral baseline).
#
# Other settings inherited from THAF biomedclip:
#   - 5-attr HierarchicalRandomLoadText pipeline
#   - tct_ngc_attr_biomedclip_per_attr.pth (512d)
#   - LR begin=1, T_max=12 (matches THAF, controls for schedule)
#   - bbox_head.head_module.embed_dims=512

class_text_path = "data/texts/tct_ngc_fullnames_30_attr_train.json"
attr_cache_path = "data/texts/tct_ngc_attr_biomedclip_per_attr.pth"
class_keys_path = "data/texts/tct_ngc_class_keys_30.json"

text_channels = 512
num_attr_types = 5

model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoPerClassWeightedBiomedCLIPLanguageBackbone",
            attr_emb_cache_path=attr_cache_path,
            class_keys_json=class_keys_path,
            num_attr_types=num_attr_types,
            embed_dim=text_channels,
            class_key_attr_idx=1,
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_pcw_biomedclip_2gpu"
