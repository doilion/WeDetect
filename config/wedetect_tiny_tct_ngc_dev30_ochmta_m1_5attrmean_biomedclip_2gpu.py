_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py"]

# Row 3.5: M1 baseline + 5-attribute-mean text cache (vs Row 3's 1-PSC cache).
#
# Tests whether the 5-attribute structured text format brings more signal
# than single-prompt-per-class (1-PSC). Mean-pooled offline so no new code:
# this is a drop-in `text_embed_path` swap on top of the M1 (Row 3) config.
#
# Diff vs Row 3 (= base config):
#   - text_embed_path swapped from 1-PSC cache to 5-attr-mean cache
#   - class_text_path UNCHANGED (still tct_ngc_fullnames_30.json) because
#     the new cache is keyed by the SAME prompt strings — 5-attr-mean values
#     are stored under each 1-PSC prompt key.
#
# Everything else (Module 1 organ mask, image encoder, training schedule,
# data pipeline) is identical to Row 3 for clean isolation of the text
# contribution.

text_embed_path = "data/texts/tct_ngc_fullnames_30_attrmean_biomedclip.pth"

model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoLanguageBackbone",
            text_embed_path=text_embed_path,
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_5attrmean_biomedclip_2gpu"
