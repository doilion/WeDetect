# THAF + BiomedCLIP ckpt 的 visproto-only eval-time config.
#
# 用途：用 THAF 训出来的 ckpt（image_encoder + neck + bbox_head 都是 THAF 训过的），
#       在 inference 时把 text_model 从 PseudoHierarchicalBiomedCLIPLanguageBackbone
#       换成 PseudoLanguageBackbone（直接吃 cached visproto emb，没 fusion）。
#
# 为什么 OK：
#   - visproto-only 路径完全绕过 text encoder（class_vec 来自 image-side prototype，
#     不需要 text fusion 输出）。THAF 的 fusion 参数加载到 PseudoLanguageBackbone 时
#     被 strict=False 忽略 —— 这正是预期的行为。
#   - 我们要测的是：THAF 训过的 head（包括 BNContrastiveHead 的 BN running stats +
#     logit_scale + bias）对 visproto class vector 的反应。
#
# 跟 eval_novel_split.py guard 的关系：guard 检查 cfg.model.backbone.text_model.type
# 是否含 "Hierarchical"，本 config 把它换成 PseudoLanguageBackbone → guard 不触发 ✅
#
# 使用：
#   python tools/eval_novel_split.py \
#       --config config/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_eval_visproto.py \
#       --checkpoint work_dirs/.../thaf_biomedclip_2gpu/best_*.pth \
#       --data-root /home1/liwenjie/TCT_NGC/ \
#       --ann-file annotations/instances_test_<split>_novel.json \
#       --text-json data/texts/tct_ngc_novel_<split>.json \
#       --text-emb data/texts/tct_ngc_novel_<split>_visproto_emb_thaf_biomedclip.pth \
#       --work-dir work_dirs/.../eval_thaf_visproto_<split>

_base_ = ["./wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu.py"]

# Placeholder; eval_novel_split.py 会用 --text-emb 覆盖
text_embed_path = "data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth"

model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoLanguageBackbone",
            text_embed_path=text_embed_path,
        ),
    ),
)
