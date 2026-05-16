_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m2_biomedclip_2gpu.py"]

# Row 6c — "clean ord_loss" ablation.
#
# Tests whether ord_loss is fundamentally useful, with all the other HTA
# components stripped away. Pure question: does an ordinal-regression
# auxiliary signal (over clean medical-label axes only) help / hurt /
# neutral when applied on top of Stage 1 attribute processing only?
#
# Changes from M2 完整方法 (parent config):
#   1. HTA.skip_stage2=True         — disable per-organ MoE
#      (bypass experiment showed this is the main novel killer)
#   2. HTA.skip_stage3_rank_emb=True — disable rank-emb lookup table
#      (bypass experiment showed this contributes -2.3pp on novel zero-shot;
#       novel ranks fetch initialization noise from the table)
#   3. OrganOrdinalLoss.exclude_organ_axes = [
#         (0, 0),   # respiratory tract primary axis — PSC Category II
#                   # has 5 normal cell types all at rank 2; MSE collision
#                   # destroys class discriminability
#         (1, 0),   # Serous effusion primary axis — only 2 classes,
#                   # binary distinction (not ordinal)
#         (4, 2),   # TCT_CCD infection axis — different pathogens (monilia
#                   # / dysbacteriosis / vaginalis), not severity ordering
#      ]
#   4. OrganOrdinalLoss.skip_collision_ranks=True — within KEPT axes, drop
#      classes whose rank value collides with another class. Removes
#      Thyroid (Macrophages, NS) at rank 1 and (FC, Negative samples) at
#      rank 2, leaving rank-unique exemplars only.
#
# Remaining ord_loss supervision after exclusion + collision drop:
#   - Thyroid axis 0:    AUC(3), SPTC(5), PTC(6)  — 3 base classes
#   - Urine axis 0:      NHGUC(2), AUC(3), SHGUC(4), HGUC(5) — 4 base
#   - TCT_CCD axis 0:    ASCUS(1), ASCH(2), LSIL(3), HSIL(4) — 4 base
#   Total: 3 axes × ~3-4 classes = clean medical ordinality, no collisions
#
# Expected outcomes:
#   - novel ≥ 0.18 → ord_loss conceptually useful, write into paper as
#     part of main method (with proper label curation)
#   - novel 0.15-0.18 → ord_loss is roughly neutral (M1-5attr 平均 0.164 baseline)
#   - novel < 0.15 → ord_loss is net-negative even on clean axes; the
#     1-D MSE projection formulation is structurally wrong; switch to
#     pairwise ranking loss or drop ord_loss entirely
#
# Comparison baselines (corrected protocol, test set):
#   M1-5attr 平均:        base 0.340 / novel 0.164  (zero trainable text params)
#   M2 完整方法:           base 0.344 / novel 0.105
#   M2-auxfix v2:        base 0.343 / novel 0.056
#   Row 6c (this config): TBD / TBD

class_metadata_path = "data/texts/tct_ngc_class_metadata_base30.pt"
text_channels = 512

model = dict(
    backbone=dict(
        text_model=dict(
            adapter=dict(
                # Stage 2 / Stage 3 knockouts
                skip_stage2=True,
                skip_stage3_rank_emb=True,
                # The remaining lambda_* for Stage 1 stay at M2 baseline values
                # (inherited); rank_norm and gate_entropy auto-disabled by the
                # skip flags in get_aux_losses().
            ),
        ),
    ),
    ordinal_loss=dict(
        # Filter broken axes + collision-rank classes
        exclude_organ_axes=[(0, 0), (1, 0), (4, 2)],
        skip_collision_ranks=True,
        # Inherit normalization='mean' (parent default); loss_weight=0.3
        # (parent). Sum normalization here is overkill for 3 clean axes
        # × ~3-4 classes — keep mean to match M2 baseline magnitude.
    ),
)

# AdapterCollapseGuard from parent config still applies, but the rank_norm
# / gate_entropy thresholds will silently report N/A since those losses are
# disabled. Pool entropy / proj drift remain valid.

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m2_ordclean_biomedclip_2gpu"
