"""Unit test: verify cross-organ cls scores have ZERO effect on training.

Procedure:
  1. Load M1 config + build model on GPU 0.
  2. Pull one training batch.
  3. Forward to get (cls_scores, bbox_preds, bbox_dist_preds) — the canonical
     intermediate tensors that loss_by_feat consumes.
  4. Call loss_by_feat once with the canonical scores -> losses_A.
  5. Build `modified_cls_scores`: identical to canonical EXCEPT cross-organ
     class channels are overwritten with arbitrary noise (we use a separate
     RNG so values are radically different from what the network produced).
  6. Call loss_by_feat with modified -> losses_B.
  7. Compare: if cross-organ scores truly have no effect, losses_A and
     losses_B must be bit-identical for loss_cls, loss_bbox, loss_dfl.

If they DIFFER, the cross-organ contamination claim is correct and we need
to mask the assigner input. If IDENTICAL, current loss-mask-only design is
mathematically equivalent to a fully organ-isolated forward.

This test bypasses model.forward by intercepting head outputs and calling
loss_by_feat twice on the same captured intermediates with only the
cross-organ slice modified.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import copy
import torch
from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    cfg = Config.fromfile(
        'config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py')
    cfg.work_dir = './work_dirs/test_cross_organ_isolation'
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    model.train()

    head = model.bbox_head
    organ_mask = head.organ_class_mask  # [C, O]
    C = organ_mask.shape[0]
    print(f'organ_class_mask: {organ_mask.shape}  per-organ class counts: '
          f'{organ_mask.sum(dim=0).int().tolist()}')

    # --- Pull one batch ------------------------------------------------
    batch = next(iter(runner.train_dataloader))
    batch = model.data_preprocessor(batch, training=True)
    img_metas = batch['data_samples']['img_metas']
    organ_ids = [m['organ_id'] for m in img_metas]
    print(f'batch organ_ids: {organ_ids}')

    # --- Forward once to get intermediates ----------------------------
    with torch.no_grad():
        img_feats, txt_feats = model.extract_feat(
            batch['inputs'], batch['data_samples'])
        outs = head(img_feats, txt_feats)        # tuple (cls_scores, bbox_preds, bbox_dist_preds)
    cls_scores, bbox_preds, bbox_dist_preds = outs
    print(f'cls_scores[0] shape: {cls_scores[0].shape}  '
          f'(num scales={len(cls_scores)})')

    # --- Build per-sample cross-organ mask in class-channel space -----
    # For each scale, cls_scores[s] has shape [B, C, H, W]. We want to
    # zero out (or scramble) the C-axis values at positions where the
    # class is cross-organ for that image.
    B = cls_scores[0].shape[0]
    # per_image_keep[b, c] = 1 if class c is in image b's organ, else 0
    per_image_keep = torch.stack([organ_mask[:, o] for o in organ_ids], dim=0).to(
        cls_scores[0].device, dtype=cls_scores[0].dtype)        # [B, C]
    per_image_drop = (1.0 - per_image_keep)                     # [B, C], 1 on cross-organ

    # Use a fresh RNG so the noise is decorrelated from anything in the
    # network — if the loss is sensitive to cross-organ values, large
    # random values will produce a visibly different loss.
    rng = torch.Generator(device=cls_scores[0].device).manual_seed(99991)

    def with_modified_cross_organ(cls_scores_in, fill='noise'):
        """Return a new tuple where cross-organ channels are replaced.

        fill='noise': uniform[-5, 5] random values.
        fill='zero':  zero out (sanity check; should produce identical losses).
        fill='huge':  +10 (saturated logit).
        """
        out = []
        for s, cs in enumerate(cls_scores_in):
            new = cs.clone()
            B_, C_, H, W = new.shape
            drop_mask = per_image_drop.view(B_, C_, 1, 1)       # broadcast over H, W
            if fill == 'noise':
                rand = torch.empty_like(new).uniform_(-5.0, 5.0, generator=rng)
                new = new * (1 - drop_mask) + rand * drop_mask
            elif fill == 'zero':
                new = new * (1 - drop_mask)
            elif fill == 'huge':
                new = new * (1 - drop_mask) + 10.0 * drop_mask
            else:
                raise ValueError(fill)
            out.append(new)
        return tuple(out)

    # --- Run loss_by_feat with both versions --------------------------
    bboxes_labels = batch['data_samples']['bboxes_labels']
    img_metas_list = batch['data_samples']['img_metas']

    def compute_losses(cs):
        with torch.no_grad():
            return head.loss_by_feat(
                cs, bbox_preds, bbox_dist_preds, bboxes_labels, img_metas_list)

    losses_A = compute_losses(cls_scores)
    print(f'\n[A] canonical cls_scores:')
    for k, v in losses_A.items():
        print(f'  {k} = {v.item():.10f}')

    for label, fill in [('zero', 'zero'),
                        ('huge (+10)', 'huge'),
                        ('noise [-5,5]', 'noise')]:
        modified = with_modified_cross_organ(cls_scores, fill)
        losses_B = compute_losses(modified)
        print(f'\n[B] cross-organ replaced with {label}:')
        for k, v in losses_B.items():
            print(f'  {k} = {v.item():.10f}')
        print(f'\nΔ canonical - {label}:')
        all_same = True
        for k in losses_A:
            d = (losses_A[k] - losses_B[k]).abs().item()
            verdict = 'EXACT' if d == 0 else ('CLOSE' if d < 1e-6 else 'DIFFER')
            if verdict != 'EXACT':
                all_same = False
            print(f'  {k}: |Δ|={d:.6g}  ({verdict})')
        if all_same:
            print(f'  CONCLUSION: cross-organ values have ZERO effect on losses.')
        else:
            print(f'  CONCLUSION: cross-organ values LEAK into losses — P1 #5 is REAL.')


if __name__ == '__main__':
    main()
