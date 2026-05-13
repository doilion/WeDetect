"""Reproduce P1 #5 assigner-mask shape crash with direct assigner call."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    cfg = Config.fromfile(
        'config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py')
    cfg.work_dir = './work_dirs/debug_assigner_mask'
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    model.train()

    head = model.bbox_head
    print(f'organ_class_mask: shape={head.organ_class_mask.shape}  '
          f'device={head.organ_class_mask.device}')

    # iterate many batches looking for the failing case
    loader = iter(runner.train_dataloader)
    failed = []
    for itr in range(20):
        try:
            batch = next(loader)
        except StopIteration:
            break
        batch = model.data_preprocessor(batch, training=True)
        try:
            losses = model.loss(batch['inputs'], batch['data_samples'])
            organ_ids = [m['organ_id'] for m in batch['data_samples']['img_metas']]
            print(f'iter {itr:3d}: OK  loss_cls={losses["loss_cls"].item():.1f}  '
                  f'organs={sorted(set(organ_ids))}')
        except RuntimeError as e:
            organ_ids = [m['organ_id'] for m in batch['data_samples']['img_metas']]
            print(f'iter {itr:3d}: CRASH  organs={organ_ids}')
            print(f'  {e}')
            failed.append((itr, str(e)))
    print(f'\n{len(failed)} crashes / {itr+1} iterations')
    if failed:
        for f in failed:
            print(f'  iter {f[0]}: {f[1][:100]}')
    return

    # legacy single-batch path below (unreachable, kept for reference)
    loader = iter(runner.train_dataloader)
    batch = next(loader)
    batch = model.data_preprocessor(batch, training=True)

    # forward up to assigner manually
    img_feats, txt_feats = model.extract_feat(batch['inputs'], batch['data_samples'])
    cls_scores, bbox_preds, bbox_dist_preds = head(img_feats, txt_feats)
    batch_img_metas = batch['data_samples']['img_metas']
    batch_gt_instances = batch['data_samples']['bboxes_labels']
    num_imgs = len(batch_img_metas)

    print(f'cls_scores[0].shape={cls_scores[0].shape}')

    # Build flatten tensors like loss_by_feat does
    if head.featmap_sizes_train != [c.shape[2:] for c in cls_scores]:
        head.featmap_sizes_train = [c.shape[2:] for c in cls_scores]
        mlvl = head.prior_generator.grid_priors(
            head.featmap_sizes_train, dtype=cls_scores[0].dtype,
            device=cls_scores[0].device, with_stride=True)
        head.flatten_priors_train = torch.cat(mlvl, dim=0)
        head.stride_tensor = head.flatten_priors_train[..., [2]]

    flatten_cls_preds = torch.cat([
        c.permute(0, 2, 3, 1).reshape(num_imgs, -1, head.num_classes)
        for c in cls_scores
    ], dim=1)
    flatten_bbox_preds = torch.cat([
        b.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for b in bbox_preds
    ], dim=1)
    flatten_pred_bboxes = head.bbox_coder.decode(
        head.flatten_priors_train[..., :2], flatten_bbox_preds,
        head.stride_tensor[..., 0])

    print(f'flatten_cls_preds.shape={flatten_cls_preds.shape}')
    print(f'flatten_pred_bboxes.shape={flatten_pred_bboxes.shape}')
    print(f'flatten_priors_train.shape={head.flatten_priors_train.shape}')

    # Build organ mask
    from wedetect.models.dense_heads.yolo_world_head import YOLOWorldHead  # noqa
    organ_mask = head.organ_class_mask
    B, _, C = flatten_cls_preds.shape
    per_image_mask = flatten_cls_preds.new_zeros(B, C)
    for b, meta in enumerate(batch_img_metas):
        per_image_mask[b] = organ_mask[:, int(meta['organ_id'])]

    # gt prep
    from wedetect.models.dense_heads.utils import gt_instances_preprocess
    gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
    gt_labels = gt_info[:, :, :1]
    gt_bboxes = gt_info[:, :, 1:]
    pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

    for tag, assigner_input in [
        ('UNMASKED', flatten_cls_preds.detach().sigmoid()),
        ('MASKED',
         flatten_cls_preds.detach().sigmoid() * per_image_mask.unsqueeze(1)),
    ]:
        print(f'\n=== {tag} assigner input ===')
        print(f'  input.shape={assigner_input.shape}  '
              f'nonzero%={100 * (assigner_input != 0).float().mean().item():.1f}')
        result = head.assigner(
            flatten_pred_bboxes.detach().type(gt_bboxes.dtype),
            assigner_input, head.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)
        ab = result['assigned_bboxes']
        ascr = result['assigned_scores']
        fm = result['fg_mask_pre_prior']
        print(f'  assigned_bboxes.shape={ab.shape}  '
              f'assigned_scores.shape={ascr.shape}  '
              f'fg_mask.shape={fm.shape}  fg_count={fm.sum().item()}')

        # Repro line 651 downstream
        assigned_bboxes_norm = ab / head.stride_tensor
        prior_bbox_mask = fm.unsqueeze(-1).repeat([1, 1, 4])
        print(f'  prior_bbox_mask.shape={prior_bbox_mask.shape}  '
              f'sum={prior_bbox_mask.sum().item()}  '
              f'sum%4={prior_bbox_mask.sum().item() % 4}')

        assigned_ltrb = head.bbox_coder.encode(
            head.flatten_priors_train[..., :2] / head.stride_tensor,
            assigned_bboxes_norm,
            max_dis=head.head_module.reg_max - 1,
            eps=0.01)
        print(f'  assigned_ltrb.shape={assigned_ltrb.shape}')
        try:
            sel = torch.masked_select(assigned_ltrb, prior_bbox_mask)
            print(f'  masked_select size={sel.shape}  '
                  f'reshape to [-1, 4] OK')
        except RuntimeError as e:
            print(f'  CRASH on masked_select: {e}')


if __name__ == '__main__':
    main()
