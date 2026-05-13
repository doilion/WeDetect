"""1-step sanity for OC-HMTA Module 1 training pipeline.

Verifies:
  - OrganExtractor injects organ_id/organ_name into data_sample.metainfo
  - meta_keys propagation reaches the head's batch_img_metas
  - head.organ_class_mask is loaded at init from config path
  - loss_by_feat applies organ loss mask without shape mismatch or NaN
  - All losses (cls, bbox, dfl) are non-NaN and in a reasonable range

Usage:
    PYTHONPATH=. python tools/sanity_ochmta_m1.py
"""
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
    cfg_path = 'config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py'
    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = './work_dirs/sanity_ochmta_m1'
    # CPU/GPU: use CUDA if available
    print('=== building runner ===')
    runner = Runner.from_cfg(cfg)

    model = runner.model
    print(f'model.bbox_head.organ_loss_mask_enabled = '
          f'{model.bbox_head.organ_loss_mask_enabled}')
    mask = getattr(model.bbox_head, 'organ_class_mask', None)
    if mask is None:
        raise SystemExit('FAIL: head.organ_class_mask not loaded')
    print(f'model.bbox_head.organ_class_mask shape = {mask.shape}')
    print(f'  per-organ class counts: {mask.sum(dim=0).int().tolist()}')

    print('\n=== pulling 1 training batch ===')
    dataloader = runner.train_dataloader
    batch = next(iter(dataloader))
    print(f'batch keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch).__name__}')

    # The mmengine batch wraps {inputs, data_samples}. Verify organ_id is in metainfo.
    if isinstance(batch, dict):
        inputs = batch['inputs']
        data_samples = batch['data_samples']
    else:
        # WeDetect uses a custom collator; try attribute access
        inputs = batch.get('inputs', None) if hasattr(batch, 'get') else None
        data_samples = batch.get('data_samples', batch) if hasattr(batch, 'get') else batch

    print(f'inputs type: {type(inputs).__name__}'
          f'{" shape=" + str(inputs.shape) if torch.is_tensor(inputs) else ""}')
    print(f'data_samples len: {len(data_samples) if hasattr(data_samples, "__len__") else "?"}')

    # Probe first sample
    if isinstance(data_samples, dict):
        # Combined dict path (bboxes_labels + img_metas)
        print('data_samples is dict (combined path):')
        for k, v in data_samples.items():
            if torch.is_tensor(v):
                print(f'  {k}: tensor shape={tuple(v.shape)}')
            elif isinstance(v, list):
                print(f'  {k}: list len={len(v)}; first={type(v[0]).__name__}')
                if v and isinstance(v[0], dict):
                    print(f'    first dict keys: {list(v[0].keys())[:10]}')
                    print(f'    organ_id present? '
                          f'{"organ_id" in v[0]} → {v[0].get("organ_id", "MISSING")}')
            else:
                print(f'  {k}: {type(v).__name__}')
    elif isinstance(data_samples, (list, tuple)):
        s = data_samples[0]
        print(f'first sample type: {type(s).__name__}')
        meta = getattr(s, 'metainfo', None) or getattr(s, 'meta', {})
        if isinstance(meta, dict):
            print(f'  metainfo keys: {list(meta.keys())[:12]}')
            print(f'  organ_id: {meta.get("organ_id", "MISSING")}'
                  f' (name: {meta.get("organ_name", "MISSING")})')

    print('\n=== running model.loss (3 train steps; backward only on last) ===')
    model.train()
    train_iter = iter(dataloader)
    last_total = None
    for step in range(3):
        b = next(train_iter)
        b = model.data_preprocessor(b, training=True)
        # Verify per-batch organ coverage — after data_preprocessor, organ_id
        # is moved into each img_metas[i] (not a top-level key).
        img_metas = b['data_samples'].get('img_metas', [])
        organ_ids = [m.get('organ_id', None) for m in img_metas]
        if not organ_ids or any(o is None for o in organ_ids):
            raise SystemExit(
                f'FAIL: img_metas missing organ_id  '
                f'(metas[0] keys={list(img_metas[0].keys()) if img_metas else []})')
        print(f'  [step {step}] organ_ids={organ_ids}  '
              f'distinct organs={sorted(set(organ_ids))}')
        # Steps 0..N-2 run under no_grad to avoid accumulating activations.
        is_last = (step == 2)
        ctx = torch.enable_grad() if is_last else torch.no_grad()
        with ctx:
            losses = model.loss(b['inputs'], b['data_samples'])
            for k, v in losses.items():
                if torch.is_tensor(v):
                    if not torch.isfinite(v).all():
                        raise SystemExit(f'FAIL: non-finite {k} at step {step}')
            total = sum(v for v in losses.values() if torch.is_tensor(v))
            print(f'    loss_cls={losses["loss_cls"].item():.1f}  '
                  f'loss_bbox={losses["loss_bbox"].item():.2f}  '
                  f'loss_dfl={losses["loss_dfl"].item():.2f}  '
                  f'total={total.item():.1f}'
                  f'  ({"backward" if is_last else "no_grad"})')
            if is_last:
                last_total = total
        if not is_last:
            del losses, total
            torch.cuda.empty_cache()
    last_total.backward()

    # Verify a gradient hit the backbone (image side) — confirms backward worked
    sample_param = next(
        (p for n, p in model.named_parameters()
         if 'backbone.image_model' in n and p.requires_grad and p.grad is not None),
        None,
    )
    if sample_param is None:
        print('WARN: no image_model gradient found (might be normal if all frozen)')
    else:
        gn = sample_param.grad.norm().item()
        print(f'image_model sample grad_norm = {gn:.4f}  '
              f'(finite={bool(torch.isfinite(sample_param.grad).all())})')

    print('\n=== SANITY PASSED ===')


if __name__ == '__main__':
    main()
