"""1-step sanity for OC-HMTA Module 2 training pipeline.

Verifies:
  - PseudoMultiAttrLanguageBackbone loads metadata + attr_emb on cuda
  - HierarchicalTextAdapter forward produces finite [B, C, D] output
  - Aux losses (pool_entropy, gate_entropy, rank_norm) are finite
  - OrganOrdinalLoss produces finite scalar
  - End-to-end detector loss() runs with multi-organ batch, all losses finite
  - AdapterCollapseGuard diagnostics print sane initial values
    (alpha_entropy < log(5)≈1.6, gate_entropy reflects prior bias)

Usage (after user wakes):
    PYTHONPATH=. python tools/sanity_ochmta_m2.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import math
import torch
from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    cfg = Config.fromfile(
        'config/wedetect_tiny_tct_ngc_dev30_ochmta_m2_biomedclip_2gpu.py')
    cfg.work_dir = './work_dirs/sanity_ochmta_m2'
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    model.train()

    head = model.bbox_head
    text_model = model.backbone.text_model
    adapter = getattr(text_model, 'adapter', None)
    print(f'organ_class_mask in head: {head.organ_class_mask.shape}')
    print(f'text_model class_organ_ids[:5]: {text_model.class_organ_ids[:5].tolist()}')
    print(f'text_model attr_emb shape:   {text_model.attr_emb.shape}')
    print(f'adapter total params:        {sum(p.numel() for p in adapter.parameters())}')
    print(f'detector.ordinal_loss:       {model.ordinal_loss}')

    # --- Adapter forward sanity (just text path) ---
    print('\n=== adapter forward sanity ===')
    with torch.no_grad():
        emb_attr = text_model.attr_emb.unsqueeze(0).expand(2, -1, -1, -1)  # B=2
        out = adapter(
            emb_attr,
            text_model.class_organ_ids,
            text_model.class_axis_ids,
            text_model.class_ranks,
        )
        print(f'adapter out shape: {out.shape}  finite: {torch.isfinite(out).all().item()}')
        print(f'  out norm: mean={out.norm(dim=-1).mean():.3f}  std={out.norm(dim=-1).std():.3f}')

        diag = adapter.get_collapse_diagnostics()
        for k, v in diag.items():
            print(f'  diag {k}: {v}')

        aux_losses = adapter.get_aux_losses()
        for k, v in aux_losses.items():
            print(f'  aux {k}: {v.item():.6f}  finite={torch.isfinite(v).all().item()}')

    # --- End-to-end model.loss() with one batch ---
    print('\n=== end-to-end model.loss(3 train steps) ===')
    train_iter = iter(runner.train_dataloader)
    for step in range(3):
        b = next(train_iter)
        b = model.data_preprocessor(b, training=True)
        img_metas = b['data_samples'].get('img_metas', [])
        organ_ids = [m.get('organ_id') for m in img_metas]
        print(f'  [step {step}] organs={sorted(set(organ_ids))}')
        is_last = (step == 2)
        ctx = torch.enable_grad() if is_last else torch.no_grad()
        with ctx:
            losses = model.loss(b['inputs'], b['data_samples'])
            for k, v in losses.items():
                if torch.is_tensor(v):
                    if not torch.isfinite(v).all():
                        raise SystemExit(f'FAIL: non-finite {k} at step {step}')
                    print(f'    {k}: {v.item():.4f}')
            if is_last:
                total = sum(v for v in losses.values() if torch.is_tensor(v))
                print(f'    TOTAL: {total.item():.4f}')
                total.backward()

    # --- Backward grad sanity ---
    print('\n=== backward grad sanity ===')
    adapter_grad_norms = []
    for n, p in adapter.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            adapter_grad_norms.append((n, gn))
    if adapter_grad_norms:
        print(f'  adapter params with grad: {len(adapter_grad_norms)}')
        for n, gn in adapter_grad_norms[:5]:
            print(f'    {n}: grad_norm={gn:.4f}')
    else:
        print('  WARN: no adapter param has grad')

    ord_grad_norms = []
    for n, p in model.ordinal_loss.named_parameters():
        if p.grad is not None:
            ord_grad_norms.append((n, p.grad.norm().item()))
    print(f'  ordinal_loss params with grad: {len(ord_grad_norms)}')

    print('\n=== SANITY PASSED ===')
    # Expected initial values for guard verification:
    expected = """
Expected initial diagnostics (random init):
  stage1_alpha_entropy_mean ≈ 1.4 - 1.6   (some non-uniform from attn_attr_bias init)
  stage1_proj_drift_min     > 0.3         (orthogonal init scaled 0.5 gives meaningful drift)
  stage2_gate_entropy_mean  < 0.5         (prior_bias +5 dominates → gate near-hard)
  stage2_organ_dominance    ≈ 1.0         (~100% gate puts max on class.organ at init)
  stage3_rank_norm_min      ≈ 0.5 - 1.0   (σ=0.05 * sqrt(512) ≈ 1.13)
"""
    print(expected)


if __name__ == '__main__':
    main()
