"""Inter-class cosine similarity analysis for OC-HMTA text representations.

For each row (Row 3 1-PSC / Row 3.5 5-attr-mean / Row 4d M2 adapted),
computes the 30x30 pairwise cos matrix on base classes and renders a
science-style heatmap (RdBu_r: red=high, blue=low).

Also reports within-organ vs cross-organ mean similarity to quantify
whether organ-prior + adapter actually pushes cross-organ pairs apart
(lower cos) while keeping within-organ structure.

Output:
  data/texts/class_cos_heatmaps.png   (3-panel side-by-side)
  data/texts/class_cos_stats.json     (within/cross summary)

Usage:
  python tools/analyze_class_cos.py \\
      --row4d-ckpt work_dirs/.../best_coco_bbox_mAP_epoch_10.pth
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from mmdet.utils import register_all_modules

register_all_modules()

import wedetect  # noqa
from mmdet.registry import MODELS


def cos_matrix(emb: torch.Tensor) -> np.ndarray:
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    return (emb @ emb.T).cpu().numpy()


def within_vs_cross(m: np.ndarray, organ_ids: np.ndarray):
    """Mean off-diagonal cos within same organ vs cross organ."""
    n = len(organ_ids)
    within = []
    cross = []
    for i in range(n):
        for j in range(i + 1, n):
            if organ_ids[i] == organ_ids[j]:
                within.append(m[i, j])
            else:
                cross.append(m[i, j])
    return {
        'within_organ_mean': float(np.mean(within)),
        'within_organ_max': float(np.max(within)),
        'within_organ_min': float(np.min(within)),
        'cross_organ_mean': float(np.mean(cross)),
        'cross_organ_max': float(np.max(cross)),
        'cross_organ_min': float(np.min(cross)),
        'separation_gap': float(np.mean(within) - np.mean(cross)),
        'n_within_pairs': len(within),
        'n_cross_pairs': len(cross),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--row4d-ckpt', default=None,
                   help='Row 4d best ckpt (M2 adapter trained). Optional.')
    p.add_argument('--out-png', default='data/texts/class_cos_heatmaps.png')
    p.add_argument('--out-json', default='data/texts/class_cos_stats.json')
    p.add_argument('--m2-config',
                   default='config/wedetect_tiny_tct_ngc_dev30_ochmta_m2_biomedclip_2gpu.py')
    args = p.parse_args()

    # --- Load class metadata (organ ordering + axes + 5-attr emb) ---
    meta = torch.load('data/texts/tct_ngc_class_metadata_base30.pt',
                      weights_only=False, map_location='cpu')
    class_names = meta['class_names']
    organ_ids = meta['organ_ids'].numpy()           # [30]
    axis_ids = meta['axis_ids']
    rank_along_axis = meta['rank_along_axis']
    attr_emb = meta['attr_emb']                      # [30, 5, 512]

    organ_names = ['respiratory tract', 'Serous effusion', 'Thyroid gland', 'Urine', 'TCT_CCD']
    # Organ block boundaries for heatmap
    boundaries = []
    cur = -1
    for i, o in enumerate(organ_ids):
        if o != cur:
            boundaries.append(i)
            cur = o
    boundaries.append(len(organ_ids))  # tail

    # --- Row 2/3 (1-PSC) ---
    psc_cache = torch.load(
        'data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth',
        weights_only=False, map_location='cpu')
    psc_prompts = json.loads(
        Path('data/texts/tct_ngc_fullnames_30.json').read_text())
    psc_emb = torch.stack([psc_cache[p[0]] for p in psc_prompts])   # [30, 512]
    print(f'Row 3 (1-PSC) embeddings: {psc_emb.shape}')

    # --- Row 3.5 (5-attr-mean) ---
    mean_cache = torch.load(
        'data/texts/tct_ngc_fullnames_30_attrmean_biomedclip.pth',
        weights_only=False, map_location='cpu')
    mean_emb = torch.stack([mean_cache[p[0]] for p in psc_prompts])
    print(f'Row 3.5 (5-attr-mean) embeddings: {mean_emb.shape}')

    representations = {
        'Row 3 (1-PSC)': psc_emb,
        'Row 3.5 (5-attr-mean)': mean_emb,
    }

    # --- Row 4d (M2 adapted) — load adapter from ckpt + forward ---
    if args.row4d_ckpt and Path(args.row4d_ckpt).is_file():
        from mmengine.config import Config
        cfg = Config.fromfile(args.m2_config)
        adapter_cfg = cfg.model.backbone.text_model.adapter
        adapter = MODELS.build(adapter_cfg)
        adapter.eval()

        # Load adapter state dict from full model ckpt
        ckpt = torch.load(args.row4d_ckpt, weights_only=False, map_location='cpu')
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        prefix = 'backbone.text_model.adapter.'
        adapter_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing = set(adapter.state_dict().keys()) - set(adapter_sd.keys())
        if missing:
            print(f'WARN: adapter ckpt missing {len(missing)} keys, e.g. {list(missing)[:3]}')
        adapter.load_state_dict(adapter_sd, strict=False)

        # Forward through adapter
        with torch.no_grad():
            attr_emb_b1 = attr_emb.unsqueeze(0)              # [1, 30, 5, 512]
            row4d_emb = adapter(
                attr_emb_b1,
                meta['organ_ids'].long(),
                meta['axis_ids'].long(),
                meta['rank_along_axis'].long(),
            ).squeeze(0)                                       # [30, 512]
        print(f'Row 4d (M2 adapted) embeddings: {row4d_emb.shape}')
        representations['Row 4d (M2 adapted)'] = row4d_emb
    else:
        print(f'Row 4d ckpt not found ({args.row4d_ckpt}), skipping M2.')

    # --- Compute matrices + stats ---
    matrices = {}
    stats = {}
    for name, emb in representations.items():
        m = cos_matrix(emb)
        matrices[name] = m
        stats[name] = within_vs_cross(m, organ_ids)
        print(f'\n{name}:')
        print(f'  within-organ mean cos = {stats[name]["within_organ_mean"]:.3f}  '
              f'(max={stats[name]["within_organ_max"]:.3f}, min={stats[name]["within_organ_min"]:.3f})')
        print(f'  cross-organ mean cos = {stats[name]["cross_organ_mean"]:.3f}  '
              f'(max={stats[name]["cross_organ_max"]:.3f}, min={stats[name]["cross_organ_min"]:.3f})')
        print(f'  separation gap = within - cross = {stats[name]["separation_gap"]:+.3f}')

    # --- Save JSON ---
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(stats, indent=2))
    print(f'\nstats saved: {args.out_json}')

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plot')
        return

    n_panels = len(matrices)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6.5),
                             squeeze=False)
    axes = axes[0]

    vmin, vmax = 0.0, 1.0
    for ax, (name, m) in zip(axes, matrices.items()):
        im = ax.imshow(m, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       aspect='equal', interpolation='nearest')
        # Draw organ block boundary lines
        for b in boundaries[1:-1]:
            ax.axhline(y=b - 0.5, color='black', linewidth=0.8, alpha=0.7)
            ax.axvline(x=b - 0.5, color='black', linewidth=0.8, alpha=0.7)
        ax.set_title(f'{name}\nwithin={stats[name]["within_organ_mean"]:.3f}  '
                     f'cross={stats[name]["cross_organ_mean"]:.3f}  '
                     f'gap={stats[name]["separation_gap"]:+.3f}',
                     fontsize=10)
        # Tick labels for organs (block centers)
        block_centers = [(boundaries[i] + boundaries[i + 1]) / 2 - 0.5
                         for i in range(len(boundaries) - 1)]
        ax.set_xticks(block_centers)
        ax.set_yticks(block_centers)
        ax.set_xticklabels([o.replace(' ', '\n') for o in organ_names], fontsize=8)
        ax.set_yticklabels([o.replace(' ', '\n') for o in organ_names], fontsize=8)
        ax.set_xlabel('class j', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('class i', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8, label='cos similarity')

    fig.suptitle('Inter-class text embedding cosine similarity (30 base classes)\n'
                 'red = high similarity (collapsed),  blue = low (well-separated)',
                 fontsize=11, y=0.99)
    plt.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=150, bbox_inches='tight')
    print(f'plot saved: {args.out_png}')


if __name__ == '__main__':
    main()
