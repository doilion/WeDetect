"""Unit tests for OrganOrdinalLoss.

Verifies the two normalization modes and protects the contract that any
``OrganOrdinalLoss(normalization=...)`` swap must be matched by config-side
ckpt accountability — silent mismatch produces a different model, see the
docstring of OrganOrdinalLoss for context.

Run: ``PYTHONPATH=. python tests/test_organ_ordinal_loss.py``
(No pytest dependency; small enough for a __main__ runner.)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wedetect.models.losses.organ_ordinal_loss import OrganOrdinalLoss


def _fixture(seed: int = 0):
    """Build a deterministic 2-organ × 2-axis fixture with 6 active classes.

    Organ 0 axis 0: ranks 0, 1, 2  -> 3 classes (active, n=3)
    Organ 0 axis 1:                 -> 0 classes (degenerate, skipped)
    Organ 1 axis 0: ranks 0, 1     -> 2 classes (active, n=2)
    Organ 1 axis 1: rank -1        -> 1 class (degenerate, skipped)
    => 2 active (organ, axis) pairs.
    """
    torch.manual_seed(seed)
    embed_dim = 8
    num_classes = 6
    emb_final = torch.randn(2, num_classes, embed_dim)            # [B=2, C, D]
    class_organ_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    class_axis_ids = torch.tensor([0, 0, 0, 0, 0, 1])
    class_ranks = torch.tensor([0, 1, 2, 0, 1, -1])
    return emb_final, class_organ_ids, class_axis_ids, class_ranks


def test_normalization_validates():
    try:
        OrganOrdinalLoss(normalization='avg')   # bogus
    except ValueError as e:
        assert 'mean' in str(e) and 'sum' in str(e), e
    else:
        raise AssertionError('expected ValueError for normalization=avg')


def test_default_is_mean():
    loss = OrganOrdinalLoss(embed_dim=8, num_organs=2, max_axes_per_organ=2)
    assert loss.normalization == 'mean'


def test_sum_is_n_active_times_mean():
    """SUM = MEAN * active_axes when active_axes > 0."""
    emb, organ, axis, rank = _fixture()
    common = dict(embed_dim=emb.shape[-1], num_organs=2,
                  max_axes_per_organ=2, loss_weight=1.0,
                  monotonicity_weight=0.5)

    torch.manual_seed(42)
    mean_loss = OrganOrdinalLoss(**common, normalization='mean')
    torch.manual_seed(42)   # same init for head_w / head_b
    sum_loss = OrganOrdinalLoss(**common, normalization='sum')

    out_mean = mean_loss(emb, organ, axis, rank)['loss_ord'].item()
    out_sum = sum_loss(emb, organ, axis, rank)['loss_ord'].item()

    # Fixture has 2 active (organ, axis) pairs (organ 0 axis 0, organ 1 axis 0).
    expected_ratio = 2.0
    assert math.isclose(out_sum / max(out_mean, 1e-9),
                        expected_ratio, rel_tol=1e-4), (
        f'expected sum/mean ≈ {expected_ratio}, '
        f'got mean={out_mean:.6f} sum={out_sum:.6f} '
        f'ratio={out_sum / max(out_mean, 1e-9):.4f}')


def test_loss_weight_scales_output():
    emb, organ, axis, rank = _fixture()
    common = dict(embed_dim=emb.shape[-1], num_organs=2,
                  max_axes_per_organ=2, monotonicity_weight=0.5,
                  normalization='sum')
    torch.manual_seed(42)
    base = OrganOrdinalLoss(**common, loss_weight=1.0)
    torch.manual_seed(42)
    scaled = OrganOrdinalLoss(**common, loss_weight=0.3)
    a = base(emb, organ, axis, rank)['loss_ord'].item()
    b = scaled(emb, organ, axis, rank)['loss_ord'].item()
    assert math.isclose(b / max(a, 1e-9), 0.3, rel_tol=1e-4), (
        f'loss_weight scale broken: 0.3*base={0.3*a:.6f} vs scaled={b:.6f}')


def test_all_degenerate_axes_returns_zero():
    """If no (organ, axis) has >=2 valid-rank classes, loss is 0."""
    embed_dim = 8
    emb = torch.randn(1, 3, embed_dim)
    organ = torch.tensor([0, 1, 2])      # all different organs
    axis = torch.tensor([0, 0, 0])
    rank = torch.tensor([0, 0, 0])       # 1 class per (organ, axis)
    for norm in ('mean', 'sum'):
        loss = OrganOrdinalLoss(embed_dim=embed_dim, num_organs=3,
                                max_axes_per_organ=1, loss_weight=1.0,
                                normalization=norm)
        out = loss(emb, organ, axis, rank)['loss_ord'].item()
        assert out == 0.0, f'{norm}: expected 0 for all-degenerate, got {out}'


def test_gradients_flow():
    """Gradient should flow back to emb_final and the linear head params."""
    emb, organ, axis, rank = _fixture()
    emb = emb.requires_grad_(True)
    loss = OrganOrdinalLoss(embed_dim=emb.shape[-1], num_organs=2,
                            max_axes_per_organ=2, loss_weight=1.0,
                            normalization='sum')
    out = loss(emb, organ, axis, rank)['loss_ord']
    out.backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0
    assert loss.head_w.grad is not None and loss.head_w.grad.abs().sum() > 0


def test_exclude_organ_axes_skips_listed_pairs():
    """exclude_organ_axes=[(0,0)] should make ord_loss skip that pair entirely."""
    emb, organ, axis, rank = _fixture()
    common = dict(embed_dim=emb.shape[-1], num_organs=2,
                  max_axes_per_organ=2, loss_weight=1.0,
                  monotonicity_weight=0.5, normalization='sum')
    torch.manual_seed(42)
    full = OrganOrdinalLoss(**common)
    torch.manual_seed(42)
    excl = OrganOrdinalLoss(**common, exclude_organ_axes=[(0, 0)])
    a = full(emb, organ, axis, rank)['loss_ord'].item()
    b = excl(emb, organ, axis, rank)['loss_ord'].item()
    # fixture has 2 active pairs ((0,0) with 3 classes, (1,0) with 2 classes)
    # excluding (0,0) leaves only (1,0). So b should be ~half of a.
    assert b < a, f'exclude should reduce loss; got full={a:.4f} excl={b:.4f}'
    assert b > 0, f'(1,0) should still contribute non-zero loss; got {b:.4f}'


def test_min_unique_ranks_skips_binary_axis():
    """An axis with only 2 unique ranks should be skipped when min_unique_ranks=3."""
    embed_dim = 8
    emb = torch.randn(1, 4, embed_dim)
    # (0,0): 4 classes with only 2 unique ranks (binary axis like Serous)
    organ = torch.tensor([0, 0, 0, 0])
    axis = torch.tensor([0, 0, 0, 0])
    rank = torch.tensor([0, 0, 1, 1])
    loss = OrganOrdinalLoss(embed_dim=embed_dim, num_organs=1,
                            max_axes_per_organ=1, loss_weight=1.0,
                            normalization='sum',
                            min_unique_ranks=3)  # require >=3 unique ranks
    out = loss(emb, organ, axis, rank)['loss_ord'].item()
    assert out == 0.0, (
        f'binary axis (2 unique ranks) should be skipped at min_unique_ranks=3, '
        f'got {out}'
    )


def test_skip_collision_ranks_drops_colliding_classes():
    """skip_collision_ranks should drop classes whose rank value collides.

    Setup: (organ 0, axis 0) with classes at ranks 1, 1, 3, 5.
    - skip_collision_ranks=False: all 4 classes participate (rank 1 collision OK)
    - skip_collision_ranks=True: rank 1 classes dropped, only rank 3 + rank 5
      participate (and min_unique_ranks=2 just barely satisfied).
    """
    embed_dim = 8
    emb = torch.randn(1, 4, embed_dim)
    organ = torch.tensor([0, 0, 0, 0])
    axis = torch.tensor([0, 0, 0, 0])
    rank = torch.tensor([1, 1, 3, 5])
    common = dict(embed_dim=embed_dim, num_organs=1, max_axes_per_organ=1,
                  loss_weight=1.0, monotonicity_weight=0.5, normalization='sum')
    torch.manual_seed(42)
    a = OrganOrdinalLoss(**common, skip_collision_ranks=False)(
        emb, organ, axis, rank)['loss_ord'].item()
    torch.manual_seed(42)
    b = OrganOrdinalLoss(**common, skip_collision_ranks=True)(
        emb, organ, axis, rank)['loss_ord'].item()
    # The losses should differ — collision-dropping removes 2 samples from MSE
    assert a != b, (
        f'skip_collision_ranks should change the loss value; '
        f'got with={a:.4f} without={b:.4f}'
    )


def test_skip_collision_ranks_skips_axis_when_no_unique_left():
    """If skipping collisions leaves <2 unique ranks, the whole axis is skipped."""
    embed_dim = 8
    emb = torch.randn(1, 4, embed_dim)
    organ = torch.tensor([0, 0, 0, 0])
    axis = torch.tensor([0, 0, 0, 0])
    # All ranks collide (5 normal cells at rank 2 scenario):
    rank = torch.tensor([2, 2, 2, 2])
    loss = OrganOrdinalLoss(embed_dim=embed_dim, num_organs=1,
                            max_axes_per_organ=1, loss_weight=1.0,
                            normalization='sum', skip_collision_ranks=True)
    out = loss(emb, organ, axis, rank)['loss_ord'].item()
    assert out == 0.0, (
        f'all-collision axis should yield 0 loss when skip_collision_ranks=True, '
        f'got {out}'
    )


def main():
    tests = [
        test_normalization_validates,
        test_default_is_mean,
        test_sum_is_n_active_times_mean,
        test_loss_weight_scales_output,
        test_all_degenerate_axes_returns_zero,
        test_gradients_flow,
        test_exclude_organ_axes_skips_listed_pairs,
        test_min_unique_ranks_skips_binary_axis,
        test_skip_collision_ranks_drops_colliding_classes,
        test_skip_collision_ranks_skips_axis_when_no_unique_left,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e}')
        except Exception as e:                                          # noqa: BLE001
            failed += 1
            print(f'ERROR {t.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
