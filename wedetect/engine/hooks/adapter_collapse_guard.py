"""mmengine hook that monitors HierarchicalTextAdapter health.

Reads diagnostic state from the adapter after every N training iters
(and at every val end) and logs scalar collapse indicators. If any drops
below configured threshold, optionally halts training.

Thresholds default to values chosen for the OC-HMTA Module 2 design:
  - stage1 alpha_entropy_max: < log(5) ≈ 1.609  (uniform = bad)
  - stage1 proj_drift_min:    > 0.05  (close to identity = bad)
  - stage2 gate_entropy_max:  < log(5) ≈ 1.609  (uniform = bad)
  - stage2 organ_dominance_min: > 0.5  (gate should still favor class.organ)
  - stage3 rank_norm_min:     > 0.05

Note: stage1/stage2 entropy MAX upper bound is what we monitor (close to
uniform). We don't care about lower bound (over-concentration is fine).
"""
from typing import Optional

from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmdet.registry import HOOKS


@HOOKS.register_module()
class AdapterCollapseGuard(Hook):
    """Monitor HierarchicalTextAdapter for collapse and optionally halt.

    Args:
        check_interval: log every N training iters. Default 500.
        check_at_val: also log at the start of validation (where adapter
                      has freshest state from last train iter).
        halt_on_red: if True, raises RuntimeError when any threshold is
                     exceeded. If False, just logs WARNING.
        alpha_entropy_max:    Stage 1 pool entropy upper bound (~log(5)).
        proj_drift_min:       Stage 1 proj drift lower bound (anti-identity).
        gate_entropy_max:     Stage 2 gate entropy upper bound.
        organ_dominance_min:  Stage 2 gate organ-target hit-rate lower bound.
        rank_norm_min:        Stage 3 rank embedding norm lower bound.
    """

    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        check_interval: int = 500,
        check_at_val: bool = True,
        halt_on_red: bool = False,
        alpha_entropy_max: float = 1.5,
        proj_drift_min: float = 0.05,
        gate_entropy_max: float = 1.5,
        organ_dominance_min: float = 0.5,
        rank_norm_min: float = 0.05,
    ):
        self.check_interval = check_interval
        self.check_at_val = check_at_val
        self.halt_on_red = halt_on_red
        self.thresholds = {
            'stage1_alpha_entropy_max': ('stage1_alpha_entropy_max', alpha_entropy_max, 'max'),
            'stage1_proj_drift_min':    ('stage1_proj_drift_min',    proj_drift_min,    'min'),
            'stage2_gate_entropy_max':  ('stage2_gate_entropy_mean', gate_entropy_max,  'max'),
            'stage2_organ_dominance':   ('stage2_organ_dominance',   organ_dominance_min, 'min'),
            'stage3_rank_norm_min':     ('stage3_rank_norm_min',     rank_norm_min,     'min'),
        }

    def _get_adapter(self, runner):
        model = runner.model
        # Unwrap DDP if needed
        if hasattr(model, 'module'):
            model = model.module
        text_model = getattr(model.backbone, 'text_model', None)
        if text_model is None:
            return None
        return getattr(text_model, 'adapter', None)

    def _check(self, runner, tag: str):
        adapter = self._get_adapter(runner)
        if adapter is None or not hasattr(adapter, 'get_collapse_diagnostics'):
            return
        diag = adapter.get_collapse_diagnostics()
        if not diag:
            return

        logger = MMLogger.get_current_instance()
        msg_parts = [f'[AdapterCollapseGuard {tag}]']
        red = []

        for label, (diag_key, threshold, kind) in self.thresholds.items():
            val = diag.get(diag_key)
            if val is None:
                continue
            if kind == 'max':
                bad = val > threshold
                marker = 'X' if bad else 'OK'
                msg_parts.append(f'{label}={val:.3f}(<={threshold}) {marker}')
            elif kind == 'min':
                bad = val < threshold
                marker = 'X' if bad else 'OK'
                msg_parts.append(f'{label}={val:.3f}(>={threshold}) {marker}')
            else:
                continue
            if bad:
                red.append((label, val, threshold, kind))

        # also log raw per-attr drifts if available
        per_attr = diag.get('stage1_proj_drift_per_attr')
        if per_attr is not None:
            msg_parts.append(
                f"per_attr_drift={['%.2f' % d for d in per_attr]}"
            )

        logger.info(' | '.join(msg_parts))

        if red and self.halt_on_red:
            details = '\n  '.join(
                f'{name}: value={v:.4f}, threshold={t} ({k})'
                for name, v, t, k in red
            )
            raise RuntimeError(
                f'AdapterCollapseGuard halted training due to:\n  {details}'
            )

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (batch_idx + 1) % self.check_interval != 0:
            return
        self._check(runner, f'iter {runner.iter + 1}')

    def before_val_epoch(self, runner):
        if not self.check_at_val:
            return
        self._check(runner, f'pre-val ep {runner.epoch + 1}')
