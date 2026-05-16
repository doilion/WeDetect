"""OC-HMTA Module 2 ordinal aux loss.

Per (organ, axis) ordinal regression on adapter outputs. For each non-
degenerate (organ, axis) with >= 2 classes:
  - Linear head projects class vector to a scalar predicted rank
  - MSE loss against true rank_along_axis
  - Monotonicity penalty: for class pairs c1, c2 with rank[c1] < rank[c2],
    require pred_rank[c1] < pred_rank[c2] (relu(pred[c1]-pred[c2]))

Cervical (TCT_CCD) has 4 axes (squamous/glandular/infection/adequacy);
squamous and infection have >=2 classes each so get heads; glandular and
adequacy are 1-class and skipped (no rank ladder).

Total loss = (mean | sum) over active (organ, axis) pairs, selected by the
``normalization`` flag — see OrganOrdinalLoss docstring.

Label quality (audited 2026-05-15): rank_along_axis comes from PSC/Bethesda/
Paris Roman-numeral parsing in tools/build_taxonomy_metadata.py. Some axes
have multiple classes sharing the same rank (medically correct: classes
within the same severity tier of the system, e.g., 5 normal cell types all
at PSC Category II for respiratory tract). MSE-based ord_loss forces such
classes to project to the same scalar, fighting cls loss that wants
class discriminability. Use ``exclude_organ_axes`` / ``min_unique_ranks``
/ ``skip_collision_ranks`` to filter these problematic axes.
"""
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class OrganOrdinalLoss(nn.Module):
    """Per-axis ordinal regression with monotonicity penalty.

    Args:
        embed_dim: class vector dimension (== adapter output dim).
        num_organs: total organs.
        max_axes_per_organ: max axes per organ in the taxonomy.
        loss_weight: scalar multiplier on the total ordinal loss.
        monotonicity_weight: weight on monotonicity penalty within (organ, axis).
        normalization: 'mean' divides the total by the count of active
            (organ, axis) pairs (original design; used by the M2 baseline
            ckpt). 'sum' leaves the per-axis MSE+monotonicity terms summed
            (used by the auxfix v2 and axisstruct ckpts). On TCT_NGC 'mean'
            makes loss_ord ~1e-3 which is drowned by loss_cls; 'sum' brings
            it to ~5-30 with real gradient impact. Callers MUST match this
            flag to whatever the trained checkpoint was produced with — a
            silent mismatch retrains a different model.
        exclude_organ_axes: list of (organ_id, axis_id) tuples to skip
            entirely. Used to disable broken axes where the medical rank
            labels are not actually ordinal (e.g., TCT_CCD infection axis 2
            where the labels are different organisms, not severity levels)
            or where the loss formulation conflicts with class
            discriminability (e.g., respiratory axis 0 where 5 normal cell
            types share PSC Category II = rank 2 — MSE forces them to the
            same scalar projection, fighting cls loss).
        min_unique_ranks: skip (organ, axis) pairs that have fewer than
            this many unique ranks among active classes. Default 2 (the
            old "skip degenerate axis" behavior is min_unique_ranks=2 since
            a single-rank axis has 1 unique rank). Setting to 3 also skips
            binary axes (e.g., Serous effusion Negative vs Diseased).
        skip_collision_ranks: when True, drop classes whose rank value is
            shared by another class within the same (organ, axis). Keeps
            only rank-unique exemplars in ord_loss supervision. Mitigates
            the MSE collision problem (5 normal cells at rank 2 become
            skipped, leaving only Impurity and Diseased for the projection
            to learn from).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_organs: int = 5,
        max_axes_per_organ: int = 4,
        loss_weight: float = 0.3,
        monotonicity_weight: float = 0.5,
        normalization: str = 'mean',
        exclude_organ_axes: Optional[Iterable[Tuple[int, int]]] = None,
        min_unique_ranks: int = 2,
        skip_collision_ranks: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_organs = num_organs
        self.max_axes_per_organ = max_axes_per_organ
        self.loss_weight = loss_weight
        self.monotonicity_weight = monotonicity_weight
        if normalization not in ('mean', 'sum'):
            raise ValueError(
                f"normalization must be 'mean' or 'sum', got {normalization!r}")
        self.normalization = normalization
        # Normalize exclude_organ_axes to a frozenset of int tuples for fast lookup
        if exclude_organ_axes is None:
            self.exclude_organ_axes = frozenset()
        else:
            self.exclude_organ_axes = frozenset(
                (int(o), int(a)) for o, a in exclude_organ_axes
            )
        if min_unique_ranks < 2:
            raise ValueError(
                f'min_unique_ranks must be >= 2 (need at least 2 ranks to '
                f'define order), got {min_unique_ranks}')
        self.min_unique_ranks = min_unique_ranks
        self.skip_collision_ranks = skip_collision_ranks

        # Per (organ, axis) ordinal head: linear scalar projection.
        # Shape [num_organs, max_axes, embed_dim] for weight; [num_organs, max_axes] for bias.
        # Only the (organ, axis) cells with active classes will be trained;
        # others receive zero gradient and remain at init.
        self.head_w = nn.Parameter(
            torch.randn(num_organs, max_axes_per_organ, embed_dim) * 0.01
        )
        self.head_b = nn.Parameter(
            torch.zeros(num_organs, max_axes_per_organ)
        )

        # Active-axis lookup table — populated lazily on first forward from
        # the (organ_ids, axis_ids, ranks) input. Result is cached because
        # class metadata never changes during training, so the per-axis
        # decision (exclude / skip_collision / min_unique_ranks) is constant.
        # Eliminates per-iter CPU-GPU sync from `.tolist()` calls.
        # Stored on CPU; moved to emb.device on each forward (cheap copy of
        # tiny LongTensors, no sync).
        #
        # CAVEAT: This cache is bound to the FIRST forward call's class
        # metadata. If you ever swap the dataset / class set on the same
        # OrganOrdinalLoss instance (e.g. fine-tune base → eval novel on
        # the same loss object), call `self._active_axes = None` to force
        # a rebuild. Within a single training run the broadcast-text setup
        # keeps the class set fixed, so this never matters in practice.
        self._active_axes: Optional[List[Tuple[int, int, torch.Tensor]]] = None

    def _build_active_axes(
        self,
        class_organ_ids: torch.Tensor,
        class_axis_ids: torch.Tensor,
        class_ranks: torch.Tensor,
    ) -> List[Tuple[int, int, torch.Tensor]]:
        """One-time CPU-side computation of which (organ, axis) pairs and
        which class indices participate in ord_loss, after applying the
        exclude_organ_axes / skip_collision_ranks / min_unique_ranks
        filters. Called once on first forward and cached on self.
        """
        from collections import Counter
        organs = class_organ_ids.cpu().tolist()
        axes = class_axis_ids.cpu().tolist()
        ranks = class_ranks.cpu().tolist()
        out: List[Tuple[int, int, torch.Tensor]] = []
        for o in range(self.num_organs):
            for a in range(self.max_axes_per_organ):
                if (o, a) in self.exclude_organ_axes:
                    continue
                idxs = [i for i, (oi, ai, ri)
                        in enumerate(zip(organs, axes, ranks))
                        if oi == o and ai == a and ri >= 0]
                if len(idxs) < 2:
                    continue
                r_sub = [ranks[i] for i in idxs]
                if self.skip_collision_ranks:
                    rank_count = Counter(r_sub)
                    keep = [k for k, r in enumerate(r_sub)
                            if rank_count[r] == 1]
                    if len(keep) < 2:
                        continue
                    idxs = [idxs[k] for k in keep]
                    r_sub = [r_sub[k] for k in keep]
                if len(set(r_sub)) < self.min_unique_ranks:
                    continue
                out.append((o, a, torch.tensor(idxs, dtype=torch.long)))
        return out

    def forward(
        self,
        emb_final: torch.Tensor,
        class_organ_ids: torch.Tensor,
        class_axis_ids: torch.Tensor,
        class_ranks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-axis ordinal loss + monotonicity.

        Args:
            emb_final: [B, C, D] adapter output. Average over batch since
                       all batch items share the same class set under our
                       broadcast text setup.
            class_organ_ids: [C]
            class_axis_ids:  [C]
            class_ranks:     [C]  (rank_along_axis, -1 = unknown)

        Returns:
            {'loss_ord': scalar tensor}
        """
        # Average emb_final over batch (all batch items share the same C
        # class vectors under our broadcast text setup). Under DDP each rank
        # only averages its local batch; the ordinal head's gradients are
        # then allreduced across ranks. Mathematically OK (just adds noise);
        # documented here for future readers.
        emb = emb_final.mean(dim=0)  # [C, D]
        C, D = emb.shape

        device = emb.device
        total_loss = emb.new_zeros(1)
        active_axes = 0

        if self._active_axes is None:
            self._active_axes = self._build_active_axes(
                class_organ_ids, class_axis_ids, class_ranks)

        # Iterate the pre-filtered (organ, axis, kept_idxs) triples.
        # No CPU-GPU sync per iter — idxs is a tiny CPU LongTensor moved
        # to device once per axis (cheap).
        for o, a, idxs_cpu in self._active_axes:
            idxs = idxs_cpu.to(device, non_blocking=True)
            emb_sub = emb[idxs]               # [n, D]
            ranks_sub = class_ranks[idxs].to(emb.dtype)   # [n]

            # Predict scalar rank
            w = self.head_w[o, a]              # [D]
            b = self.head_b[o, a]
            pred = (emb_sub @ w) + b           # [n]

            # MSE
            mse = F.mse_loss(pred, ranks_sub)

            # Monotonicity: for each pair (i, j) with rank[i] < rank[j],
            # penalize pred[i] >= pred[j].
            pair_rank_diff = ranks_sub.unsqueeze(0) - ranks_sub.unsqueeze(1)  # [n, n]
            pair_pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)            # [n, n]
            # Valid pairs: rank[i] < rank[j] (column > row)
            valid = (pair_rank_diff > 0).float()
            # Violation: pred[j] - pred[i] should be > 0 (pred increases with rank)
            # i.e., pair_pred_diff > 0. Penalize when <= 0.
            violation = F.relu(-pair_pred_diff) * valid
            mono = violation.sum() / valid.sum().clamp(min=1.0)

            total_loss = total_loss + mse + self.monotonicity_weight * mono
            active_axes += 1

        if self.normalization == 'mean' and active_axes > 0:
            total_loss = total_loss / active_axes
        # 'sum' leaves total_loss as the sum over active (organ, axis) pairs.
        # See class docstring for which trained ckpts used which mode.

        return {'loss_ord': self.loss_weight * total_loss.squeeze()}
