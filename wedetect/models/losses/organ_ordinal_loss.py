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

Total loss = sum over active (organ, axis) pairs.
"""
from typing import Dict

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
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_organs: int = 5,
        max_axes_per_organ: int = 4,
        loss_weight: float = 0.3,
        monotonicity_weight: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_organs = num_organs
        self.max_axes_per_organ = max_axes_per_organ
        self.loss_weight = loss_weight
        self.monotonicity_weight = monotonicity_weight

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

        # Iterate unique (organ, axis) combos
        for o in range(self.num_organs):
            for a in range(self.max_axes_per_organ):
                # Classes belonging to this (organ, axis) with valid rank
                mask = ((class_organ_ids == o) & (class_axis_ids == a)
                        & (class_ranks >= 0))
                idxs = mask.nonzero(as_tuple=False).squeeze(-1)
                if idxs.numel() < 2:
                    continue  # degenerate axis (single class)
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
                n = pred.shape[0]
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

        if active_axes > 0:
            total_loss = total_loss / active_axes

        return {'loss_ord': self.loss_weight * total_loss.squeeze()}
