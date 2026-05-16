"""OC-HMTA Module 2: Hierarchical Text Adapter.

Three orthogonal routing dimensions on top of frozen 5-attribute BiomedCLIP
embeddings:

  Stage 1 — Per-attribute projection + content-aware attention pool
            (routing dim: attribute index 0..4; NOT per-system to avoid
             collinearity with Stage 2's organ axis on TCT_NGC where
             system_id and organ_id are bijective).

  Stage 2 — Per-organ soft MoE with prior bias
            (routing dim: organ_id 0..4; soft gate biased toward
             class.organ_id, allows borderline soft mixing).

  Stage 3 — Rank embedding additive bypass
            (routing dim: severity rank within organ_axis; cervical
             has 4 sub-axes (squamous/glandular/infection/adequacy),
             other organs have a single 'primary' axis).

Anti-collapse guards (DEAD-6 prevention):
  - Non-uniform init bias for Stage 1 attention (favor discriminative attrs)
  - Orthogonal init scaled 0.5 for Stage 1 projections (avoid identity)
  - Stage 2 prior_bias init = +5 to class.organ (gate dominates organ at init)
  - Stage 3 rank_emb init σ=0.05 (non-zero start)
  - get_aux_losses() returns regularizers that penalize:
      * uniform α (Stage 1 pool entropy)
      * tiny rank_emb norm (Stage 3 magnitude)
      * uniform gate (Stage 2 entropy)
  - get_collapse_diagnostics() returns scalar health metrics for
    AdapterCollapseGuard hook to log/halt on.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class HierarchicalTextAdapter(BaseModule):
    """Module 2 hierarchical adapter — attribute proj + organ MoE + rank emb."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_attrs: int = 5,
        attr_hidden: int = 128,
        num_organs: int = 5,
        organ_hidden: int = 128,
        # Rank embedding table is constructed from per-(organ, axis) max ranks.
        # We allocate generously and use only the valid (organ, axis, rank) slots.
        max_axes_per_organ: int = 4,    # cervical has 4 axes; others have 1
        max_rank_per_axis: int = 7,     # PSC/Bethesda go up to VI; spare slack
        # Stage 1 attention pool init bias favoring discriminative attrs
        # (a=2 cytomorph, a=3 background, a=4 key_distinguishing).
        attn_init_bias=(-0.5, -0.5, 0.5, 0.5, 0.5),
        # Stage 2 prior bias magnitude. softmax(+5, 0, 0, 0, 0) ≈ 0.95 dominance.
        gate_prior_strength: float = 5.0,
        # Stage 3 rank embedding init scale.
        rank_emb_init_std: float = 0.05,
        # Anti-collapse regularizer weights (kept tiny, primary loss = det).
        lambda_pool_entropy: float = 0.02,
        lambda_proj_drift: float = 0.001,
        lambda_gate_entropy: float = 0.02,
        lambda_rank_norm: float = 0.01,
        rank_norm_eps: float = 0.05,    # rank_emb norm should stay > this
        # Row 5: (organ, axis)-conditional structure loss.
        # Same (organ, axis) pairs are pulled together (cos >= axis_attract_target),
        # cross-organ pairs are pushed apart (cos <= cross_organ_repel_target),
        # same-organ cross-axis pairs are left neutral (no penalty). This avoids
        # the M2 trade-off where Stage 2 organ MoE forces cross-organ kin (e.g.,
        # respiratory-Alveolar macrophages vs Thyroid-Macrophages) into different
        # subspaces despite identical cytomorphology. Disabled by default
        # (lambda=0); enable for the OCHMTA + axis 结构损失 ablation row.
        lambda_axis_attract: float = 0.0,
        lambda_cross_organ_repel: float = 0.0,
        axis_attract_target: float = 0.5,     # same (organ, axis) cos floor
        cross_organ_repel_target: float = 0.1,  # cross-organ cos ceiling
        # Row 6c knockout flags — used to isolate which HTA stages help/hurt.
        # bypass-experiment results (Table C) showed Stage 2 organ MoE is the
        # main novel-zero-shot killer (+4.8pp on bypass) and Stage 3 rank emb
        # lookup is a secondary killer (+2.3pp). These flags allow training a
        # model from scratch with those stages disabled, to confirm whether
        # the harm is intrinsic to the design or recoverable via training.
        skip_stage2: bool = False,
        skip_stage3_rank_emb: bool = False,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg)
        self.embed_dim = embed_dim
        self.num_attrs = num_attrs
        self.num_organs = num_organs
        self.gate_prior_strength = gate_prior_strength
        self.lambda_pool_entropy = lambda_pool_entropy
        self.lambda_proj_drift = lambda_proj_drift
        self.lambda_gate_entropy = lambda_gate_entropy
        self.lambda_rank_norm = lambda_rank_norm
        self.rank_norm_eps = rank_norm_eps
        self.lambda_axis_attract = lambda_axis_attract
        self.lambda_cross_organ_repel = lambda_cross_organ_repel
        self.axis_attract_target = axis_attract_target
        self.cross_organ_repel_target = cross_organ_repel_target
        self.max_axes_per_organ = max_axes_per_organ
        self.max_rank_per_axis = max_rank_per_axis
        self.skip_stage2 = skip_stage2
        self.skip_stage3_rank_emb = skip_stage3_rank_emb

        # --- Stage 1: per-attribute projection ---
        # Each attribute has its own thin MLP. Independent because attributes
        # carry semantically different signals (specimen vs cytomorphology vs
        # diagnostic_code, etc.).
        self.attr_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, attr_hidden),
                nn.GELU(),
                nn.Linear(attr_hidden, embed_dim),
            )
            for _ in range(num_attrs)
        ])
        # Orthogonal init scaled 0.5 — avoids identity-mapping as a trivial
        # solution (anti-collapse for proj_drift regularizer).
        for proj in self.attr_projs:
            for layer in proj:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Stage 1 content-aware attention pool: score(c, a) = linear(adapted)
        self.attn_score = nn.Linear(embed_dim, 1, bias=True)
        # Non-uniform init bias per attribute (favor a=2/3/4 discriminative)
        # NOTE: the bias here is added to the per-class content score AFTER
        # linear; the per-attribute bias is implemented as a learnable
        # per-attribute scalar that gets summed with the content-derived score.
        self.register_parameter(
            'attn_attr_bias',
            nn.Parameter(torch.tensor(list(attn_init_bias), dtype=torch.float32)),
        )

        # --- Stage 2: per-organ soft MoE with prior bias ---
        self.organ_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, organ_hidden),
                nn.GELU(),
                nn.Linear(organ_hidden, embed_dim),
            )
            for _ in range(num_organs)
        ])
        for expert in self.organ_experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        # Gate is content-driven: g_o = softmax(W·x + prior_bias[class.organ])
        # where prior_bias[class.organ] is a one-hot scaled by gate_prior_strength.
        self.gate_W = nn.Linear(embed_dim, num_organs, bias=True)
        # Identity-ish init for content-driven part — the dominant signal
        # at start is the prior_bias, content learns to add nuance.
        nn.init.zeros_(self.gate_W.weight)
        nn.init.zeros_(self.gate_W.bias)

        # --- Stage 3: rank embedding ---
        # Shape: [num_organs, max_axes_per_organ, max_rank_per_axis, embed_dim]
        # Only the (organ, axis, rank) cells corresponding to actual classes
        # in the taxonomy are populated; others remain at init noise but are
        # never indexed (so they don't affect output).
        self.rank_emb = nn.Parameter(
            torch.randn(
                num_organs, max_axes_per_organ, max_rank_per_axis, embed_dim
            ) * rank_emb_init_std
        )

        # --- Diagnostic buffers for collapse guard ---
        # Populated on every forward; consumed by hooks.
        self._diag = {}

        # When skip_stage2 / skip_stage3_rank_emb are True the corresponding
        # forward path is bypassed but the nn.Parameters still live on the
        # module and are tracked by DDP's `find_unused_parameters=True` (which
        # scans every backward at O(#params) cost) and allreduced (a wasted
        # buffer). Freezing them removes them from autograd + DDP traffic so
        # iter time matches the equivalent slim model.
        if self.skip_stage2:
            for p in self.organ_experts.parameters():
                p.requires_grad = False
            for p in self.gate_W.parameters():
                p.requires_grad = False
        if self.skip_stage3_rank_emb:
            self.rank_emb.requires_grad = False

    def _stage1_attribute(self, emb_attr: torch.Tensor) -> torch.Tensor:
        """Apply per-attribute projection + content-aware attention pool.

        Args:
            emb_attr: [B, C, A, D]  pre-pooled BiomedCLIP attr embeddings

        Returns:
            emb_pooled: [B, C, D]

        Side effect: stores LIVE (non-detached) tensors in self._diag for the
        anti-collapse regularizers to consume in get_aux_losses(). Diagnostics
        in get_collapse_diagnostics() detach for scalar logging.
        """
        B, C, A, D = emb_attr.shape
        assert A == self.num_attrs, f'expected {self.num_attrs} attrs, got {A}'

        adapted_list = []
        for a in range(A):
            x = emb_attr[:, :, a, :]                   # [B, C, D]
            adapted = self.attr_projs[a](x)            # [B, C, D]
            adapted_list.append(adapted)
        adapted_stack = torch.stack(adapted_list, dim=2)  # [B, C, A, D]

        # Content-aware attention scores + per-attribute prior bias
        content_score = self.attn_score(adapted_stack).squeeze(-1)  # [B, C, A]
        score = content_score + self.attn_attr_bias.view(1, 1, -1)
        alpha = F.softmax(score, dim=-1)                  # [B, C, A]
        emb_pooled = (alpha.unsqueeze(-1) * adapted_stack).sum(dim=2)  # [B, C, D]

        # Stash LIVE tensors for differentiable regularizers + diagnostics
        self._diag['stage1_attn_alpha'] = alpha            # [B, C, A] live
        self._diag['stage1_adapted_per_attr'] = adapted_stack  # [B, C, A, D] live
        self._diag['stage1_raw_per_attr'] = emb_attr       # [B, C, A, D] live (input)

        return emb_pooled

    def _stage2_organ(
        self,
        emb_pooled: torch.Tensor,
        class_organ_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply per-organ soft MoE with class-organ prior bias.

        Args:
            emb_pooled: [B, C, D]
            class_organ_ids: [C] organ_id of each class

        Returns:
            emb_organ: [B, C, D]
            sets self._diag['stage2_gate'] = [B, C, O]
                 self._diag['stage2_organ_dominance'] = fraction of classes
                    whose gate puts max mass on their class.organ_id.
        """
        B, C, D = emb_pooled.shape

        # Content-driven gate logits
        content_logits = self.gate_W(emb_pooled)         # [B, C, O]

        # Per-class prior bias: one-hot to class.organ scaled by gate_prior_strength
        prior_bias = F.one_hot(class_organ_ids, num_classes=self.num_organs).to(
            dtype=content_logits.dtype, device=content_logits.device
        ) * self.gate_prior_strength                      # [C, O]
        gate_logits = content_logits + prior_bias.unsqueeze(0)  # [B, C, O]
        gate = F.softmax(gate_logits, dim=-1)             # [B, C, O]

        # Compute expert outputs then weighted sum
        expert_outs = []
        for o in range(self.num_organs):
            out_o = self.organ_experts[o](emb_pooled)    # [B, C, D]
            expert_outs.append(out_o)
        expert_stack = torch.stack(expert_outs, dim=-2)  # [B, C, O, D]
        emb_organ = (gate.unsqueeze(-1) * expert_stack).sum(dim=-2)  # [B, C, D]

        # Stash LIVE gate for differentiable entropy regularizer
        self._diag['stage2_gate'] = gate                 # [B, C, O] live
        self._diag['stage2_class_organ_ids'] = class_organ_ids   # [C] for dominance metric

        return emb_organ

    def _stage3_rank(
        self,
        emb_organ: torch.Tensor,
        class_organ_ids: torch.Tensor,
        class_axis_ids: torch.Tensor,
        class_ranks: torch.Tensor,
    ) -> torch.Tensor:
        """Add rank embedding (organ, axis, rank) to emb_organ.

        Args:
            emb_organ: [B, C, D]
            class_organ_ids: [C]
            class_axis_ids:  [C]   (0 = primary, cervical: 0..3)
            class_ranks:     [C]   (rank_along_axis, may be -1 for unknown)

        Returns:
            emb_final: [B, C, D]
            sets self._diag['stage3_rank_norm_min'] = min ||rank_emb|| across
                 active (organ, axis, rank) cells.
        """
        B, C, D = emb_organ.shape

        # Build [C, D] rank embedding lookup
        # For unknown rank (rank == -1), use zero (no rank info, neutral add).
        rank_clamped = class_ranks.clamp(min=0)          # [C]
        rank_emb_lookup = self.rank_emb[
            class_organ_ids, class_axis_ids, rank_clamped
        ]                                                # [C, D]
        # Zero out classes with unknown rank
        valid_mask = (class_ranks >= 0).to(rank_emb_lookup.dtype).unsqueeze(-1)
        rank_emb_lookup = rank_emb_lookup * valid_mask    # [C, D]

        emb_final = emb_organ + rank_emb_lookup.unsqueeze(0)  # [B, C, D]

        # Stash LIVE tensors for differentiable rank-norm regularizer + diag.
        # Use the masked-out lookup (cross-organ already zeroed) plus the
        # valid_mask so the regularizer only sees real (organ, axis, rank) cells.
        self._diag['stage3_rank_emb_lookup'] = rank_emb_lookup  # [C, D] live (mask-applied)
        self._diag['stage3_rank_valid_mask'] = (class_ranks >= 0)  # [C] bool

        return emb_final

    def forward(
        self,
        emb_attr: torch.Tensor,
        class_organ_ids: torch.Tensor,
        class_axis_ids: torch.Tensor,
        class_ranks: torch.Tensor,
    ) -> torch.Tensor:
        """Run all 3 stages.

        Args:
            emb_attr: [B, C, A, D]
            class_organ_ids: [C]
            class_axis_ids:  [C]
            class_ranks:     [C]

        Returns:
            emb_final: [B, C, D]
        """
        emb_pooled = self._stage1_attribute(emb_attr)
        if self.skip_stage2:
            # Bypass Stage 2 organ MoE entirely (Row 6c). Stage 1 output flows
            # directly into Stage 3 (or directly out if Stage 3 is also skipped).
            # get_aux_losses() gates on `not self.skip_stage2 and 'stage2_gate'
            # in self._diag`, so no _diag write is needed here.
            emb_organ = emb_pooled
        else:
            emb_organ = self._stage2_organ(emb_pooled, class_organ_ids)
        if self.skip_stage3_rank_emb:
            # Bypass Stage 3 rank embedding lookup (Row 6c). The class vector
            # does not get a per-(organ,axis,rank) additive vector — protects
            # novel classes from the noise-init rank embedding for unseen
            # ranks. ord_loss can still operate on emb_organ directly.
            # get_aux_losses() gates on `not self.skip_stage3_rank_emb and
            # 'stage3_rank_emb_lookup' in self._diag`, so no _diag write
            # is needed here.
            emb_final = emb_organ
        else:
            emb_final = self._stage3_rank(
                emb_organ, class_organ_ids, class_axis_ids, class_ranks
            )
        # Stash live emb_final + class metadata for the (organ, axis)-cond
        # structure loss in get_aux_losses(). Only populated when the Row 5
        # losses are actually enabled — otherwise these tensors are unused
        # per-iter overhead. Mean over batch dim (all batch items share the
        # same C class vectors under our broadcast text setup).
        if self.lambda_axis_attract > 0 or self.lambda_cross_organ_repel > 0:
            self._diag['final_emb_mean'] = emb_final.mean(dim=0)    # [C, D] live
            self._diag['class_organ_ids'] = class_organ_ids          # [C] long
            self._diag['class_axis_ids'] = class_axis_ids            # [C] long
        return emb_final

    def get_aux_losses(self) -> Dict[str, torch.Tensor]:
        """Differentiable anti-collapse regularizer losses.

        Must be called AFTER forward (uses live tensors stored in self._diag).
        Every regularizer has true gradient flow back to its target params:
          - loss_pool_entropy   -> attn_score, attn_attr_bias, attr_projs
          - loss_proj_drift     -> attr_projs weights (via raw vs adapted diff)
          - loss_gate_entropy   -> gate_W, organ_experts
          - loss_rank_norm      -> rank_emb (only active organ-axis-rank cells)
        """
        losses = {}
        if not self._diag:
            return losses

        # Stage 1 pool entropy: penalize uniform attn (= mean-pool collapse).
        # Mean entropy is the natural scalar; backprop pushes alpha distribution
        # toward concentrated per-class winners.
        alpha = self._diag['stage1_attn_alpha']           # [B, C, A] live
        entropy_alpha = -(alpha * (alpha.clamp(min=1e-9).log())).sum(dim=-1)
        losses['loss_pool_entropy'] = (
            self.lambda_pool_entropy * entropy_alpha.mean()
        )

        # Stage 1 proj drift: penalize attr_proj(x) ≈ x (identity collapse).
        # drift_per_attr is the mean L2 distance ratio over (B, C) samples;
        # relu(min_drift_floor - drift) creates gradient ONLY when an attr's
        # projection moves too close to identity.
        adapted = self._diag['stage1_adapted_per_attr']    # [B, C, A, D] live
        raw = self._diag['stage1_raw_per_attr']            # [B, C, A, D]
        drift = ((adapted - raw).norm(dim=-1)              # [B, C, A]
                 / raw.norm(dim=-1).clamp(min=1e-6))
        drift_per_attr = drift.mean(dim=(0, 1))            # [A]
        drift_penalty = F.relu(0.05 - drift_per_attr).pow(2).sum()
        losses['loss_proj_drift'] = self.lambda_proj_drift * drift_penalty

        # Stage 2 gate entropy: penalize uniform gate (= single big MLP).
        # Skip when Stage 2 is disabled (skip_stage2=True): no gate to regularize.
        if not self.skip_stage2 and 'stage2_gate' in self._diag:
            gate = self._diag['stage2_gate']                   # [B, C, O] live
            entropy_gate = -(gate * (gate.clamp(min=1e-9).log())).sum(dim=-1)
            losses['loss_gate_entropy'] = (
                self.lambda_gate_entropy * entropy_gate.mean()
            )

        # Stage 3 rank norm: penalize rank_emb learning -> 0 vector.
        # Skip when Stage 3 rank emb is disabled: there's no rank embedding
        # lookup to regularize.
        if not self.skip_stage3_rank_emb and 'stage3_rank_emb_lookup' in self._diag:
            rank_emb_lookup = self._diag['stage3_rank_emb_lookup']   # [C, D] live
            valid_mask = self._diag['stage3_rank_valid_mask']        # [C] bool
            if valid_mask.any():
                active_norms = rank_emb_lookup[valid_mask].norm(dim=-1)  # [n_valid]
                min_norm = active_norms.min()
                penalty = F.relu(self.rank_norm_eps - min_norm).pow(2)
                losses['loss_rank_norm'] = self.lambda_rank_norm * penalty

        # Row 5: (organ, axis)-conditional structure loss
        if (self.lambda_axis_attract > 0 or self.lambda_cross_organ_repel > 0) \
                and 'final_emb_mean' in self._diag:
            emb = F.normalize(self._diag['final_emb_mean'], dim=-1)  # [C, D]
            organ_ids = self._diag['class_organ_ids']                # [C]
            axis_ids = self._diag['class_axis_ids']                  # [C]
            C = emb.shape[0]
            cos = emb @ emb.T                                         # [C, C]
            # Mask out diagonal and lower triangle (use each pair once).
            triu = torch.triu(
                torch.ones(C, C, dtype=torch.bool, device=emb.device), diagonal=1
            )
            same_organ = (organ_ids.unsqueeze(0) == organ_ids.unsqueeze(1))
            same_axis = (axis_ids.unsqueeze(0) == axis_ids.unsqueeze(1))
            # (a) Attract: same (organ, axis) pairs cos should be >= target
            attract_mask = triu & same_organ & same_axis
            if self.lambda_axis_attract > 0 and attract_mask.any():
                attract_pairs = cos[attract_mask]
                attract_loss = F.relu(
                    self.axis_attract_target - attract_pairs
                ).pow(2).mean()
                losses['loss_axis_attract'] = self.lambda_axis_attract * attract_loss
            # (b) Repel: cross-organ pairs cos should be <= target
            repel_mask = triu & (~same_organ)
            if self.lambda_cross_organ_repel > 0 and repel_mask.any():
                repel_pairs = cos[repel_mask]
                repel_loss = F.relu(
                    repel_pairs - self.cross_organ_repel_target
                ).pow(2).mean()
                losses['loss_cross_organ_repel'] = self.lambda_cross_organ_repel * repel_loss
            # (c) Same-organ cross-axis: neutral, no loss (preserves cross-organ-kin
            # like Thyroid-Macrophages ↔ Thyroid-PTC at adapter's discretion).

        return losses

    def get_collapse_diagnostics(self) -> Dict[str, float]:
        """Scalar health metrics for AdapterCollapseGuard hook to log/halt on.

        Reads the same live tensors used by get_aux_losses() but detaches
        and converts to Python floats for non-differentiable logging.
        """
        diag = {}
        if 'stage1_attn_alpha' in self._diag:
            alpha = self._diag['stage1_attn_alpha'].detach()
            entropy = -(alpha * (alpha.clamp(min=1e-9).log())).sum(dim=-1)
            diag['stage1_alpha_entropy_mean'] = float(entropy.mean().item())
            diag['stage1_alpha_entropy_max'] = float(entropy.max().item())
        if 'stage1_adapted_per_attr' in self._diag:
            with torch.no_grad():
                adapted = self._diag['stage1_adapted_per_attr']
                raw = self._diag['stage1_raw_per_attr']
                drift = ((adapted - raw).norm(dim=-1)
                         / raw.norm(dim=-1).clamp(min=1e-6))
                drift_per_attr = drift.mean(dim=(0, 1))         # [A]
                drifts_list = [float(d.item()) for d in drift_per_attr]
            diag['stage1_proj_drift_min'] = float(min(drifts_list))
            diag['stage1_proj_drift_per_attr'] = drifts_list
        if 'stage2_gate' in self._diag:
            with torch.no_grad():
                gate = self._diag['stage2_gate']
                entropy = -(gate * (gate.clamp(min=1e-9).log())).sum(dim=-1)
                diag['stage2_gate_entropy_mean'] = float(entropy.mean().item())
                # Organ dominance: fraction of (B, C) where argmax(gate)==class.organ
                class_organ_ids = self._diag.get('stage2_class_organ_ids')
                if class_organ_ids is not None:
                    dominant = gate.argmax(dim=-1)
                    target = class_organ_ids.view(1, -1).expand_as(dominant)
                    diag['stage2_organ_dominance'] = float(
                        (dominant == target).float().mean().item())
        if 'stage3_rank_emb_lookup' in self._diag:
            with torch.no_grad():
                lookup = self._diag['stage3_rank_emb_lookup']
                valid = self._diag['stage3_rank_valid_mask']
                if valid.any():
                    active_norms = lookup[valid].norm(dim=-1)
                    diag['stage3_rank_norm_min'] = float(
                        active_norms.min().item())
                else:
                    diag['stage3_rank_norm_min'] = float('nan')
        return diag
