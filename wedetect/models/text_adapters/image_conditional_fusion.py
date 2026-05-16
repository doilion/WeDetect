"""Design A — Image-Conditional Fusion (ICF).

Replaces THAF's class-agnostic learnable fusion_query with an image-derived
context vector. The fused per-class embedding now varies per image, breaking
the mean-pool basin of attraction that killed THAF (α→0 collapse, memory
DEAD-11 / feedback_thaf_fusion_bypassed.md).

Architecture (F2 — with per-attribute expert MLPs, ~2.4M params):

    image_top_feat [B, C_img, H, W]
        -> adaptive_avg_pool2d 1x1                 # global context
        -> Linear image_proj : C_img -> D          # match text dim
        -> image_ctx [B, D]

    attr_emb [B, C, A, D]      (cached BiomedCLIP 5-attr embeddings)
        -> for a in range(A): attr_experts[a](attr_emb[:,:,a,:])
                                                   # per-attribute MLP
        -> attr_adapted [B, C, A, D]
        + attr_type_pe positional emb
        -> K, V flattened to [B*C, A, D]

    Q := image_ctx broadcast over C classes -> [B*C, 1, D]
    cross-attn(Q, K, V)
        -> fused [B*C, 1, D]
    output_proj + L2-norm
        -> [B, C, D]

NO residual blend back to attr_mean (THAF's α gate gave optimizer an escape
hatch). NO 4D expansion MLP (kept small for stable cold start). Pre-norm
attention for training stability.

The per-attribute experts are STATIC routing — attribute index decides which
expert handles it. Unlike M2 Stage 2 per-organ MoE (which split training
data by organ and failed novel zero-shot), every training sample contributes
to every attribute expert, so novel-class attribute_a vectors are processed
by an expert trained on the full data distribution.

Collapse diagnostics consumed by ICFCollapseGuard hook (3 metrics):
  - fused_pairwise_cos_mean: mean off-diagonal cosine between fused class
        vectors of the SAME class across images in the batch. Should be in
        [0.5, 0.95]. > 0.99 means image-invariant fusion = mean-pool collapse.
  - attn_entropy_mean: mean entropy of attention over 5 attribute keys.
        Should be below log(5) ≈ 1.609; > 1.58 means uniform = mean pool.
  - cos_to_attr_mean_mean: cosine between fused output and (normalized)
        attr_mean. Should be < 0.95; > 0.97 means fused == mean direction.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class ImageConditionalFusion(BaseModule):
    """Image-conditional cross-attention pooling over text attributes."""

    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 768,
        num_attrs: int = 5,
        attr_hidden: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg)
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.num_attrs = num_attrs

        # (a) Image global context -> text dim
        self.image_proj = nn.Linear(image_dim, text_dim)

        # (b) Per-attribute expert MLPs (F2). 5 parallel attribute-specific
        # 2-layer MLPs, each ≈ 131K params. Static routing: attribute index
        # decides which expert. NOT routing-style MoE.
        self.attr_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, attr_hidden),
                nn.GELU(),
                nn.Linear(attr_hidden, text_dim),
            )
            for _ in range(num_attrs)
        ])

        # (c) Per-attribute positional embedding (so cross-attn can
        # distinguish which of the 5 attributes a key came from)
        self.attr_type_pe = nn.Embedding(num_attrs, text_dim)

        # (d) Pre-norm both sides of cross-attention
        self.norm_q = nn.LayerNorm(text_dim)
        self.norm_kv = nn.LayerNorm(text_dim)

        # (e) Single cross-attn block — Q from image context, K/V from 5 attrs
        self.cross_attn = nn.MultiheadAttention(
            text_dim, num_heads, dropout=dropout, batch_first=True
        )

        # (f) Output projection (NO 4D expansion, NO residual back to attr_mean)
        self.output_proj = nn.Linear(text_dim, text_dim)

        self._init_weights()

        # Live diagnostic buffer (populated each forward, consumed by
        # ICFCollapseGuard hook; cleared on each forward call).
        self._diag: Dict[str, torch.Tensor] = {}

    def _init_weights(self) -> None:
        # image_proj / output_proj: orthogonal init at scale 0.5 — non-identity
        # but not too aggressive. image_proj at init gives a usable image
        # context vector even before training.
        nn.init.orthogonal_(self.image_proj.weight, gain=0.5)
        nn.init.zeros_(self.image_proj.bias)
        nn.init.orthogonal_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)

        # Per-attribute experts: orthogonal init scaled 0.5 so each expert
        # starts as a non-trivial transform of its attribute.
        for expert in self.attr_experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

        # attr_type positional emb: small Gaussian (standard transformer style)
        nn.init.normal_(self.attr_type_pe.weight, std=0.02)

        # Cross-attn weights: xavier with moderate gain (0.5). Smaller gains
        # (e.g. 0.1, which THAF used) leave Q ≈ 0 at init -> attention is
        # uniform -> output becomes image-invariant noise. Larger gain (0.5)
        # keeps the Q-K similarity strong enough that img_ctx differences
        # actually shift the attention pattern at init.
        nn.init.xavier_uniform_(self.cross_attn.in_proj_weight, gain=0.5)
        nn.init.xavier_uniform_(self.cross_attn.out_proj.weight, gain=0.5)
        if self.cross_attn.in_proj_bias is not None:
            nn.init.zeros_(self.cross_attn.in_proj_bias)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(
        self,
        image_top_feat: torch.Tensor,
        attr_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse per-class attribute embeddings conditioned on image context.

        Args:
            image_top_feat: [B, C_img, H, W] pre-neck deepest backbone feature
                (ConvNext-tiny stage 4 = 768d, stride 32).
            attr_emb: [B, C, A, D] cached per-class 5-attribute embeddings
                (L2-normalized upstream by PseudoMultiAttrLanguageBackbone
                running with pool_mode='none').

        Returns:
            [B, C, D] image-conditional class embeddings, L2-normalized.
        """
        if attr_emb.dim() != 4:
            raise ValueError(
                f'ImageConditionalFusion expects attr_emb shape [B, C, A, D] '
                f'(raw 5-attr embeddings); got shape {tuple(attr_emb.shape)}. '
                f'Check that text_model has pool_mode="none" set in the '
                f'PseudoMultiAttrLanguageBackbone config.'
            )
        B, C, A, D = attr_emb.shape
        assert A == self.num_attrs, (
            f'expected {self.num_attrs} attrs, got A={A}')
        assert D == self.text_dim, (
            f'attr_emb dim {D} != text_dim {self.text_dim}')

        # 1. Image -> global context vector in text space
        img_global = F.adaptive_avg_pool2d(image_top_feat, 1).flatten(1)
        if img_global.shape[1] != self.image_dim:
            raise ValueError(
                f'image_top_feat has {img_global.shape[1]} channels, expected '
                f'image_dim={self.image_dim}. Check backbone stage choice.')
        img_ctx = self.image_proj(img_global)                       # [B, D]

        # 2. Per-attribute expert MLPs (F2)
        attr_adapted = torch.stack(
            [self.attr_experts[a](attr_emb[:, :, a, :]) for a in range(A)],
            dim=2,
        )                                                            # [B, C, A, D]

        # 3. Add per-attribute positional embedding to K/V
        attr_pe = self.attr_type_pe.weight.view(1, 1, A, D)
        kv = (attr_adapted + attr_pe).reshape(B * C, A, D)

        # 4. Build Q (image_ctx broadcast to C classes -> [B*C, 1, D])
        q = img_ctx.unsqueeze(1).expand(B, C, D).reshape(B * C, 1, D)

        # 5. Cross-attention
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        fused, attn_weights = self.cross_attn(
            q_norm, kv_norm, kv,
            need_weights=True,
            average_attn_weights=True,
        )                                                            # [B*C, 1, D], [B*C, 1, A]
        fused = fused.squeeze(1)                                     # [B*C, D]

        # 6. Output projection + L2 norm
        out = self.output_proj(fused).reshape(B, C, D)
        out = F.normalize(out, dim=-1)

        # Stash live tensors for the collapse-guard hook (no detach — we
        # want grad-flow checks to be possible). Each forward overwrites.
        # We do NOT keep the adapted attr stack on the live diag (too big);
        # only the small final tensors needed for 3 scalar metrics.
        self._diag = {
            'fused_emb': out,
            'attn_weights': attn_weights.squeeze(1).reshape(B, C, A),
            'attr_mean_normed': F.normalize(attr_emb.mean(dim=2), dim=-1),
        }
        return out

    @torch.no_grad()
    def get_collapse_diagnostics(self) -> Dict[str, float]:
        """Three scalar health metrics consumed by ICFCollapseGuard hook.

        All metrics are well-scaled (in [0, 1] or [0, log(A)]) so thresholds
        in the guard hook don't depend on D.
        """
        diag: Dict[str, float] = {}
        if not self._diag:
            return diag

        fused = self._diag['fused_emb'].detach()                     # [B, C, D]
        attn = self._diag['attn_weights'].detach()                   # [B, C, A]
        attr_mean = self._diag['attr_mean_normed'].detach()          # [B, C, D]

        # (1) Mean pairwise cosine between fused vectors of the SAME class
        # across different images in the batch. Image-conditional fusion
        # should produce different outputs for different images of the same
        # class; collapse to mean-pool would produce identical outputs
        # (cos = 1). Range: [-1, 1], healthy ~ [0.5, 0.95], red > 0.99.
        B_, C_, D_ = fused.shape
        if B_ >= 2:
            # cos_matrix[b, a, c] = cos(fused[b, c], fused[a, c]); shape [B, B, C]
            cos_matrix = torch.einsum('bcd,acd->bac', fused, fused)
            eye = torch.eye(B_, dtype=torch.bool, device=fused.device)
            off_diag_mask = (~eye).unsqueeze(-1).expand_as(cos_matrix)
            mean_pairwise_cos = cos_matrix[off_diag_mask].mean()
            diag['fused_pairwise_cos_mean'] = float(mean_pairwise_cos.item())
            diag['fused_pairwise_dist_mean'] = float(
                (1.0 - mean_pairwise_cos).item())
        else:
            diag['fused_pairwise_cos_mean'] = float('nan')
            diag['fused_pairwise_dist_mean'] = float('nan')

        # (2) Mean attention entropy over the 5 attribute keys.
        # log(num_attrs) is uniform = effectively mean pool.
        entropy = -(attn * attn.clamp(min=1e-9).log()).sum(dim=-1)
        diag['attn_entropy_mean'] = float(entropy.mean().item())

        # (3) Mean cosine to attr_mean direction.
        # 1.0 means fused output == mean pool direction.
        cos = (fused * attr_mean).sum(dim=-1)
        diag['cos_to_attr_mean_mean'] = float(cos.mean().item())

        return diag
