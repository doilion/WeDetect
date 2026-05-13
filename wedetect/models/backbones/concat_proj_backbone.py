"""Option 4 — Concat + Linear projection backbones (no attention, no per-class).

Replaces THAF's cross-attention fusion (which trains to alpha=0,
see `docs/figures/thaf_diagnostic/`) with a simple concatenation +
linear projection.

Forward formula:
    attr_embs: [B, C, A, D]   (each L2-normalized per attribute)
    flat = attr_embs.reshape(B, C, A * D)
    fused = LayerNorm(flat)
    out = output_proj(fused)            # A*D -> D (with intermediate D)
    out = L2_normalize(out)

Rationale: the cross-attention attempts to learn how to combine 5 attribute
embeddings into one class vector. Phase 3.5 diagnostic showed the learned
α-residual gates the attention output to ~0, leaving only `attr_mean`.
A plain linear projection sidesteps the residual issue: the projection
*must* contribute or the model can't fit the contrastive loss.

Parameter count:
  - BiomedCLIP (D=512, A=5): A*D × D + D = 2560×512 + 512 ≈ 1.31M
  - XLM-R (D=768, A=5): 3840×768 + 768 ≈ 2.95M
Bigger than the per-class weighted variant (~200 params) but smaller
than THAF cross-attention (3.15M-7.1M).

Transfers to novel zero-shot naturally: the same projection is applied to
any 5-attribute input — the projection learns a "general 5-attr → class
vector" mapping that doesn't depend on which classes were seen at
training.

Only Pseudo variants are exposed (encoder frozen in all THAF phases).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor


def _validate_cache(cache: dict, embed_dim: int, cache_path: str) -> List[str]:
    if not isinstance(cache, dict) or not cache:
        raise ValueError(
            f"attr cache must be non-empty dict[str, Tensor]; got {type(cache).__name__}"
        )
    keys = list(cache.keys())
    for k in keys:
        v = cache[k]
        if not isinstance(v, Tensor):
            raise ValueError(f"cache[{k!r}] is {type(v).__name__}, expected Tensor")
        if v.dim() != 1 or v.shape[0] != embed_dim:
            raise ValueError(
                f"cache[{k!r}] shape {tuple(v.shape)} mismatch ({embed_dim},) "
                f"in {cache_path}"
            )
    return keys


class _ConcatProjBase(BaseModule):
    """Shared implementation for concat+proj backbones.

    Projection: `Linear(A*D -> D) -> GELU -> Dropout -> Linear(D -> D)`.
    Designed to be small enough not to overfit 30 classes but flexible
    enough to learn non-linear attribute combinations.
    """

    def __init__(
        self,
        *,
        attr_emb_cache_path: str,
        embed_dim: int,
        num_attr_types: int = 5,
        dropout: float = 0.1,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        cache = torch.load(attr_emb_cache_path, map_location="cpu")
        keys = _validate_cache(cache, embed_dim, attr_emb_cache_path)
        table = torch.stack([cache[k].float() for k in keys], dim=0)
        self.register_buffer("attr_emb_table", table, persistent=False)
        self._str_to_idx: Dict[str, int] = {k: i for i, k in enumerate(keys)}

        self.num_attr_types = num_attr_types
        self.embed_dim = embed_dim

        self.input_norm = nn.LayerNorm(num_attr_types * embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(num_attr_types * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        # Init projections small-ish so the model has to learn from data
        # but not so tiny that gradients vanish (cross-attention's gain=0.1
        # caused the alpha→0 collapse; we use gain=1.0 here so the projection
        # has real influence from step 0).
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def _lookup_attr_indices(self, text: List[List[List[str]]]) -> Tensor:
        B = len(text)
        C = len(text[0])
        A = self.num_attr_types
        idx = torch.empty(B * C * A, dtype=torch.long)
        unknown: List[str] = []
        i = 0
        for tb in text:
            for ca in tb:
                if len(ca) != A:
                    raise ValueError(f"expected {A} attrs per class; got len={len(ca)}")
                for s in ca:
                    if s in self._str_to_idx:
                        idx[i] = self._str_to_idx[s]
                    else:
                        unknown.append(s)
                    i += 1
        if unknown:
            raise KeyError(
                f"{len(unknown)} attribute string(s) missing from cache. "
                f"sample: {[s[:60] for s in unknown[:3]]}"
            )
        return idx

    def forward(self, text: List[List[List[str]]]) -> Tensor:
        if not isinstance(text, list) or not text:
            raise ValueError("text must be a non-empty batch list.")
        B = len(text)
        C = len(text[0])
        for tb in text:
            if not isinstance(tb, list) or len(tb) != C:
                raise ValueError("all batch entries must have same num_classes.")

        device = self.attr_emb_table.device
        attr_idx = self._lookup_attr_indices(text).to(device)
        attr_embeds = self.attr_emb_table.index_select(0, attr_idx).reshape(
            B, C, self.num_attr_types * self.embed_dim
        )

        normed = self.input_norm(attr_embeds)
        out = self.proj(normed)
        return F.normalize(out, dim=-1)

    def train(self, mode: bool = True):
        return super().train(mode)


@MODELS.register_module()
class PseudoConcatProjXLMRLanguageBackbone(_ConcatProjBase):
    """768-dim concat+proj backbone (uses XLM-R per-attr cache)."""

    def __init__(
        self,
        *,
        attr_emb_cache_path: str,
        num_attr_types: int = 5,
        embed_dim: int = 768,
        dropout: float = 0.1,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            attr_emb_cache_path=attr_emb_cache_path,
            embed_dim=embed_dim,
            num_attr_types=num_attr_types,
            dropout=dropout,
            init_cfg=init_cfg,
        )


@MODELS.register_module()
class PseudoConcatProjBiomedCLIPLanguageBackbone(_ConcatProjBase):
    """512-dim concat+proj backbone (uses BiomedCLIP per-attr cache)."""

    def __init__(
        self,
        *,
        attr_emb_cache_path: str,
        num_attr_types: int = 5,
        embed_dim: int = 512,
        dropout: float = 0.1,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            attr_emb_cache_path=attr_emb_cache_path,
            embed_dim=embed_dim,
            num_attr_types=num_attr_types,
            dropout=dropout,
            init_cfg=init_cfg,
        )
