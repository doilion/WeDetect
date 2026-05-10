"""Hierarchical attribute fusion language backbones for THAF (Phase 3a).

Two backbones are exposed:

  1. HierarchicalXLMRLanguageBackbone — full encoder + fusion. Accepts
     `text: List[List[List[str]]] = [batch][num_classes][num_attr_types]`,
     encodes every attribute string via the inherited XLM-Roberta encoder,
     then fuses via cross-attention (learnable fusion_query) + alpha-residual
     to produce `[B, num_classes, embed_dim]`.

     Inherits from `XLMRobertaLanguageBackbone` so that `self.model` and
     `self.head` keep their original parameter names. This lets us load the
     dev30 best checkpoint (which has `backbone.text_model.model.*` and
     `backbone.text_model.head.*` keys) into the parent module untouched;
     the new fusion-module attributes start fresh.

  2. PseudoHierarchicalXLMRLanguageBackbone — accelerated training variant.
     Loads pre-cached per-attr embeddings (dict of attribute string to
     768-dim Tensor) at construction; forward path skips the encoder and
     looks up cached vectors before running the same fusion module. Used
     when the encoder is frozen (which is the entire Phase 3a setup) so
     re-encoding 5 × 80 strings every batch is wasted compute. Pre-cache
     must contain every attribute string the training pipeline can sample,
     including the empty-string padding value if `padding_to_max=True`.

Both classes share the same fusion module shape (~7.1M parameters at
embed_dim=768): attr_type_embed + fusion_query + cross_attn (8 heads) +
norm1/2 + output_proj (D → 4D → D) + alpha (learnable residual scalar).

Adapted from /home/25_liwenjie/code/YOLO-World-Medical/yolo_world/models/
backbones/mm_backbone.py:3258-3497 (BiomedCLIP variant) — same fusion
topology, swapped encoder, num_attr_types 4 → 5.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor

from .mm_backbone import XLMRobertaLanguageBackbone


def _build_fusion_module(
    embed_dim: int,
    num_attr_types: int,
    num_heads: int,
    dropout: float,
    residual_alpha: float,
) -> Dict[str, nn.Module]:
    return {
        "attr_type_embed": nn.Embedding(num_attr_types, embed_dim),
        "fusion_query": nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02),
        "cross_attn": nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        ),
        "norm1": nn.LayerNorm(embed_dim),
        "norm2": nn.LayerNorm(embed_dim),
        "output_proj": nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        ),
        "alpha": nn.Parameter(torch.tensor(residual_alpha)),
    }


def _init_fusion_weights(module: nn.Module) -> None:
    nn.init.normal_(module.attr_type_embed.weight, std=0.02)
    nn.init.xavier_uniform_(module.cross_attn.in_proj_weight, gain=0.1)
    nn.init.xavier_uniform_(module.cross_attn.out_proj.weight, gain=0.1)
    nn.init.zeros_(module.cross_attn.out_proj.bias)
    for m in module.output_proj:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)


def _fuse_attr_embeds(
    module: nn.Module,
    attr_embeds: Tensor,
    num_attr_types: int,
    embed_dim: int,
) -> Tensor:
    """Run the cross-attention + residual fusion on per-attr embeddings.

    Args:
        module: holder of fusion submodules (cross_attn / norm1 / norm2 /
            output_proj / alpha / attr_type_embed / fusion_query).
        attr_embeds: [B, C, num_attr_types, embed_dim], **already
            L2-normalized** per attribute (the upstream encoder is expected
            to normalize before passing in, matching parent
            XLMRobertaLanguageBackbone behavior).

    Returns:
        Tensor of shape [B, C, embed_dim], L2-normalized.
    """
    B, C, A, D = attr_embeds.shape
    if A != num_attr_types or D != embed_dim:
        raise ValueError(
            f"attr_embeds shape ({B},{C},{A},{D}) mismatch num_attr_types="
            f"{num_attr_types} or embed_dim={embed_dim}"
        )

    attr_type = module.attr_type_embed.weight.unsqueeze(0).unsqueeze(0)
    attr_embeds = attr_embeds + attr_type  # [B, C, A, D]

    attr_flat = attr_embeds.reshape(B * C, A, D)
    query = module.fusion_query.expand(B * C, -1, -1)  # [B*C, 1, D]

    q_norm = module.norm1(query)
    k_norm = module.norm1(attr_flat)
    fused, _ = module.cross_attn(
        query=q_norm,
        key=k_norm,
        value=attr_flat,
        need_weights=False,
    )
    fused = fused.squeeze(1)  # [B*C, D]

    proj_out = module.output_proj(module.norm2(fused))
    attr_mean = attr_flat.mean(dim=1)  # [B*C, D]
    output = module.alpha * proj_out + (1.0 - module.alpha) * attr_mean
    output = F.normalize(output, dim=-1)

    return output.reshape(B, C, D)


@MODELS.register_module()
class HierarchicalXLMRLanguageBackbone(XLMRobertaLanguageBackbone):
    """XLM-Roberta encoder + trainable hierarchical attribute fusion.

    Inherits from XLMRobertaLanguageBackbone to keep `self.model` and
    `self.head` parameter names — letting the dev30 ckpt's
    `backbone.text_model.{model,head}.*` weights load into the parent
    module without remapping. Fusion submodules are added as siblings.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_size: str,
        num_attr_types: int = 5,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual_alpha: float = 0.3,
        frozen_modules: Sequence[str] = ("all",),
        training_use_cache: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_size=model_size,
            frozen_modules=frozen_modules,
            dropout=dropout,
            training_use_cache=training_use_cache,
            init_cfg=init_cfg,
        )

        if model_size in ("base", "tiny"):
            embed_dim = 768
        elif model_size == "large":
            embed_dim = 768
        elif model_size == "xlarge":
            embed_dim = 1024
        else:
            raise ValueError(f"unknown model_size {model_size!r}")

        self.num_attr_types = num_attr_types
        self.embed_dim = embed_dim

        modules = _build_fusion_module(
            embed_dim, num_attr_types, num_heads, dropout, residual_alpha
        )
        self.attr_type_embed = modules["attr_type_embed"]
        self.fusion_query = modules["fusion_query"]
        self.cross_attn = modules["cross_attn"]
        self.norm1 = modules["norm1"]
        self.norm2 = modules["norm2"]
        self.output_proj = modules["output_proj"]
        self.alpha = modules["alpha"]
        _init_fusion_weights(self)

    def forward(self, text: List[List[List[str]]]) -> Tensor:
        if not isinstance(text, list) or not text:
            raise ValueError("text must be a non-empty batch list.")
        B = len(text)
        C = len(text[0])
        if C == 0:
            raise ValueError("each batch must contain at least one class.")
        for tb in text:
            if not isinstance(tb, list) or len(tb) != C:
                raise ValueError(
                    "all batch entries must have same num_classes."
                )

        flat: List[str] = []
        for tb in text:
            for cls_attrs in tb:
                if not isinstance(cls_attrs, list) or len(cls_attrs) != self.num_attr_types:
                    raise ValueError(
                        f"expected {self.num_attr_types} attrs per class; "
                        f"got len={len(cls_attrs) if isinstance(cls_attrs, list) else 'non-list'}"
                    )
                flat.extend(cls_attrs)

        # Use the parent's tokenize+encode path; treat all flattened strings
        # as a single 'batch' so the parent's reshape gives [1, B*C*A, D].
        encoded = super().forward([flat])  # [1, B*C*num_attr, D]
        attr_embeds = encoded.reshape(B, C, self.num_attr_types, self.embed_dim)

        return _fuse_attr_embeds(
            self, attr_embeds, self.num_attr_types, self.embed_dim
        )


@MODELS.register_module()
class PseudoHierarchicalXLMRLanguageBackbone(BaseModule):
    """Cache-backed hierarchical fusion backbone (training accelerator).

    Loads `attr_emb_cache_path` containing `dict[str, Tensor[D]]` for every
    attribute string the dataloader can emit. Forward path indexes the
    pre-stacked table and runs the same fusion module as
    `HierarchicalXLMRLanguageBackbone`. Bypasses the encoder entirely —
    only valid when the encoder is frozen (which is the Phase 3 design).
    """

    def __init__(
        self,
        *,
        attr_emb_cache_path: str,
        num_attr_types: int = 5,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual_alpha: float = 0.3,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        cache = torch.load(attr_emb_cache_path, map_location="cpu")
        if not isinstance(cache, dict):
            raise ValueError(
                f"attr cache must be dict[str, Tensor]; got {type(cache).__name__}"
            )
        if not cache:
            raise ValueError(f"attr cache at {attr_emb_cache_path} is empty.")

        keys = list(cache.keys())
        for k in keys:
            v = cache[k]
            if not isinstance(v, Tensor):
                raise ValueError(f"cache[{k!r}] is {type(v).__name__}, expected Tensor")
            if v.dim() != 1 or v.shape[0] != embed_dim:
                raise ValueError(
                    f"cache[{k!r}] shape {tuple(v.shape)} mismatch ({embed_dim},)"
                )

        table = torch.stack([cache[k].float() for k in keys], dim=0)  # [N, D]
        self.register_buffer("attr_emb_table", table, persistent=False)
        self._str_to_idx: Dict[str, int] = {k: i for i, k in enumerate(keys)}

        self.num_attr_types = num_attr_types
        self.embed_dim = embed_dim

        modules = _build_fusion_module(
            embed_dim, num_attr_types, num_heads, dropout, residual_alpha
        )
        self.attr_type_embed = modules["attr_type_embed"]
        self.fusion_query = modules["fusion_query"]
        self.cross_attn = modules["cross_attn"]
        self.norm1 = modules["norm1"]
        self.norm2 = modules["norm2"]
        self.output_proj = modules["output_proj"]
        self.alpha = modules["alpha"]
        _init_fusion_weights(self)

    def _lookup_indices(self, text: List[List[List[str]]]) -> Tensor:
        B = len(text)
        C = len(text[0])
        A = self.num_attr_types
        idx = torch.empty(B * C * A, dtype=torch.long)
        unknown: List[str] = []
        i = 0
        for tb in text:
            for ca in tb:
                if len(ca) != A:
                    raise ValueError(
                        f"expected {A} attrs per class; got len={len(ca)}"
                    )
                for s in ca:
                    if s in self._str_to_idx:
                        idx[i] = self._str_to_idx[s]
                    else:
                        unknown.append(s)
                    i += 1
        if unknown:
            sample = unknown[:3]
            raise KeyError(
                f"{len(unknown)} attribute string(s) missing from cache. "
                f"sample: {[s[:60] for s in sample]}"
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

        idx = self._lookup_indices(text).to(self.attr_emb_table.device)
        attr_embeds = self.attr_emb_table.index_select(0, idx).reshape(
            B, C, self.num_attr_types, self.embed_dim
        )

        return _fuse_attr_embeds(
            self, attr_embeds, self.num_attr_types, self.embed_dim
        )

    def train(self, mode: bool = True):  # noqa: D401 — match parent signature
        return super().train(mode)
