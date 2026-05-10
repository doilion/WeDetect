"""BiomedCLIP-PubMedBERT hierarchical attribute fusion backbone for THAF (Phase 3b).

Adapted from /home/25_liwenjie/code/YOLO-World-Medical/yolo_world/models/
backbones/mm_backbone.py:3258-3497. Two changes vs the source:

  - num_attr_types defaults to 5 (we use 5 attr fields, not 4)
  - reuses fusion-module helpers from hierarchical_mm_backbone.py instead of
    duplicating them — the fusion topology is identical regardless of
    encoder family

Two backbones are exposed, mirroring the XLM-R pair:

  1. HierarchicalBiomedCLIPLanguageBackbone — full encoder + fusion. Loads
     BiomedCLIP via open_clip from HF Hub
     (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), freezes the
     encoder, and runs the cross-attention fusion on per-attribute encoder
     outputs. Output dim is 512 (BiomedCLIP's native).

  2. PseudoHierarchicalBiomedCLIPLanguageBackbone — accelerated training
     variant. Loads pre-cached `attr_emb_cache_path` and runs fusion only.
     Used during full training (encoder is frozen anyway, so re-encoding
     adds no signal — 5 × 80 strings per batch is wasted compute).

Phase 3b head must use embed_dims=512 (vs Phase 3a's 768). ContrastiveHead
is dim-agnostic, but the head_module.cls_preds Conv2d output channels must
match — see config wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu.py.

Token-length note: BiomedCLIP-PubMedBERT_256 limits sequences to 256 tokens.
Our 5-attr cells fit (max per-attr = 74, max concat-5 = 192 < 256).
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor

from .hierarchical_mm_backbone import (
    _build_fusion_module,
    _fuse_attr_embeds,
    _init_fusion_weights,
)


_BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _import_open_clip():
    try:
        import open_clip  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "open_clip_torch is not installed. Install with "
            "`pip install open_clip_torch>=2.23.0`."
        ) from e
    return open_clip


@MODELS.register_module()
class HierarchicalBiomedCLIPLanguageBackbone(BaseModule):
    """BiomedCLIP encoder + trainable hierarchical attribute fusion."""

    def __init__(
        self,
        *,
        model_name: str = _BIOMEDCLIP_HF,
        num_attr_types: int = 5,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual_alpha: float = 0.3,
        frozen_modules: Sequence[str] = ("all",),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        open_clip = _import_open_clip()
        self.model_name = model_name
        self.num_attr_types = num_attr_types
        self.embed_dim = embed_dim
        self.frozen_modules = frozen_modules

        self.model, _ = open_clip.create_model_from_pretrained(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._freeze_biomedclip()

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

    def _freeze_biomedclip(self) -> None:
        if not self.frozen_modules:
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            return
        # Partial freeze isn't supported here — BiomedCLIP via open_clip
        # exposes a single nn.Module; call sites should always use 'all'.
        raise ValueError(
            f"only frozen_modules=('all',) is supported for "
            f"HierarchicalBiomedCLIPLanguageBackbone; got {self.frozen_modules!r}"
        )

    def forward(self, text: List[List[List[str]]]) -> Tensor:
        if not isinstance(text, list) or not text:
            raise ValueError("text must be a non-empty batch list.")
        B = len(text)
        C = len(text[0])
        if C == 0:
            raise ValueError("each batch must contain at least one class.")
        for tb in text:
            if not isinstance(tb, list) or len(tb) != C:
                raise ValueError("all batch entries must have same num_classes.")

        flat: List[str] = []
        for tb in text:
            for cls_attrs in tb:
                if not isinstance(cls_attrs, list) or len(cls_attrs) != self.num_attr_types:
                    raise ValueError(
                        f"expected {self.num_attr_types} attrs per class; "
                        f"got len={len(cls_attrs) if isinstance(cls_attrs, list) else 'non-list'}"
                    )
                flat.extend(cls_attrs)

        tokens = self.tokenizer(flat).to(next(self.model.parameters()).device)
        with torch.no_grad():
            attr_embeds = self.model.encode_text(tokens)  # [B*C*A, D]
            attr_embeds = attr_embeds / attr_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        attr_embeds = attr_embeds.reshape(B, C, self.num_attr_types, self.embed_dim)

        return _fuse_attr_embeds(
            self, attr_embeds, self.num_attr_types, self.embed_dim
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep encoder in eval mode regardless of training flag (frozen).
        if "all" in self.frozen_modules:
            self.model.eval()
        return self


@MODELS.register_module()
class PseudoHierarchicalBiomedCLIPLanguageBackbone(BaseModule):
    """Cache-backed BiomedCLIP fusion backbone (training accelerator).

    Mirrors PseudoHierarchicalXLMRLanguageBackbone but at embed_dim=512.
    Pre-cache must be built with `tools/build_per_attr_emb_cache.py
    --encoder biomedclip` so the cache values are 512-dim BiomedCLIP outputs.
    """

    def __init__(
        self,
        *,
        attr_emb_cache_path: str,
        num_attr_types: int = 5,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual_alpha: float = 0.3,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        cache = torch.load(attr_emb_cache_path, map_location="cpu")
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
                    f"cache[{k!r}] shape {tuple(v.shape)} mismatch ({embed_dim},)"
                )
        table = torch.stack([cache[k].float() for k in keys], dim=0)
        self.register_buffer("attr_emb_table", table, persistent=False)
        self._str_to_idx = {k: i for i, k in enumerate(keys)}

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

        idx = self._lookup_indices(text).to(self.attr_emb_table.device)
        attr_embeds = self.attr_emb_table.index_select(0, idx).reshape(
            B, C, self.num_attr_types, self.embed_dim
        )

        return _fuse_attr_embeds(
            self, attr_embeds, self.num_attr_types, self.embed_dim
        )
