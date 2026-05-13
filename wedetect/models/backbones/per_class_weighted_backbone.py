"""Option 3a — Per-class learnable attribute weighting backbones (no fusion).

Replaces THAF's cross-attention fusion module (which trains to alpha=0,
see `docs/figures/thaf_diagnostic/`) with a much smaller per-class
attribute weight matrix.

Forward formula:
    weights[cls_id, :]   # shape [A]  learnable scalars
    w = softmax(weights[cls_id, :])
    class_vec = sum_a (w[a] * attr_emb[cls, a])
    class_vec = L2_normalize(class_vec)

Parameter count: (num_known_classes + 1) × num_attr_types
  = (39 + 1) × 5 = 200 scalars  vs  THAF cross-attention 3.15M-7.1M

Class identification: each class is keyed by its diagnostic_code (the
2nd attribute string), which is unique per class in our 5-attr setup.
A learnable `_fallback_weights` row (last row of weights matrix) is used
for classes not seen at training time (i.e., novel zero-shot inference).

Two backbones mirror the THAF pair:

  1. PseudoPerClassWeightedXLMRLanguageBackbone — 768d, XLM-R cache
  2. PseudoPerClassWeightedBiomedCLIPLanguageBackbone — 512d, BiomedCLIP cache

Only the Pseudo variants are exposed: the encoder is frozen in all THAF
phases, so running the live encoder adds no signal — just compute cost.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor


def _load_class_keys(class_keys_json: str) -> List[str]:
    """Load list of unique class identifier strings.

    JSON format: a flat list of strings, one per known class. The string
    must match the value the dataloader will pass for that class in
    `text[b][c][diag_idx]` (typically diagnostic_code, the 2nd attribute).
    """
    with open(class_keys_json) as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
        raise ValueError(
            f"class_keys_json must be flat list[str], got {type(data).__name__}"
        )
    if len(set(data)) != len(data):
        dups = [s for s in set(data) if data.count(s) > 1]
        raise ValueError(f"duplicate class keys in {class_keys_json}: {dups[:3]}")
    return data


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


class _PCWBase(BaseModule):
    """Shared implementation for per-class weighted backbones.

    Subclasses set `embed_dim` and the registered name; everything else
    (cache loading, weight matrix, forward) is identical across encoders.
    """

    def __init__(
        self,
        *,
        attr_emb_cache_path: str,
        class_keys_json: str,
        embed_dim: int,
        num_attr_types: int = 5,
        class_key_attr_idx: int = 1,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        cache = torch.load(attr_emb_cache_path, map_location="cpu")
        keys = _validate_cache(cache, embed_dim, attr_emb_cache_path)
        table = torch.stack([cache[k].float() for k in keys], dim=0)
        self.register_buffer("attr_emb_table", table, persistent=False)
        self._str_to_idx: Dict[str, int] = {k: i for i, k in enumerate(keys)}

        class_keys = _load_class_keys(class_keys_json)
        self._cls_key_to_idx: Dict[str, int] = {k: i for i, k in enumerate(class_keys)}
        self.num_known_classes = len(class_keys)

        if not (0 <= class_key_attr_idx < num_attr_types):
            raise ValueError(
                f"class_key_attr_idx={class_key_attr_idx} must be in [0,{num_attr_types})"
            )
        self.class_key_attr_idx = class_key_attr_idx

        self.num_attr_types = num_attr_types
        self.embed_dim = embed_dim

        # +1 row for unknown / novel classes (softmax(0) = uniform 1/A)
        self.attr_weights = nn.Parameter(
            torch.zeros(self.num_known_classes + 1, num_attr_types)
        )
        self._fallback_idx = self.num_known_classes

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

    def _lookup_class_indices(self, text: List[List[List[str]]]) -> Tensor:
        B = len(text)
        C = len(text[0])
        cls_idx = torch.empty(B * C, dtype=torch.long)
        i = 0
        for tb in text:
            for ca in tb:
                key = ca[self.class_key_attr_idx]
                cls_idx[i] = self._cls_key_to_idx.get(key, self._fallback_idx)
                i += 1
        return cls_idx

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
            B, C, self.num_attr_types, self.embed_dim
        )

        cls_idx = self._lookup_class_indices(text).to(device)
        raw = self.attr_weights.index_select(0, cls_idx).reshape(
            B, C, self.num_attr_types
        )
        w = F.softmax(raw, dim=-1).unsqueeze(-1)  # [B, C, A, 1]

        class_vec = (w * attr_embeds).sum(dim=2)  # [B, C, D]
        return F.normalize(class_vec, dim=-1)

    def train(self, mode: bool = True):
        return super().train(mode)


@MODELS.register_module()
class PseudoPerClassWeightedXLMRLanguageBackbone(_PCWBase):
    """768-dim per-class weighted backbone (uses XLM-R per-attr cache)."""

    def __init__(
        self,
        *,
        attr_emb_cache_path: str,
        class_keys_json: str,
        num_attr_types: int = 5,
        embed_dim: int = 768,
        class_key_attr_idx: int = 1,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            attr_emb_cache_path=attr_emb_cache_path,
            class_keys_json=class_keys_json,
            embed_dim=embed_dim,
            num_attr_types=num_attr_types,
            class_key_attr_idx=class_key_attr_idx,
            init_cfg=init_cfg,
        )


@MODELS.register_module()
class PseudoPerClassWeightedBiomedCLIPLanguageBackbone(_PCWBase):
    """512-dim per-class weighted backbone (uses BiomedCLIP per-attr cache)."""

    def __init__(
        self,
        *,
        attr_emb_cache_path: str,
        class_keys_json: str,
        num_attr_types: int = 5,
        embed_dim: int = 512,
        class_key_attr_idx: int = 1,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            attr_emb_cache_path=attr_emb_cache_path,
            class_keys_json=class_keys_json,
            embed_dim=embed_dim,
            num_attr_types=num_attr_types,
            class_key_attr_idx=class_key_attr_idx,
            init_cfg=init_cfg,
        )
