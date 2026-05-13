"""CytologySAVPE — spatial-mask visual prompt encoder for cytology detection.

Adapted from YOLOE's SAVPE (Semantic-Activated Visual Prompt Encoder)
at /home/25_liwenjie/code/yoloe/ultralytics/nn/modules/head.py.

Differences vs YOLOE:
  - ConvNext-tiny FPN channels [96, 192, 384] (vs YOLOv8s)
  - Output dim 512 (matches BiomedCLIP text emb, not 256/CLIP-default)
  - No segmentation branch — detection only
  - Same dual-branch (semantic + activation) + spatial softmax design

Forward:
    fpn_feats: List[3] of [B, C_i, H_i, W_i]  (ConvNext FPN)
    vp_masks:  [B, Q, H, W]  binary mask at stride-8 resolution
               (Q = number of visual prompts per image, e.g. 1-3 base classes)
    → class_vec: [B, Q, 512]  L2-normalized
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor


class _Conv(nn.Module):
    """Conv + BN + SiLU (matches YOLOE's Conv block topology)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


@MODELS.register_module()
class CytologySAVPE(BaseModule):
    """Spatial-mask visual prompt encoder.

    Two-branch design: a deep semantic branch produces high-dim features,
    a shallow activation branch produces low-dim spatial attention scores,
    and the binary visual prompt mask gates the attention. The aggregated
    features are then L2-normalized to match BiomedCLIP text embeddings.

    Args:
        in_channels: FPN output channels at 3 scales, e.g. [96, 192, 384]
            for ConvNext-tiny.
        intermediate_dim: branch hidden dim (typically 128).
        embed_dim: output embedding dim (must match text encoder, e.g.
            512 for BiomedCLIP).
        activation_dim: low-dim activation branch channels (typically 16).
            Larger = finer-grained spatial attention but more FLOPs.
    """

    def __init__(
        self,
        *,
        in_channels: List[int],
        intermediate_dim: int = 128,
        embed_dim: int = 512,
        activation_dim: int = 16,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if len(in_channels) != 3:
            raise ValueError(f"expected 3 FPN scales, got {len(in_channels)}")
        if embed_dim % activation_dim != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by activation_dim={activation_dim}"
            )

        self.activation_dim = activation_dim
        self.embed_dim = embed_dim

        # Semantic branch (deep, embed-dim) — 2 Convs per scale, upsample to s8
        self.sem = nn.ModuleList()
        for i, c in enumerate(in_channels):
            layers = [_Conv(c, intermediate_dim, 3), _Conv(intermediate_dim, intermediate_dim, 3)]
            if i == 1:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            elif i == 2:
                layers.append(nn.Upsample(scale_factor=4, mode="nearest"))
            self.sem.append(nn.Sequential(*layers))
        self.sem_proj = nn.Conv2d(3 * intermediate_dim, embed_dim, 1)

        # Activation branch (shallow, activation-dim) — 1 Conv per scale
        self.act = nn.ModuleList()
        for i, c in enumerate(in_channels):
            layers = [_Conv(c, intermediate_dim, 1)]
            if i == 1:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            elif i == 2:
                layers.append(nn.Upsample(scale_factor=4, mode="nearest"))
            self.act.append(nn.Sequential(*layers))
        self.act_proj = nn.Conv2d(3 * intermediate_dim, activation_dim, 3, padding=1)

        # VP mask encoder
        self.vp_enc = nn.Conv2d(1, activation_dim, 3, padding=1)

        # Attention head: combine activation map + VP encoding
        self.attn = nn.Sequential(
            _Conv(2 * activation_dim, activation_dim, 3),
            nn.Conv2d(activation_dim, activation_dim, 3, padding=1),
        )

    def _multi_scale_concat(self, fpn_feats: List[Tensor], branch: nn.ModuleList) -> Tensor:
        """Apply per-scale conv (with upsample) then concat along channels."""
        outs = [branch[i](fpn_feats[i]) for i in range(3)]
        # All outs are at stride-8 resolution by design (s1, s2 upsampled).
        return torch.cat(outs, dim=1)

    def forward(self, fpn_feats: List[Tensor], vp_masks: Tensor) -> Tensor:
        """
        Args:
            fpn_feats: 3 FPN levels [B, C_i, H_i, W_i] from ConvNext + neck.
                Stride-8 / 16 / 32 expected (we upsample 16 and 32 to match 8).
            vp_masks: [B, Q, H, W] binary mask, H/W at stride-8 resolution.

        Returns:
            class_vec: [B, Q, embed_dim] L2-normalized.
        """
        if len(fpn_feats) != 3:
            raise ValueError(f"expected 3 fpn levels, got {len(fpn_feats)}")
        if vp_masks.dim() != 4:
            raise ValueError(
                f"vp_masks must be [B, Q, H, W]; got shape {tuple(vp_masks.shape)}"
            )

        B = fpn_feats[0].shape[0]
        Q = vp_masks.shape[1]

        # Semantic features [B, embed_dim, H, W]
        sem = self._multi_scale_concat(fpn_feats, self.sem)
        sem = self.sem_proj(sem)
        Hs, Ws = sem.shape[2], sem.shape[3]

        if vp_masks.shape[2:] != (Hs, Ws):
            vp_masks = F.interpolate(vp_masks.float(), size=(Hs, Ws), mode="nearest")

        # Activation features [B, activation_dim, H, W]
        act = self._multi_scale_concat(fpn_feats, self.act)
        act = self.act_proj(act)

        # Per-prompt attention map
        act_q = (
            act.reshape(B, 1, self.activation_dim, Hs, Ws)
            .expand(-1, Q, -1, -1, -1)
            .reshape(B * Q, self.activation_dim, Hs, Ws)
        )
        vp_flat = vp_masks.reshape(B * Q, 1, Hs, Ws).float()
        vp_enc = self.vp_enc(vp_flat)  # [B*Q, activation_dim, H, W]
        score_map = self.attn(torch.cat((act_q, vp_enc), dim=1))  # [B*Q, activation_dim, H, W]
        score_map = score_map.reshape(B, Q, self.activation_dim, Hs * Ws)
        vp_flat = vp_flat.reshape(B, Q, 1, Hs * Ws)

        # Identify all-zero masks (no visual prompt available) → emit zero vector
        # to avoid softmax(-inf) → NaN. Caller is responsible for marking which
        # classes have valid visual prompts via the `vp_mask_valid` return.
        vp_any = (vp_flat.sum(dim=-1) > 0).float()  # [B, Q, 1]

        # Mask out positions where VP=0 by setting score to -inf, then softmax over spatial
        neg_inf = torch.finfo(score_map.dtype).min
        score_map = score_map * vp_flat + (1.0 - vp_flat) * neg_inf
        # For all-zero VP rows, replace with uniform so softmax is well-defined
        # (the output will be zeroed out via vp_any below).
        all_zero = (vp_flat.sum(dim=-1, keepdim=True) == 0).to(score_map.dtype)
        score_map = score_map * (1.0 - all_zero) + 0.0 * all_zero  # leave 0s on rows w/ no VP
        score_map = F.softmax(score_map.float(), dim=-1).to(sem.dtype)
        # score_map: [B, Q, activation_dim, H*W]

        # Aggregate semantic features by score, per activation channel.
        # YOLOE trick: transpose Q and activation_dim so the matmul broadcasts:
        #   score.transpose(-2,-3): [B, activation_dim, Q, H*W]
        #   sem_groups.transpose(-1,-2): [B, activation_dim, H*W, embed/activation_dim]
        #   matmul: [B, activation_dim, Q, embed/activation_dim]
        #   transpose back + reshape: [B, Q, embed_dim]
        sem_groups = sem.reshape(B, self.activation_dim, self.embed_dim // self.activation_dim, -1)
        aggregated = score_map.transpose(-2, -3) @ sem_groups.transpose(-1, -2)
        # aggregated: [B, activation_dim, Q, embed/activation_dim]
        out = aggregated.transpose(-2, -3).reshape(B, Q, self.embed_dim)
        out = F.normalize(out, dim=-1, p=2)
        # Zero out classes with no visual prompt (so caller knows to fall back to text-only)
        out = out * vp_any
        return out
