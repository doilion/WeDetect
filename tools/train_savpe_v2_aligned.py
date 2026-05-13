#!/usr/bin/env python
"""SAVPE-v2 training with cross-modal alignment — Phase 5e (Paper §A method).

Addresses 5 root causes from v1 cell-contrastive (which was NEGATIVE result,
strict avg 0.098 < inference-only baseline 0.105):

  Fix #1 (TOP PRIORITY): cross-modal alignment loss `L_align`
       Forces vis_emb to live in same direction as BiomedCLIP text_emb
       (which detection head was trained against). Without this, vis_emb
       and text_emb live in orthogonal cosine spaces (empirically:
       30-class base mean cos = -0.384, all negative).

  Fix #2: Multi-scale supervision
       Detection inference uses 3 FPN scales (stride 8/16/32). v1 only
       trained at stride-8. Now compute L_cell at all 3 scales.

  Fix #3: BN-aligned cell features (not L2-norm)
       v1 used `F.normalize(cls_preds[i](fpn[i]))` but detection inference
       uses `BNContrastiveHead.norm(cls_preds[i](fpn[i]))`. BN is critical.

  Fix #4: Use detection head's logit_scale + bias
       v1 used raw cosine * temp=10. Inference uses learnable
       logit_scale.exp() + bias. Now apply same transform.

  Fix #5: Cross-class contrastive
       Push vis_emb[c1] apart from vis_emb[c2] for c1≠c2. Plus implicit
       via L_align (text_emb is already cross-class separated).

Total loss:
    L = L_cell + λ_align * L_align + λ_cross * L_cross
    Default: λ_align=1.0 (most important), λ_cross=0.1

Frozen vs trainable:
  - Frozen: ConvNext image_encoder, neck, ALL of bbox_head
            (incl. cls_preds[i], cls_contrasts[i].norm, logit_scale, bias)
  - Trainable: SAVPE module only (~1.56M params)

Usage:
    torchrun --nproc_per_node=2 --master_port=29501 tools/train_savpe_v2_aligned.py \\
        --base-config config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py \\
        --base-ckpt work_dirs/.../noTHAF.../best_coco_bbox_mAP_epoch_11.pth \\
        --train-ann /home1/liwenjie/TCT_NGC_640/annotations/instances_train_dev_disjoint_dev30.json \\
        --data-root /home1/liwenjie/TCT_NGC_640/ \\
        --fullnames-json data/texts/tct_ngc_fullnames_30.json \\
        --text-cache data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth \\
        --lambda-align 1.0 --lambda-cross 0.1 \\
        --epochs 3 --batch 64 --workers 8 --lr 4e-3 \\
        --out work_dirs/savpe_v2_aligned_v1

Sanity-only mode (no training, just verify):
    python tools/train_savpe_v2_aligned.py --sanity-only ...

The --sanity-only flag runs:
  - Cache key mapping verification
  - BNContrastiveHead access
  - 3-scale FPN shape probe
  - L_align init value check
  - Single backward gradient flow check
  Then exits without training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mmdet.utils import register_all_modules

register_all_modules()

from mmdet.apis import init_detector

from wedetect.models.backbones.cytology_savpe import CytologySAVPE
from wedetect.utils import resolve_latest_checkpoint


# DDP env vars (set by torchrun)
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
IS_DDP = WORLD_SIZE > 1
IS_MAIN = RANK == 0


def log(msg: str) -> None:
    if IS_MAIN:
        print(msg, flush=True)


# ───────────────────────────── data ────────────────────────────────────

class CellContrastiveDataset(Dataset):
    """Same as v1: yields (image_tensor, list of (class_idx, bbox_xywh))."""

    def __init__(
        self,
        ann_file: str,
        data_root: str,
        img_prefix: str,
        cat_id_sorted: List[int],
        input_size: int = 640,
    ) -> None:
        self.input_size = input_size
        self.cat_to_idx = {cid: i for i, cid in enumerate(cat_id_sorted)}
        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)
        self.images_by_id = {im["id"]: im for im in ann["images"]}
        self.data_root = Path(data_root)
        self.img_prefix = img_prefix
        self.by_image: Dict[int, List[dict]] = {}
        for a in ann["annotations"]:
            if a["category_id"] not in self.cat_to_idx:
                continue
            self.by_image.setdefault(a["image_id"], []).append(a)
        self.image_ids = sorted(self.by_image.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[int, List[float]]]]:
        img_id = self.image_ids[idx]
        info = self.images_by_id[img_id]
        img_path = self.data_root / self.img_prefix / info["file_name"]
        if not img_path.is_file():
            img_path = self.data_root / info["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(img_path)
        H0, W0 = bgr.shape[:2]
        # Training data is already cached at 640×640, so cv2.resize is a no-op
        # for in-distribution train images. (Cache was built with proper
        # KeepRatio+Letter; we trust the cache geometry.)
        if (H0, W0) != (self.input_size, self.input_size):
            resized = cv2.resize(bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            scale_x = self.input_size / W0
            scale_y = self.input_size / H0
        else:
            resized = bgr
            scale_x = scale_y = 1.0
        bboxes = []
        for a in self.by_image[img_id]:
            x, y, w, h = a["bbox"]
            bx = max(0.0, min(self.input_size, x * scale_x))
            by = max(0.0, min(self.input_size, y * scale_y))
            bw = max(0.0, min(self.input_size - bx, w * scale_x))
            bh = max(0.0, min(self.input_size - by, h * scale_y))
            if bw <= 1 or bh <= 1:
                continue
            bboxes.append((self.cat_to_idx[a["category_id"]], [bx, by, bw, bh]))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return tensor, bboxes


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    bbox_lists = [b[1] for b in batch]
    return imgs, bbox_lists


# ─────────────────────────── mask building ──────────────────────────────

def build_class_masks_at_scale(
    bbox_lists: List[List[Tuple[int, List[float]]]],
    num_classes: int,
    feat_H: int,
    feat_W: int,
    input_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns:
        masks: [B, num_classes, feat_H, feat_W] binary
        valid: [B, num_classes] 1 if class c has ≥1 bbox in image b
    """
    B = len(bbox_lists)
    masks = torch.zeros(B, num_classes, feat_H, feat_W, dtype=torch.float32)
    valid = torch.zeros(B, num_classes, dtype=torch.float32)
    sx = feat_W / input_size
    sy = feat_H / input_size
    for b, bboxes in enumerate(bbox_lists):
        for cls_idx, (x, y, w, h) in bboxes:
            x1 = max(0, int(math.floor(x * sx)))
            y1 = max(0, int(math.floor(y * sy)))
            x2 = min(feat_W, int(math.ceil((x + w) * sx)))
            y2 = min(feat_H, int(math.ceil((y + h) * sy)))
            if x2 > x1 and y2 > y1:
                masks[b, cls_idx, y1:y2, x1:x2] = 1.0
                valid[b, cls_idx] = 1.0
    return masks, valid


# ─────────────────────────── loss helpers ──────────────────────────────

def focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def alignment_loss(
    vis_emb: torch.Tensor,   # [B, C, D] L2-normed (from SAVPE)
    text_emb: torch.Tensor,  # [C, D] L2-normed (frozen target)
    valid_mask: torch.Tensor,  # [B, C] in {0, 1}
) -> torch.Tensor:
    """L_align = mean over valid (b, c) of ||vis_emb[b, c] - text_emb[c]||²

    Only computed for classes that have GT presence in image (valid=1).
    """
    B, C, D = vis_emb.shape
    text_b = text_emb.unsqueeze(0).expand(B, -1, -1)  # [B, C, D]
    sq = ((vis_emb - text_b) ** 2).sum(dim=-1)        # [B, C]
    weighted = sq * valid_mask                        # mask out invalid
    n = valid_mask.sum().clamp(min=1.0)
    return weighted.sum() / n


def cross_class_contrastive(
    vis_emb: torch.Tensor,   # [B, C, D] L2-normed
    valid_mask: torch.Tensor,  # [B, C]
    tau: float = 0.1,
) -> torch.Tensor:
    """For each batch, compute pairwise cosines between valid classes.
    Off-diagonal cosines should be small. Penalize large positive off-diag.

    L_cross = mean over valid pairs of max(0, cos(vi, vj) - margin)²
    """
    B, C, D = vis_emb.shape
    # Cosines: [B, C, C]
    cos = torch.einsum('bcd,bjd->bcj', vis_emb, vis_emb)  # all pairs

    # Zero diagonal (don't penalize self-cos)
    eye = torch.eye(C, device=cos.device).unsqueeze(0)  # [1, C, C]
    off_diag = cos * (1.0 - eye)

    # Valid pair mask: both classes must be valid in image b
    valid_pair = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)  # [B, C, C]
    valid_pair = valid_pair * (1.0 - eye)  # exclude diagonal

    # Penalty: cosines > 0.2 are penalized
    margin = 0.2
    pen = F.relu(off_diag - margin) ** 2  # [B, C, C]
    pen = pen * valid_pair
    n_pairs = valid_pair.sum().clamp(min=1.0)
    return pen.sum() / n_pairs


# ───────────────────────── BNContrastiveHead 接入 ────────────────────────

def cell_logits_via_bn_head(
    model,
    fpn_feats: List[torch.Tensor],
    vis_emb: torch.Tensor,
) -> List[torch.Tensor]:
    """Mimic the detection head's exact scoring path:
      cell_feat_i = bbox_head.head_module.cls_preds[i](fpn[i])
      cell_logits_i = bbox_head.head_module.cls_contrasts[i](cell_feat_i, vis_emb)
        ↑ this applies BN + einsum + logit_scale + bias

    Returns: list of 3 score tensors [B, K, H_i, W_i] (one per FPN scale).
    """
    head_module = model.bbox_head.head_module
    out = []
    for i in range(3):
        cell_i = head_module.cls_preds[i](fpn_feats[i])  # [B, D, H, W]
        # cls_contrasts[i] is a BNContrastiveHead. Its forward takes (x, w):
        #   x = BN(cells), w = L2(vis_emb), score = einsum * logit_scale.exp() + bias
        score_i = head_module.cls_contrasts[i](cell_i, vis_emb)  # [B, K, H, W]
        out.append(score_i)
    return out


# ───────────────────────────── training ────────────────────────────────

@torch.no_grad()
def extract_fpn(model, imgs: torch.Tensor) -> List[torch.Tensor]:
    img_feats = model.backbone.image_model(imgs)
    fpn = model.neck(img_feats)
    return list(fpn)


def build_text_emb_per_class(
    fullnames_json: str,
    text_cache_path: str,
    cat_id_sorted: List[int],
    device: str,
) -> Tuple[torch.Tensor, List[str]]:
    """Map cat_id_sorted[idx] → fullnames_json[idx][0] → text_cache[primary_key].

    Per Fix #5 in plan: cache keys are prompt strings (long-form),
    not COCO category names. Use fullnames JSON's first variant as key.

    Returns:
        text_emb [num_classes, embed_dim] L2-normed, on device
        primary_keys (list of strings) for sanity logging
    """
    with open(fullnames_json) as f:
        attrs = json.load(f)
    if len(attrs) != len(cat_id_sorted):
        raise ValueError(
            f"fullnames JSON has {len(attrs)} classes but cat_id_sorted has "
            f"{len(cat_id_sorted)}. Order mismatch will silently corrupt training."
        )
    cache = torch.load(text_cache_path, map_location="cpu", weights_only=False)
    embs = []
    primary_keys = []
    for idx, cid in enumerate(cat_id_sorted):
        key = attrs[idx][0]
        if key not in cache:
            raise KeyError(
                f"primary key {key!r} for cat_id={cid} not in text cache "
                f"{text_cache_path}"
            )
        embs.append(cache[key])
        primary_keys.append(key)
    text_emb = torch.stack(embs, dim=0).float().to(device)  # [C, D]
    text_emb = F.normalize(text_emb, dim=-1, p=2)
    return text_emb, primary_keys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--train-ann", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--img-prefix", default="images/")
    p.add_argument("--fullnames-json", required=True,
                   help="Class fullnames JSON, e.g. data/texts/tct_ngc_fullnames_30.json")
    p.add_argument("--text-cache", required=True,
                   help="BiomedCLIP text emb cache, keyed by primary fullname")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--lambda-align", type=float, default=1.0)
    p.add_argument("--lambda-cross", type=float, default=0.1)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260512)
    p.add_argument("--sanity-only", action="store_true",
                   help="Run sanity checks (cache keys, BN access, gradient flow) then exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    if IS_MAIN:
        out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + RANK)
    np.random.seed(args.seed + RANK)
    random.seed(args.seed + RANK)

    # ── 0. DDP setup ────────────────────────────────────────────────────
    if IS_DDP:
        dist.init_process_group("nccl")
        torch.cuda.set_device(LOCAL_RANK)
        device = f"cuda:{LOCAL_RANK}"
        log(f"[ddp] WORLD_SIZE={WORLD_SIZE} RANK={RANK} LOCAL_RANK={LOCAL_RANK}")
    else:
        device = args.device

    # ── 1. Load detector (frozen) ───────────────────────────────────────
    ckpt = resolve_latest_checkpoint(args.base_ckpt, "")
    log(f"[init] loading detector ckpt: {ckpt}")
    model = init_detector(args.base_config, ckpt, device=device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # ── 2. Resolve class list ───────────────────────────────────────────
    with open(args.train_ann, "r", encoding="utf-8") as f:
        ann = json.load(f)
    cat_id_sorted = sorted({c["id"] for c in ann["categories"]})
    num_classes = len(cat_id_sorted)
    log(f"[init] num_classes={num_classes}")

    # ── 3. Build text_emb_per_class (Fix #5: primary key mapping) ──────
    text_emb_per_class, primary_keys = build_text_emb_per_class(
        args.fullnames_json, args.text_cache, cat_id_sorted, device
    )
    log(f"[init] text_emb shape: {tuple(text_emb_per_class.shape)}, "
        f"mean norm: {text_emb_per_class.norm(dim=-1).mean().item():.4f}")
    log(f"[init] sample primary keys: {primary_keys[:3]}")

    # ── 4. Probe FPN shape + BNContrastiveHead access ──────────────────
    with torch.no_grad():
        probe = torch.zeros(1, 3, args.input_size, args.input_size, device=device)
        fpn = extract_fpn(model, probe)
    in_channels = [f.shape[1] for f in fpn]
    feat_shapes = [(f.shape[2], f.shape[3]) for f in fpn]
    log(f"[init] FPN channels: {in_channels}")
    log(f"[init] FPN shapes: {feat_shapes}")

    head_module = model.bbox_head.head_module
    assert hasattr(head_module, "cls_preds"), "head_module missing cls_preds"
    assert hasattr(head_module, "cls_contrasts"), "head_module missing cls_contrasts"
    bn_logit_bias = []
    for i in range(3):
        bn = head_module.cls_contrasts[i].norm
        ls = head_module.cls_contrasts[i].logit_scale
        bs = head_module.cls_contrasts[i].bias
        bn_logit_bias.append((bn.__class__.__name__, ls.item(), bs.item()))
    log(f"[init] cls_contrasts[i] (BN type, logit_scale, bias) per scale: {bn_logit_bias}")

    # ── 5. Build SAVPE ──────────────────────────────────────────────────
    savpe = CytologySAVPE(in_channels=in_channels, embed_dim=args.embed_dim).to(device)
    n_params = sum(p.numel() for p in savpe.parameters() if p.requires_grad)
    log(f"[init] SAVPE params: {n_params:,}")

    if IS_DDP and not args.sanity_only:
        savpe = DDP(savpe, device_ids=[LOCAL_RANK], find_unused_parameters=False)
        savpe_module = savpe.module
    else:
        savpe_module = savpe

    optimizer = torch.optim.AdamW(savpe.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── 6. Build dataloader ─────────────────────────────────────────────
    ds = CellContrastiveDataset(
        ann_file=args.train_ann,
        data_root=args.data_root,
        img_prefix=args.img_prefix,
        cat_id_sorted=cat_id_sorted,
        input_size=args.input_size,
    )
    log(f"[init] dataset size: {len(ds)} images")

    if IS_DDP and not args.sanity_only:
        sampler = DistributedSampler(ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    steps_per_epoch = len(loader)
    log(f"[init] steps/epoch (per rank): {steps_per_epoch}, total per rank: {args.epochs * steps_per_epoch}")
    log(f"[init] effective batch: {args.batch * WORLD_SIZE}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.01
    )

    # ── 7. Sanity-only mode ─────────────────────────────────────────────
    if args.sanity_only:
        log("\n=== SANITY-ONLY MODE ===")

        # Sanity: one forward + backward to verify gradient flow
        savpe.train()
        imgs, bbox_lists = next(iter(loader))
        imgs = imgs.to(device)
        with torch.no_grad():
            fpn = extract_fpn(model, imgs)

        # Build masks at stride-8 (full resolution mask)
        B = imgs.shape[0]
        H_s8, W_s8 = feat_shapes[0]
        masks_s8, valid = build_class_masks_at_scale(
            bbox_lists, num_classes, H_s8, W_s8, args.input_size
        )
        masks_s8 = masks_s8.to(device)
        valid = valid.to(device)

        # Forward SAVPE on stride-8 mask
        vis_emb = savpe(fpn, masks_s8)  # [B, C, D]
        log(f"[sanity] vis_emb shape: {tuple(vis_emb.shape)}, "
            f"L2 norm sample: {vis_emb[0, 0].norm().item():.4f}")

        # L_align init value
        L_align = alignment_loss(vis_emb, text_emb_per_class, valid)
        log(f"[sanity] L_align (init, random SAVPE): {L_align.item():.4f}  "
            f"(expect ~2.0 for random vs unit sphere targets)")

        # L_cell multi-scale via BNContrastiveHead
        cell_scores = cell_logits_via_bn_head(model, fpn, vis_emb)
        log(f"[sanity] cell_scores shapes: {[tuple(s.shape) for s in cell_scores]}")
        L_cell = 0.0
        for i in range(3):
            H_i, W_i = cell_scores[i].shape[-2:]
            target_i = F.interpolate(masks_s8, size=(H_i, W_i), mode="nearest")
            L_cell_i = focal_bce(cell_scores[i], target_i, args.focal_alpha, args.focal_gamma)
            L_cell = L_cell + L_cell_i
        L_cell = L_cell / 3.0
        log(f"[sanity] L_cell (3 scales avg): {L_cell.item():.4f}")

        L_cross = cross_class_contrastive(vis_emb, valid)
        log(f"[sanity] L_cross (init): {L_cross.item():.4f}")

        L_total = L_cell + args.lambda_align * L_align + args.lambda_cross * L_cross
        log(f"[sanity] L_total = {L_total.item():.4f}")

        # Backward + check grads
        optimizer.zero_grad()
        L_total.backward()
        n_savpe_grad = sum(1 for p in savpe.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        n_savpe_total = sum(1 for p in savpe.parameters())
        n_det_grad = sum(1 for p in model.parameters() if p.grad is not None)
        log(f"[sanity] savpe params with grad: {n_savpe_grad}/{n_savpe_total}")
        log(f"[sanity] detector params with grad (should be 0): {n_det_grad}")

        assert n_savpe_grad == n_savpe_total, "SAVPE param missing grad"
        assert n_det_grad == 0, "detector should be frozen but has grads"
        log("\n✅ SANITY PASS — ready to launch full training")
        if IS_DDP:
            dist.destroy_process_group()
        return

    # ── 8. Full training loop ───────────────────────────────────────────
    savpe.train()
    log_f = None
    if IS_MAIN:
        log_path = out_dir / "train.log"
        log_f = open(log_path, "w", encoding="utf-8")

    H_s8, W_s8 = feat_shapes[0]
    step = 0
    for epoch in range(args.epochs):
        if IS_DDP:
            sampler.set_epoch(epoch)
        accum_total, accum_align, accum_cell, accum_cross = 0.0, 0.0, 0.0, 0.0
        accum_count = 0
        for imgs, bbox_lists in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                fpn = extract_fpn(model, imgs)

            B = imgs.shape[0]
            masks_s8, valid = build_class_masks_at_scale(
                bbox_lists, num_classes, H_s8, W_s8, args.input_size
            )
            masks_s8 = masks_s8.to(device)
            valid = valid.to(device)

            vis_emb = savpe(fpn, masks_s8)  # [B, C, D] L2-normed

            # Fix #1: alignment loss
            L_align = alignment_loss(vis_emb, text_emb_per_class, valid)

            # Fix #2-4: multi-scale BN-aligned cell-contrastive
            cell_scores = cell_logits_via_bn_head(model, fpn, vis_emb)
            L_cell = 0.0
            for i in range(3):
                H_i, W_i = cell_scores[i].shape[-2:]
                target_i = F.interpolate(masks_s8, size=(H_i, W_i), mode="nearest")
                L_cell_i = focal_bce(cell_scores[i], target_i, args.focal_alpha, args.focal_gamma)
                L_cell = L_cell + L_cell_i
            L_cell = L_cell / 3.0

            # Fix #5: cross-class contrastive
            L_cross = cross_class_contrastive(vis_emb, valid)

            L_total = L_cell + args.lambda_align * L_align + args.lambda_cross * L_cross

            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(savpe.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            accum_total += L_total.item()
            accum_align += L_align.item()
            accum_cell += L_cell.item()
            accum_cross += L_cross.item()
            accum_count += 1
            step += 1

            if step % args.log_every == 0 and IS_MAIN:
                lr_now = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[ep {epoch+1}/{args.epochs}] step {step}/{args.epochs * steps_per_epoch}  "
                    f"total={accum_total/accum_count:.4f}  "
                    f"align={accum_align/accum_count:.4f}  "
                    f"cell={accum_cell/accum_count:.4f}  "
                    f"cross={accum_cross/accum_count:.4f}  "
                    f"lr={lr_now:.2e}"
                )
                print(msg, flush=True)
                if log_f is not None:
                    log_f.write(msg + "\n")
                    log_f.flush()

        if IS_MAIN:
            epoch_msg = (
                f"=== epoch {epoch+1} done: "
                f"total avg={accum_total/accum_count:.4f}, "
                f"align={accum_align/accum_count:.4f}, "
                f"cell={accum_cell/accum_count:.4f}, "
                f"cross={accum_cross/accum_count:.4f} ==="
            )
            print(epoch_msg, flush=True)
            if log_f is not None:
                log_f.write(epoch_msg + "\n")

            ckpt_path = out_dir / f"savpe_ep{epoch+1}.pth"
            torch.save({
                "state_dict": savpe_module.state_dict(),
                "in_channels": in_channels,
                "embed_dim": args.embed_dim,
                "epoch": epoch + 1,
                "args": vars(args),
            }, ckpt_path)
            print(f"[save] {ckpt_path}", flush=True)

    if IS_MAIN:
        final_path = out_dir / "savpe_final.pth"
        torch.save({
            "state_dict": savpe_module.state_dict(),
            "in_channels": in_channels,
            "embed_dim": args.embed_dim,
            "epoch": args.epochs,
            "args": vars(args),
        }, final_path)
        print(f"[done] final → {final_path}", flush=True)
        if log_f is not None:
            log_f.close()

    if IS_DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
