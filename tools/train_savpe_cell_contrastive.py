#!/usr/bin/env python
"""SAVPE Stage-2 training — cell-level contrastive (YOLOE-inspired).

Replaces the distillation objective (which capped SAVPE at text-encoder
ceiling) with a real visual discrimination objective. Approach is adapted
from YOLOE `train_vp.py` (their Stage-2):

  - Freeze everything except SAVPE.
  - For each image + sampled visual prompt mask of class c:
        vis_emb_c = SAVPE(fpn_feats, mask_c)                    # [B, 512]
        cell_feat = bbox_head.head_module.cls_preds[i](fpn_feats[i])
                                                                # [B, 512, H, W]
        score = cosine(cell_feat, vis_emb_c) at each cell        # [B, H, W]
        target = 1 where the cell lies inside class-c GT bbox, 0 elsewhere
        loss = focal_loss(sigmoid(score / temp), target)
  - Only SAVPE params get gradient.

Difference vs YOLOE: YOLOE uses full v8 detection loss (DFL+IoU+BCE+assigner).
We simplify to cell-level binary BCE because (a) the detector backbone is
frozen so box regression doesn't help train SAVPE, (b) implementing a real
mmdet detection assigner is mmdet-coupling we don't want to introduce.

If this trains the SAVPE module successfully (validated by novel zero-shot
mAP improvement over the inference-only visproto baseline 0.105), the
contribution is:
  - new state-of-the-art on TCT_NGC dev30 4 novel splits
  - SAVPE module structure validated for cytology
  - basis for further work (multi-modal alignment in full end-to-end Stage 3)

Usage:
    python tools/train_savpe_cell_contrastive.py \
        --base-config config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py \
        --base-ckpt   work_dirs/.../noTHAF_2gpu/best_coco_bbox_mAP_epoch_11.pth \
        --train-ann   /home1/liwenjie/TCT_NGC_640/annotations/instances_train_dev_disjoint_dev30.json \
        --data-root   /home1/liwenjie/TCT_NGC_640/ \
        --epochs 3 --batch 32 --workers 8 --lr 2e-3 \
        --out work_dirs/savpe_cellctr_v1/
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

from mmengine.config import Config
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
    """Print only on rank 0."""
    if IS_MAIN:
        print(msg, flush=True)


# ───────────────────────────── data ────────────────────────────────────

class CellContrastiveDataset(Dataset):
    """Yields (image_tensor, list of (class_idx, bbox_xywh_in_resized_coords))."""

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
        resized = cv2.resize(bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        scale_x = self.input_size / W0
        scale_y = self.input_size / H0
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


def collate_fn(
    batch: List[Tuple[torch.Tensor, List[Tuple[int, List[float]]]]],
) -> Tuple[torch.Tensor, List[List[Tuple[int, List[float]]]]]:
    imgs = torch.stack([b[0] for b in batch], dim=0)
    bbox_lists = [b[1] for b in batch]
    return imgs, bbox_lists


# ─────────────────────────── mask building ──────────────────────────────

def build_class_masks_and_present(
    bbox_lists: List[List[Tuple[int, List[float]]]],
    num_classes: int,
    feat_H: int,
    feat_W: int,
    input_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns:
        masks: [B, num_classes, feat_H, feat_W] binary, 1 where any GT bbox
               of that class covers the cell (union over instances)
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


# ───────────────────────── focal loss helper ────────────────────────────

def focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal BCE between cell logits and binary target. Both [N]."""
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


# ───────────────────────────── training ────────────────────────────────

@torch.no_grad()
def extract_fpn(model, imgs: torch.Tensor) -> List[torch.Tensor]:
    """image → ConvNext backbone.image_model → neck → 3 FPN levels."""
    img_feats = model.backbone.image_model(imgs)
    fpn = model.neck(img_feats)
    return list(fpn)


@torch.no_grad()
def project_cell_features(model, fpn_feats: List[torch.Tensor], scale_idx: int = 0) -> torch.Tensor:
    """Project FPN level i through bbox_head.head_module.cls_preds[i].

    This is the same projection used at detection inference time, so cell
    features live in the same 512d space as the class vectors that the
    contrastive head dots them with.

    Returns: [B, embed_dim, H, W] for the requested scale.
    """
    return model.bbox_head.head_module.cls_preds[scale_idx](fpn_feats[scale_idx])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--train-ann", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--img-prefix", default="images/")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--cos-temp", type=float, default=10.0,
                   help="cosine score temperature (logit = temp * cos)")
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    if IS_MAIN:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Per-rank seed differs by 1000 so each rank shuffles slightly differently
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

    # ── 1. Load detector (frozen) — each rank loads its own copy ────────
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

    # ── 3. Probe FPN shape ──────────────────────────────────────────────
    with torch.no_grad():
        probe = torch.zeros(1, 3, args.input_size, args.input_size, device=device)
        fpn = extract_fpn(model, probe)
        cell0 = project_cell_features(model, fpn, scale_idx=0)
    in_channels = [f.shape[1] for f in fpn]
    feat_H, feat_W = fpn[0].shape[2], fpn[0].shape[3]
    log(f"[init] FPN channels: {in_channels}, stride-8 feat: {feat_H}×{feat_W}")
    log(f"[init] cell feature shape at stride-8: {tuple(cell0.shape)}")
    assert cell0.shape[1] == args.embed_dim

    # ── 4. Build SAVPE (DDP-wrapped if multi-GPU) ───────────────────────
    savpe = CytologySAVPE(in_channels=in_channels, embed_dim=args.embed_dim).to(device)
    n_params = sum(p.numel() for p in savpe.parameters() if p.requires_grad)
    log(f"[init] SAVPE params: {n_params:,}")
    if IS_DDP:
        savpe = DDP(savpe, device_ids=[LOCAL_RANK], find_unused_parameters=False)
        savpe_module = savpe.module
    else:
        savpe_module = savpe

    optimizer = torch.optim.AdamW(savpe.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── 5. Build dataloader (DistributedSampler if DDP) ─────────────────
    ds = CellContrastiveDataset(
        ann_file=args.train_ann,
        data_root=args.data_root,
        img_prefix=args.img_prefix,
        cat_id_sorted=cat_id_sorted,
        input_size=args.input_size,
    )
    log(f"[init] dataset size: {len(ds)} images")

    if IS_DDP:
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
    log(f"[init] effective batch: {args.batch * WORLD_SIZE} (batch={args.batch} × world_size={WORLD_SIZE})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.01
    )

    # ── 6. Train ────────────────────────────────────────────────────────
    savpe.train()
    log_f = None
    if IS_MAIN:
        log_path = out_dir / "train.log"
        log_f = open(log_path, "w", encoding="utf-8")
    step = 0
    for epoch in range(args.epochs):
        if IS_DDP:
            sampler.set_epoch(epoch)
        loss_avg = 0.0
        loss_count = 0
        for imgs, bbox_lists in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                fpn = extract_fpn(model, imgs)
                cell = project_cell_features(model, fpn, scale_idx=0)  # [B, D, H, W]
                cell = F.normalize(cell, dim=1, p=2)  # L2 along channel

            B, D, H, W = cell.shape
            masks, valid = build_class_masks_and_present(
                bbox_lists, num_classes, H, W, args.input_size
            )
            masks = masks.to(device)  # [B, C, H, W]
            valid = valid.to(device)  # [B, C]

            vis_emb = savpe(fpn, masks)  # [B, C, D]
            # vis_emb is already L2-normed by SAVPE

            # score[b, c, n] = cos(cell[b, :, n], vis_emb[b, c, :])
            # Use batched matmul to avoid creating [B, C, D, HW] intermediate
            # vis_emb: [B, C, D],  cell.reshape: [B, D, HW]
            # @ → [B, C, HW]
            cell_flat = cell.reshape(B, D, -1)                # [B, D, HW]
            score = torch.bmm(vis_emb, cell_flat)             # [B, C, HW]
            score = score * args.cos_temp                     # temperature scale

            # Build target: [B, C, HW] from masks
            target = masks.reshape(B, num_classes, -1)        # [B, C, HW]

            # Only compute loss for classes present in image
            # Flatten then mask by valid
            score_flat = score.reshape(-1)                    # [B*C*HW]
            target_flat = target.reshape(-1)                  # [B*C*HW]
            valid_flat = valid.reshape(B, num_classes, 1).expand(-1, -1, H * W).reshape(-1)

            # Filter to valid class instances only
            mask_indices = valid_flat > 0
            if mask_indices.sum() == 0:
                continue
            score_v = score_flat[mask_indices]
            target_v = target_flat[mask_indices]

            loss = focal_bce(score_v, target_v, args.focal_alpha, args.focal_gamma)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(savpe.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            loss_avg += loss.item() * mask_indices.sum().item()
            loss_count += mask_indices.sum().item()
            step += 1

            if step % args.log_every == 0 and IS_MAIN:
                lr_now = optimizer.param_groups[0]["lr"]
                n_pos = target_v.sum().item()
                n_total = target_v.numel()
                msg = (
                    f"[ep {epoch+1}/{args.epochs}] step {step}/{args.epochs * steps_per_epoch}  "
                    f"loss={loss.item():.4f}  avg={loss_avg/max(loss_count,1):.4f}  "
                    f"valid_cls={valid.sum().item():.0f}/{B*num_classes}  "
                    f"pos_rate={n_pos/n_total:.4f}  lr={lr_now:.2e}"
                )
                print(msg)
                if log_f is not None:
                    log_f.write(msg + "\n")
                    log_f.flush()

        if IS_MAIN:
            epoch_msg = (
                f"=== epoch {epoch+1} done: avg loss = {loss_avg/max(loss_count,1):.4f} ==="
            )
            print(epoch_msg)
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
            print(f"[save] {ckpt_path}")

    if IS_MAIN:
        final_path = out_dir / "savpe_final.pth"
        torch.save({
            "state_dict": savpe_module.state_dict(),
            "in_channels": in_channels,
            "embed_dim": args.embed_dim,
            "epoch": args.epochs,
            "args": vars(args),
        }, final_path)
        print(f"[done] final → {final_path}")
        if log_f is not None:
            log_f.close()

    if IS_DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
