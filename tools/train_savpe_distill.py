#!/usr/bin/env python
"""SAVPE distillation training — Phase 5 Level 0.5.

Standalone training that takes a frozen WeDetect detector (image_model + neck)
and trains a CytologySAVPE module to encode bbox regions into the BiomedCLIP
text embedding space.

Distillation objective:
    For each training image, for each class c that has ≥1 GT bbox:
        text_emb_c = mean(BiomedCLIP per-attr embeddings of class c, A=5)  # frozen target
        mask_c     = union of class c's GT bboxes at stride-8 resolution
        vis_emb_c  = SAVPE(fpn_feats, mask_c)
        loss      += || vis_emb_c - text_emb_c ||²

Why this is a good Phase 5 starting point:
  1. **Alignment by construction**: SAVPE outputs live in BiomedCLIP text space
     by training objective — so vis and text class vectors are directly
     comparable at inference (unlike the current untrained visproto baseline
     where alignment is coincidental).
  2. **Frozen detector** = no risk of regressing base 25-cls mAP (0.321 stays).
     Only SAVPE (~1.56M params) trains.
  3. **Small target** = fast training (~2-3 epoch, ~3-4h on 1 GPU).
  4. **At inference**: replace `tools/build_visual_prototype.py` flow with
     "1 forward through SAVPE per visual prompt" — no need for 5-shot mean.

Usage:
    python tools/train_savpe_distill.py \
        --base-config config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py \
        --base-ckpt   work_dirs/.../noTHAF_2gpu/best_coco_bbox_mAP_epoch_11.pth \
        --attr-cache  data/texts/tct_ngc_attr_biomedclip_per_attr.pth \
        --train-attrs data/texts/tct_ngc_fullnames_30_attr_train.json \
        --train-ann   /home1/liwenjie/TCT_NGC_640/annotations/instances_train_dev_disjoint_dev30.json \
        --data-root   /home1/liwenjie/TCT_NGC_640/ \
        --img-prefix  images/ \
        --epochs 3 \
        --batch 4 \
        --lr 1e-3 \
        --out work_dirs/savpe_distill_v1/
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmdet.apis import init_detector

from wedetect.models.backbones.cytology_savpe import CytologySAVPE
from wedetect.utils import resolve_latest_checkpoint


# ───────────────────────────── data ────────────────────────────────────

class CytologySAVPEDataset(Dataset):
    """Yields (image_tensor, list of (class_idx, bbox_xywh)).

    Loads COCO-format annotations and indexes by image. For each training
    image, returns its raw BGR image (resized to input_size) and a list of
    (class_idx, bbox_xywh_in_resized_coords) for every GT bbox in the image.

    Class indices are stable across the dataset (cat_id_sorted determines
    the mapping into [0, num_classes)).
    """

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
        # BGR → RGB, float [0, 1], CHW
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

def build_per_class_masks(
    bbox_lists: List[List[Tuple[int, List[float]]]],
    num_classes: int,
    feat_H: int,
    feat_W: int,
    input_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each (batch_idx, class_idx), build a binary mask at feature
    resolution covering the union of that class's GT bboxes.

    Returns:
        vp_masks: [B, num_classes, feat_H, feat_W] float in {0, 1}
        valid:    [B, num_classes] float in {0, 1} (1 if class has ≥1 bbox)
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


# ───────────────────── text emb target (5-attr mean pool) ───────────────

def build_text_targets(
    attr_cache_path: str,
    train_attrs_json: str,
    embed_dim: int,
) -> torch.Tensor:
    """Build [num_classes, embed_dim] text target by mean-pooling the 5 attr
    embeddings of each class. L2-normalized to match SAVPE output."""
    cache: Dict[str, torch.Tensor] = torch.load(attr_cache_path, map_location="cpu")
    with open(train_attrs_json, "r", encoding="utf-8") as f:
        attrs_per_class: List[List[str]] = json.load(f)
    targets = []
    for cls_attrs in attrs_per_class:
        embs = []
        for s in cls_attrs:
            if s not in cache:
                raise KeyError(f"attr string not in cache: {s[:80]!r}")
            embs.append(cache[s])
        mean = torch.stack(embs, dim=0).mean(dim=0)
        mean = F.normalize(mean, dim=-1, p=2)
        if mean.shape[0] != embed_dim:
            raise ValueError(f"text emb dim {mean.shape[0]} != expected {embed_dim}")
        targets.append(mean)
    return torch.stack(targets, dim=0)


# ───────────────────────────── training ────────────────────────────────

@torch.no_grad()
def extract_fpn(model, imgs: torch.Tensor) -> List[torch.Tensor]:
    """image → ConvNext backbone.image_model → neck → 3 FPN levels."""
    img_feats = model.backbone.image_model(imgs)
    fpn = model.neck(img_feats)
    return list(fpn)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--attr-cache", required=True)
    p.add_argument("--train-attrs", required=True)
    p.add_argument("--train-ann", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--img-prefix", default="images/")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", required=True, help="output dir for SAVPE ckpt + log")
    p.add_argument("--seed", type=int, default=20260512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── 1. Load detector (frozen) ───────────────────────────────────────
    ckpt = resolve_latest_checkpoint(args.base_ckpt, "")
    print(f"[init] loading detector ckpt: {ckpt}")
    model = init_detector(args.base_config, ckpt, device=args.device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()  # freeze BN/dropout

    # ── 2. Resolve class list from train ann ────────────────────────────
    with open(args.train_ann, "r", encoding="utf-8") as f:
        ann = json.load(f)
    cat_id_sorted = sorted({c["id"] for c in ann["categories"]})
    cat_names = {c["id"]: c["name"] for c in ann["categories"]}
    num_classes = len(cat_id_sorted)
    print(f"[init] num_classes={num_classes}")
    print(f"[init] sample class names: {[cat_names[c] for c in cat_id_sorted[:3]]}")

    # ── 3. Build text targets ───────────────────────────────────────────
    text_targets = build_text_targets(args.attr_cache, args.train_attrs, args.embed_dim)
    text_targets = text_targets.to(args.device)
    assert text_targets.shape == (num_classes, args.embed_dim), (
        f"text_targets shape {tuple(text_targets.shape)} != ({num_classes}, {args.embed_dim})"
    )
    print(f"[init] text_targets shape: {tuple(text_targets.shape)}, "
          f"mean norm: {text_targets.norm(dim=-1).mean().item():.4f}")

    # ── 4. Detect FPN input channels by probing ─────────────────────────
    with torch.no_grad():
        probe = torch.zeros(1, 3, args.input_size, args.input_size, device=args.device)
        fpn = extract_fpn(model, probe)
    in_channels = [f.shape[1] for f in fpn]
    feat_H, feat_W = fpn[0].shape[2], fpn[0].shape[3]
    print(f"[init] FPN channels: {in_channels}, stride-8 feat: {feat_H}×{feat_W}")

    # ── 5. Build SAVPE ──────────────────────────────────────────────────
    savpe = CytologySAVPE(in_channels=in_channels, embed_dim=args.embed_dim).to(args.device)
    n_params = sum(p.numel() for p in savpe.parameters() if p.requires_grad)
    print(f"[init] SAVPE params: {n_params:,}")

    optimizer = torch.optim.AdamW(savpe.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── 6. Build dataloader ─────────────────────────────────────────────
    ds = CytologySAVPEDataset(
        ann_file=args.train_ann,
        data_root=args.data_root,
        img_prefix=args.img_prefix,
        cat_id_sorted=cat_id_sorted,
        input_size=args.input_size,
    )
    print(f"[init] dataset size: {len(ds)} images")
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    steps_per_epoch = len(loader)
    print(f"[init] steps/epoch: {steps_per_epoch}, total steps: {args.epochs * steps_per_epoch}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.01
    )

    # ── 7. Train ────────────────────────────────────────────────────────
    savpe.train()
    log_path = out_dir / "train.log"
    log_f = open(log_path, "w", encoding="utf-8")
    step = 0
    for epoch in range(args.epochs):
        loss_avg = 0.0
        loss_count = 0
        for imgs, bbox_lists in loader:
            imgs = imgs.to(args.device, non_blocking=True)
            with torch.no_grad():
                fpn = extract_fpn(model, imgs)
            masks, valid = build_per_class_masks(
                bbox_lists, num_classes, feat_H, feat_W, args.input_size
            )
            masks = masks.to(args.device)
            valid = valid.to(args.device)

            vis_emb = savpe(fpn, masks)  # [B, num_classes, embed_dim]
            target = text_targets.unsqueeze(0).expand_as(vis_emb)  # [B, num_classes, D]

            # Per-element MSE, masked to valid entries only
            mse = ((vis_emb - target) ** 2).sum(dim=-1)  # [B, num_classes]
            weighted = mse * valid
            n_valid = valid.sum().clamp(min=1.0)
            loss = weighted.sum() / n_valid

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(savpe.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            loss_avg += loss.item() * n_valid.item()
            loss_count += n_valid.item()
            step += 1

            if step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[ep {epoch+1}/{args.epochs}] step {step}/{args.epochs * steps_per_epoch}  "
                    f"loss={loss.item():.4f}  avg={loss_avg/max(loss_count,1):.4f}  "
                    f"n_valid_per_batch={valid.sum().item():.0f}  lr={lr_now:.2e}"
                )
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()

        # End of epoch summary
        epoch_msg = (
            f"=== epoch {epoch+1} done: avg loss = {loss_avg/max(loss_count,1):.4f} "
            f"({loss_count:.0f} valid class instances) ==="
        )
        print(epoch_msg)
        log_f.write(epoch_msg + "\n")

        # Save per-epoch ckpt
        ckpt_path = out_dir / f"savpe_ep{epoch+1}.pth"
        torch.save({
            "state_dict": savpe.state_dict(),
            "in_channels": in_channels,
            "embed_dim": args.embed_dim,
            "epoch": epoch + 1,
            "args": vars(args),
        }, ckpt_path)
        print(f"[save] {ckpt_path}")

    # Save final
    final_path = out_dir / "savpe_final.pth"
    torch.save({
        "state_dict": savpe.state_dict(),
        "in_channels": in_channels,
        "embed_dim": args.embed_dim,
        "epoch": args.epochs,
        "args": vars(args),
    }, final_path)
    print(f"[done] final → {final_path}")
    log_f.close()


if __name__ == "__main__":
    main()
