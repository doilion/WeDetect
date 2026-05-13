#!/usr/bin/env python
"""Build visual prototype using TRAINED SAVPE module — Fix #3 corrected.

Bug fix vs v0 (kept as `tools/build_savpe_visproto.py.v0_bug_stretched`):
  v0 ran `cv2.resize(full_image, (640, 640))` which STRETCHES non-square
  test images. Bboxes were scaled by independent rx/ry ratios. This mismatched
  the detector's test pipeline (KeepRatioResize + LetterResize) and put
  SAVPE's mask in the wrong y-coords.

v1 (this file) uses mmengine Compose with the **real** test_pipeline transforms
  WeDetectKeepRatioResize → WeDetectLetterResize, exactly matching detector inference.
  Bboxes are transformed through `scale_factor` + `pad_param` from the same
  pipeline. Mask is then built at FPN stride-8 resolution from the
  letterbox-coordinate bbox.

Pipeline (per exemplar):
  1. LoadImageFromFile (full-res)
  2. WeDetectKeepRatioResize(scale=(640, 640))  → resize keeping aspect, no upscale
  3. WeDetectLetterResize(scale=(640, 640), allow_scale_up=False, pad_val=114)
     → pad to 640×640 (with `scale_factor` and `pad_param` keys)
  4. Convert bbox from original coords using scale_factor + pad offsets
  5. Build mask at stride-8 (80×80), forward through ConvNext + neck + SAVPE
  6. Mean over N exemplars per class, L2-normalize → save

Output schema unchanged from v0: dict[primary_text_key → tensor[embed_dim]].

Usage:
    python tools/build_savpe_visproto.py \
        --base-config config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py \
        --base-ckpt   work_dirs/.../noTHAF_2gpu/best_coco_bbox_mAP_epoch_11.pth \
        --savpe-ckpt  work_dirs/savpe_v2_aligned_v1/savpe_final.pth \
        --ann-file    annotations/instances_test_main_novel.json \
        --data-root   /home1/liwenjie/TCT_NGC/ \
        --text-json   data/texts/tct_ngc_novel_main_3.json \
        --out         data/texts/tct_ngc_novel_main_3_savpe_v2_aligned.pth \
        --n-per-class 5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector

from wedetect.models.backbones.cytology_savpe import CytologySAVPE
from wedetect.utils import resolve_latest_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--savpe-ckpt", required=True,
                   help="Trained SAVPE ckpt with {state_dict, in_channels, embed_dim}")
    p.add_argument("--ann-file", required=True,
                   help="abs path or relative to --data-root")
    p.add_argument("--data-root", default="/home1/liwenjie/TCT_NGC/")
    p.add_argument("--img-prefix", default="images/")
    p.add_argument("--text-json", required=True,
                   help="prompt JSON; first variant per class used as dict key")
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260509,
                   help="match build_visual_prototype.py seed for apples-to-apples")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--pad-val", type=int, default=114)
    return p.parse_args()


def build_test_preprocessing_pipeline(input_size: int, pad_val: int) -> Compose:
    """Compose the same KeepRatioResize + LetterResize the detector uses at test.

    Output dict carries:
      - 'img': padded 640×640 BGR uint8 array
      - 'img_shape': (640, 640)
      - 'scale_factor': (w_ratio, h_ratio) ≈ (ratio, ratio) since keep_ratio
      - 'pad_param': [top, bottom, left, right] padding in pixels
    """
    pipeline_cfg = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="WeDetectKeepRatioResize", scale=(input_size, input_size)),
        dict(
            type="WeDetectLetterResize",
            scale=(input_size, input_size),
            allow_scale_up=False,
            pad_val=dict(img=pad_val),
        ),
    ]
    return Compose(pipeline_cfg)


def transform_bbox_xywh(
    bbox_xywh: List[float],
    scale_factor: Tuple[float, float],
    pad_param: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Apply the same KeepRatio+Letter transform to an (x, y, w, h) bbox.

    scale_factor is (w_ratio, h_ratio) per WeDetectLetterResize._resize_img.
    pad_param is [top, bottom, left, right].
    """
    x, y, w, h = bbox_xywh
    w_r, h_r = float(scale_factor[0]), float(scale_factor[1])
    pad_top, _, pad_left, _ = (float(v) for v in pad_param)
    new_x = x * w_r + pad_left
    new_y = y * h_r + pad_top
    new_w = w * w_r
    new_h = h * h_r
    return new_x, new_y, new_w, new_h


def preprocess_image_via_pipeline(pipeline: Compose, img_path: Path) -> Tuple[torch.Tensor, Tuple[float, float], np.ndarray]:
    """Run the test pipeline on a single image. Returns:
        img_tensor: [1, 3, 640, 640] float32 in [0, 1] (RGB)
        scale_factor: (w_ratio, h_ratio)
        pad_param: [top, bot, left, right]
    """
    data = dict(img_path=str(img_path))
    results = pipeline(data)
    img = results["img"]  # 640x640 BGR uint8
    scale_factor = results["scale_factor"]
    # WeDetectLetterResize 不写 pad_param 到 dict (那是 PackDetInputs 之前的中间状态)
    # 重新计算 pad: shape - no_pad_shape
    # 但我们没保留 no_pad_shape；从 scale_factor 反推
    # no_pad_h = orig_h * h_r, no_pad_w = orig_w * w_r
    # 但 orig shape 也丢了。最稳：从 ori_shape (in results) 反推
    ori_shape = results.get("ori_shape", None)  # (H, W) before any resize
    if ori_shape is None:
        # 走 LoadImageFromFile, ori_shape 在 results
        raise RuntimeError("expected ori_shape in results from LoadImageFromFile")
    H0, W0 = ori_shape
    w_r, h_r = float(scale_factor[0]), float(scale_factor[1])
    no_pad_h = int(round(H0 * h_r))
    no_pad_w = int(round(W0 * w_r))
    target_h, target_w = img.shape[:2]
    padding_h = target_h - no_pad_h
    padding_w = target_w - no_pad_w
    pad_top = int(round(padding_h // 2 - 0.1)) if padding_h > 0 else 0
    pad_bot = padding_h - pad_top
    pad_left = int(round(padding_w // 2 - 0.1)) if padding_w > 0 else 0
    pad_right = padding_w - pad_left
    pad_param = np.array([pad_top, pad_bot, pad_left, pad_right], dtype=np.float32)

    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, scale_factor, pad_param


@torch.no_grad()
def extract_fpn(model, img_tensor: torch.Tensor) -> List[torch.Tensor]:
    img_feats = model.backbone.image_model(img_tensor)
    fpn = model.neck(img_feats)
    return list(fpn)


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # 1. Load detector (frozen)
    ckpt = resolve_latest_checkpoint(args.base_ckpt, "")
    print(f"[init] loading detector: {ckpt}")
    model = init_detector(args.base_config, ckpt, device=args.device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # 2. Load trained SAVPE
    savpe_blob = torch.load(args.savpe_ckpt, map_location=args.device, weights_only=False)
    in_channels = savpe_blob["in_channels"]
    embed_dim = savpe_blob["embed_dim"]
    print(f"[init] SAVPE ckpt: epoch={savpe_blob.get('epoch','?')}, "
          f"in_channels={in_channels}, embed_dim={embed_dim}")
    savpe = CytologySAVPE(in_channels=in_channels, embed_dim=embed_dim).to(args.device)
    savpe.load_state_dict(savpe_blob["state_dict"])
    savpe.eval()
    for p in savpe.parameters():
        p.requires_grad = False

    # 3. Build test preprocessing pipeline
    pipeline = build_test_preprocessing_pipeline(args.input_size, args.pad_val)
    print(f"[init] test preprocessing: KeepRatioResize + LetterResize (pad_val={args.pad_val})")

    # 4. Probe FPN dims (use letterboxed input)
    with torch.no_grad():
        probe = torch.zeros(1, 3, args.input_size, args.input_size, device=args.device)
        fpn = extract_fpn(model, probe)
    feat_H, feat_W = fpn[0].shape[2], fpn[0].shape[3]
    print(f"[init] FPN stride-8 feat: {feat_H}×{feat_W}")

    # 5. Load ann + class JSON
    ann_path = Path(args.ann_file)
    if not ann_path.is_absolute():
        ann_path = Path(args.data_root) / ann_path
    ann = json.loads(ann_path.read_text(encoding="utf-8"))

    cats = {c["id"]: c["name"] for c in ann["categories"]}
    images_by_id = {img["id"]: img for img in ann["images"]}
    by_cat: dict[int, list[dict]] = {cid: [] for cid in cats}
    for a in ann["annotations"]:
        by_cat[a["category_id"]].append(a)

    text_groups = json.loads(Path(args.text_json).read_text(encoding="utf-8"))
    if len(text_groups) != len(cats):
        raise SystemExit(
            f"prompt count {len(text_groups)} != ann categories {len(cats)}"
        )
    primary_keys = [grp[0] for grp in text_groups]
    cat_id_sorted = sorted(cats.keys())

    # 6. Build prototypes
    prototypes: dict[str, torch.Tensor] = {}
    used_anns: dict[str, list[int]] = {}
    feat_to_input = feat_H / args.input_size  # 80 / 640 = 1/8 (stride-8 mapping)

    for idx, cid in enumerate(cat_id_sorted):
        cname = cats[cid]
        primary = primary_keys[idx]
        anns = by_cat[cid]
        if not anns:
            print(f"[skip] cat_id={cid} {cname!r}: 0 anns")
            continue
        n = min(args.n_per_class, len(anns))
        chosen_idx = rng.choice(len(anns), size=n, replace=False)
        chosen = [anns[i] for i in chosen_idx]
        feats = []
        for a in chosen:
            img_info = images_by_id[a["image_id"]]
            file_name = img_info["file_name"]
            img_path = Path(args.data_root) / args.img_prefix / file_name
            if not img_path.is_file():
                img_path = Path(args.data_root) / file_name
                if not img_path.is_file():
                    img_path = ann_path.parent.parent / file_name

            try:
                img_tensor, scale_factor, pad_param = preprocess_image_via_pipeline(
                    pipeline, img_path
                )
            except FileNotFoundError as e:
                print(f"[warn] missing image {a['id']}: {e}")
                continue
            except Exception as e:
                print(f"[warn] pipeline failed on ann {a['id']}: {e}")
                continue
            img_tensor = img_tensor.to(args.device)

            with torch.no_grad():
                fpn = extract_fpn(model, img_tensor)

                # Transform bbox to letterboxed coord space, then to feature space
                lx, ly, lw, lh = transform_bbox_xywh(a["bbox"], scale_factor, pad_param)
                bx1 = max(0, int(math.floor(lx * feat_to_input)))
                by1 = max(0, int(math.floor(ly * feat_to_input)))
                bx2 = min(feat_W, int(math.ceil((lx + lw) * feat_to_input)))
                by2 = min(feat_H, int(math.ceil((ly + lh) * feat_to_input)))
                if bx2 <= bx1 or by2 <= by1:
                    print(f"[warn] degenerate bbox for ann {a['id']} after letterbox")
                    continue
                mask = torch.zeros(1, 1, feat_H, feat_W, device=args.device)
                mask[0, 0, by1:by2, bx1:bx2] = 1.0

                vis_emb = savpe(fpn, mask)  # [1, 1, embed_dim]
                vis_emb = vis_emb.squeeze(0).squeeze(0).cpu()  # [embed_dim]

            feats.append(vis_emb)

        if not feats:
            print(f"[skip] {cname!r}: all exemplars failed")
            continue
        proto = torch.stack(feats, dim=0).mean(dim=0).contiguous().float()
        proto = F.normalize(proto, dim=-1, p=2)
        prototypes[primary] = proto
        used_anns[primary] = [int(a["id"]) for a in chosen]
        print(f"[ok] cat_id={cid} {cname!r}: {len(feats)} exemplars → proto norm={proto.norm().item():.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prototypes, out_path)
    print(f"\n[done] {out_path} ({len(prototypes)} classes)")

    holdout_path = out_path.with_suffix(".holdout_anns.json")
    holdout_path.write_text(json.dumps(used_anns, indent=2), encoding="utf-8")
    print(f"[done] holdout: {holdout_path}")


if __name__ == "__main__":
    main()
