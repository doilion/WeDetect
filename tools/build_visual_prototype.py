#!/usr/bin/env python
"""Build visual exemplar class prototypes for novel zero-shot detection.

Replaces the text-derived class vector with the **mean image feature at the
GT bbox region** of N exemplar images, for each novel class. Works with
the existing PseudoLanguageBackbone cache_bank since the contrastive head
treats the class vector as a generic 768-dim embedding regardless of source.

Pipeline (per class):
  1. Sample N GT bboxes from the novel ann file (deterministic seed)
  2. Crop image to bbox + context, resize to 640x640
  3. Forward image through model.backbone.image_model → neck → head.cls_preds[i]
     (this is the same path that produces the 768-dim per-cell image features
     the contrastive head L2-norms and dots with text features)
  4. Spatial mean per FPN scale, then mean across scales → [768] vector
  5. Mean across N exemplars → class prototype
  6. Save as {primary_text → tensor} dict, drop-in replacement for the text emb cache

Output dict key = the same string `LoadText` will look up at inference (i.e. the
first variant in the novel JSON's inner list, same convention as text emb cache).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmdet.apis import init_detector

from wedetect.utils import resolve_latest_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--ann-file",
        required=True,
        help="abs or relative to --data-root, e.g. annotations/instances_test_main_novel.json",
    )
    p.add_argument("--data-root", default="/home1/liwenjie/TCT_NGC/")
    p.add_argument(
        "--img-prefix",
        default="images/",
        help="prepended to ann image file_name (matches dataset's data_prefix)",
    )
    p.add_argument(
        "--text-json",
        required=True,
        help="novel split prompt JSON; the FIRST variant in each inner list "
        "is used as the dict key (same convention as text emb cache)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260509)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--expand",
        type=float,
        default=1.5,
        help="bbox expansion ratio: how much context around the bbox to include",
    )
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="FPN scale indices to pool over; 0=stride8, 1=stride16, 2=stride32",
    )
    return p.parse_args()


def load_image_and_crop(
    img_path: Path,
    bbox_xywh: list[float],
    expand: float,
    target_size: int,
) -> np.ndarray:
    """Returns BGR uint8 [H, W, 3], same convention as cv2.imread."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    H, W = img.shape[:2]
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2, y + h / 2
    ew, eh = w * expand, h * expand
    x1 = max(0, int(cx - ew / 2))
    y1 = max(0, int(cy - eh / 2))
    x2 = min(W, int(cx + ew / 2))
    y2 = min(H, int(cy + eh / 2))
    if x2 <= x1 or y2 <= y1:
        # Degenerate bbox; fall back to original
        x1, y1, x2, y2 = 0, 0, W, H
    crop = img[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return crop_resized


def preprocess(bgr_img: np.ndarray, device: str) -> torch.Tensor:
    """Match YOLOWDetDataPreprocessor: bgr→rgb, /255, [B,3,H,W] float32."""
    rgb = bgr_img[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return tensor


@torch.no_grad()
def extract_image_embedding(model, img_tensor: torch.Tensor, scales: list[int]) -> torch.Tensor:
    """Forward img through backbone → neck → head.cls_preds[i] for each i in scales,
    spatial-mean-pool, then mean across scales → [embed_dim] (typically 768)."""
    img_feats = model.backbone.image_model(img_tensor)
    fpn_feats = model.neck(img_feats)
    head_module = model.bbox_head.head_module
    pooled = []
    for i in scales:
        proj = head_module.cls_preds[i](fpn_feats[i])  # [1, embed_dims, H, W]
        pooled.append(proj.mean(dim=(2, 3)).squeeze(0))  # [embed_dims]
    return torch.stack(pooled, dim=0).mean(dim=0)  # [embed_dims]


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    cfg = Config.fromfile(args.config)
    ckpt = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)
    print(f"[init] loading {ckpt}")
    model = init_detector(args.config, ckpt, device=args.device)
    model.eval()

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
            f"prompt count {len(text_groups)} != ann categories {len(cats)}; "
            f"text_json must have one entry per category in cat_id order."
        )
    primary_keys = [grp[0] for grp in text_groups]
    cat_id_sorted = sorted(cats.keys())

    prototypes: dict[str, torch.Tensor] = {}
    used_anns: dict[str, list[int]] = {}  # ann_ids used per class (for hold-out)
    for idx, cid in enumerate(cat_id_sorted):
        cname = cats[cid]
        primary = primary_keys[idx]
        anns = by_cat[cid]
        if not anns:
            print(f"[skip] cat_id={cid} {cname!r}: 0 annotations")
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
                # fallbacks
                img_path = Path(args.data_root) / file_name
                if not img_path.is_file():
                    img_path = ann_path.parent.parent / file_name
            try:
                bgr = load_image_and_crop(
                    img_path, a["bbox"], args.expand, args.input_size
                )
            except FileNotFoundError as e:
                print(f"[warn] missing image for ann {a['id']}: {e}")
                continue
            tensor = preprocess(bgr, args.device)
            emb = extract_image_embedding(model, tensor, args.scales).cpu()
            feats.append(emb)
        if not feats:
            print(f"[skip] {cname!r}: all exemplars failed")
            continue
        proto = torch.stack(feats, dim=0).mean(dim=0).contiguous().float()
        prototypes[primary] = proto
        used_anns[primary] = [int(a["id"]) for a in chosen]
        print(
            f"[ok] cat_id={cid} {cname!r}: {len(feats)} exemplars → "
            f"proto shape={tuple(proto.shape)}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prototypes, out_path)
    print(f"\nWrote {out_path} with {len(prototypes)} class prototypes")

    holdout_path = out_path.with_suffix(".holdout_anns.json")
    holdout_path.write_text(json.dumps(used_anns, indent=2), encoding="utf-8")
    print(
        f"Wrote {holdout_path} (ann_ids used as exemplars; exclude from eval to "
        f"avoid leakage if doing strict zero-shot reporting)"
    )


if __name__ == "__main__":
    main()
