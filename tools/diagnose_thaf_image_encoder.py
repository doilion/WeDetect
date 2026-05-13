#!/usr/bin/env python
"""Phase 3.6 diagnostic — verify Hypothesis B (image encoder OOD on novel).

Phase 3.5 refuted Hypothesis A (fusion module not collapsing novel class
vectors). The remaining suspect for THAF novel zero-shot collapse is the
image encoder side: maybe novel test images produce image features that
align poorly with novel class vectors (potentially aligning toward base
class vectors instead).

This tool:
  1. Loads the trained THAF model (full YOLOWorldDetector, not just text branch)
  2. Computes the 39 fused class vectors (30 base + 9 novel) via the trained
     fusion module — same approach as Phase 3.5
  3. For a sample of test images (base + novel), forwards through image
     encoder + neck + head_module.cls_preds. Hooks into cls_preds outputs
     to capture per-pixel image features at the 3 FPN scales.
  4. For each GT bbox, extracts the cls_embed feature at the bbox center
     pixel (at each scale, averaged) and L2-normalizes.
  5. Computes cosine of this image feature with all 39 class vectors.
  6. Records:
     - top-1 predicted class (from the 39 candidates)
     - cosine to GT class vector
     - "is GT class predicted as top-1?"
     - "is top-1 a base class when GT is novel?"
  7. Outputs JSON summary + visualization showing the alignment pattern.

Decision tree:
  - Novel images' top-1 mostly a BASE class → image encoder pulls novel
    images toward base class space (B confirmed: image-side OOD failure)
  - Novel images' top-1 = correct novel class but low cosine → image
    encoder produces low-confidence features (B partial: image encoder
    doesn't strongly project, but at least it can disambiguate)
  - Novel images' top-1 = correct novel class with high cosine → there
    must be ANOTHER issue (probably eval pipeline / cache mismatch / E)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmengine.registry import MODELS

from PIL import Image
import torchvision.transforms.functional as TF


ATTR_FIELDS = (
    "organ_specimen",
    "diagnostic_code",
    "cytomorphology",
    "background_and_immunoprofile",
    "key_distinguishing_feature",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="THAF config")
    p.add_argument("--checkpoint", required=True, help="THAF best ckpt")
    p.add_argument("--attr-json", default="data/texts/tct_ngc_fullnames_39_attr.json")
    p.add_argument("--base-json", default="data/texts/tct_ngc_attr_base30.json")
    p.add_argument("--novel-json", default="data/texts/tct_ngc_attr_novel9.json")
    # which test ann to use (we sample images from both base + novel)
    p.add_argument(
        "--base-test-ann",
        default="/home1/liwenjie/TCT_NGC/annotations/instances_test_base_clean_dev30.json",
    )
    p.add_argument(
        "--novel-test-anns",
        nargs="+",
        default=[
            "/home1/liwenjie/TCT_NGC/annotations/instances_test_main_novel.json",
            "/home1/liwenjie/TCT_NGC/annotations/instances_test_pseudo_novel.json",
            "/home1/liwenjie/TCT_NGC/annotations/instances_hard_test.json",
        ],
    )
    p.add_argument(
        "--img-root",
        default="/home1/liwenjie/TCT_NGC/images/",
        help="root path for images referenced by ann files",
    )
    p.add_argument("--n-per-class", type=int, default=30,
                   help="sample N GT bboxes per class")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260511)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────
# Step A: compute 39 fused class vectors (same as Phase 3.5)
# ────────────────────────────────────────────────────────────────────
def compute_class_vectors(
    cfg: Config,
    ckpt_path: str,
    class_names: List[str],
    attr_dict: dict,
    device: str,
) -> torch.Tensor:
    """Load PseudoHierarchical backbone weights from THAF ckpt and forward
    to get 39 fused class vectors. Mirrors tools/diagnose_thaf_fusion.py."""
    text_cfg = cfg.model.backbone.text_model.copy()
    backbone = MODELS.build(text_cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)
    prefix = "backbone.text_model."
    text_state = {
        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    backbone.load_state_dict(text_state, strict=False)
    backbone.to(device).eval()

    text = [[[attr_dict[c][f].strip() for f in ATTR_FIELDS] for c in class_names]]
    with torch.no_grad():
        fused = backbone(text).squeeze(0).to(device)  # [39, D]
    fused = F.normalize(fused, dim=-1)
    return fused


# ────────────────────────────────────────────────────────────────────
# Step B: build full THAF detector + register hooks on cls_preds
# ────────────────────────────────────────────────────────────────────
def build_full_detector(cfg: Config, ckpt_path: str, device: str) -> torch.nn.Module:
    """Build YOLOWorldDetector with THAF text branch and load weights."""
    model_cfg = cfg.model.copy()
    # Don't need full text encoder during inference (we pre-compute class vecs);
    # just keep the structure so weights load. Use the Pseudo* backbone which
    # is already in the config.
    model = MODELS.build(model_cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)
    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat.unexpected_keys:
        print(f"[warn] unexpected keys: {sorted(incompat.unexpected_keys)[:3]}...")
    model.to(device).eval()
    return model


# ────────────────────────────────────────────────────────────────────
# Step C: forward image, extract cls_embed at GT bbox center
# ────────────────────────────────────────────────────────────────────
def load_image_letterboxed(
    img_path: str, target_size: int = 640
) -> tuple[torch.Tensor, float, tuple[int, int]]:
    """Load image, letterbox-resize to (target_size, target_size). Returns
    tensor (3, H, W) normalized [0,1], scale factor (orig->resized), and
    (pad_left, pad_top)."""
    img = Image.open(img_path).convert("RGB")
    w0, h0 = img.size
    scale = target_size / max(h0, w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top
    # letterbox pad with 114
    arr = np.array(img, dtype=np.uint8)
    arr = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=114,
    )
    tensor = TF.to_tensor(arr)  # (3, 640, 640) in [0, 1]
    return tensor, scale, (pad_left, pad_top)


def gt_box_center_in_feature_map(
    bbox_orig: list[float],
    scale: float,
    pad: tuple[int, int],
    feat_size: int,
    target_size: int = 640,
) -> tuple[int, int]:
    """Map a GT bbox [x, y, w, h] in original image coords to (y, x) center
    pixel index in a feature map of size feat_size × feat_size."""
    x, y, w, h = bbox_orig
    cx_orig = x + w / 2
    cy_orig = y + h / 2
    cx_letterboxed = cx_orig * scale + pad[0]
    cy_letterboxed = cy_orig * scale + pad[1]
    # downsample factor = target_size / feat_size
    stride = target_size / feat_size
    fx = int(cx_letterboxed / stride)
    fy = int(cy_letterboxed / stride)
    fx = max(0, min(feat_size - 1, fx))
    fy = max(0, min(feat_size - 1, fy))
    return fy, fx


def extract_image_feature_at_gt(
    detector: torch.nn.Module,
    image_tensor: torch.Tensor,
    gt_boxes_orig: list[list[float]],
    gt_classes: list[int],
    scale_factor: float,
    pad: tuple[int, int],
    target_size: int = 640,
) -> tuple[torch.Tensor, list[int]]:
    """Forward image through backbone + neck + cls_preds; for each GT bbox,
    extract cls_embed feature at bbox center (avg over 3 scales).

    Returns:
        features: [N_gt, D] L2-normalized features
        labels: [N_gt] list of GT class ids
    """
    device = next(detector.parameters()).device
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, 640, 640]

    with torch.no_grad():
        # backbone image side returns image multi-scale features
        img_feats = detector.backbone.forward_image(image_tensor)
        # neck
        img_feats = detector.neck(img_feats)
        # cls_preds per scale → cls_embed in text dim
        cls_preds = detector.bbox_head.head_module.cls_preds
        # img_feats is tuple of [B, C, H_i, W_i] at 3 scales

    # For each GT, extract features at all 3 scales (center pixel) and average
    features = []
    labels = []
    for bbox, cls_id in zip(gt_boxes_orig, gt_classes):
        per_scale = []
        for i, (feat, cls_pred) in enumerate(zip(img_feats, cls_preds)):
            with torch.no_grad():
                cls_embed = cls_pred(feat)  # [B, D, H_i, W_i]
            _, _, H_i, W_i = cls_embed.shape
            fy, fx = gt_box_center_in_feature_map(
                bbox, scale_factor, pad, H_i, target_size
            )
            v = cls_embed[0, :, fy, fx]  # [D]
            per_scale.append(v)
        avg_feat = torch.stack(per_scale, dim=0).mean(dim=0)  # [D]
        features.append(F.normalize(avg_feat, dim=-1).cpu())
        labels.append(cls_id)
    return torch.stack(features, dim=0), labels


# ────────────────────────────────────────────────────────────────────
# Step D: sample test data (base + novel)
# ────────────────────────────────────────────────────────────────────
def sample_gt_instances(
    ann_path: str,
    img_root: str,
    n_per_class: int,
    seed: int,
) -> List[dict]:
    """Sample n_per_class GT bboxes per category from a COCO ann file.

    Returns list of dicts: {img_path, bbox, gt_class_name, ann_id}
    """
    with open(ann_path, "r") as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    img_id_to_name = {im["id"]: im["file_name"] for im in data["images"]}
    by_cat = {cid: [] for cid in cats}
    for a in data["annotations"]:
        by_cat[a["category_id"]].append(a)

    rng = random.Random(seed)
    out = []
    for cid, annlist in by_cat.items():
        sampled = rng.sample(annlist, min(n_per_class, len(annlist)))
        for a in sampled:
            out.append(
                dict(
                    img_path=str(Path(img_root) / img_id_to_name[a["image_id"]]),
                    bbox=a["bbox"],  # [x, y, w, h]
                    gt_class_name=cats[cid],
                    ann_id=a["id"],
                )
            )
    return out


# ────────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Class order: 30 base + 9 novel
    base_names = [grp[0] for grp in json.loads(Path(args.base_json).read_text())]
    novel_names = [grp[0] for grp in json.loads(Path(args.novel_json).read_text())]
    all_names = base_names + novel_names
    n_base, n_novel = len(base_names), len(novel_names)
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    print(f"[setup] {n_base} base + {n_novel} novel = {len(all_names)} classes")

    # Load attr dict
    attr = json.loads(Path(args.attr_json).read_text(encoding="utf-8"))

    # Build cfg + class vectors
    cfg = Config.fromfile(args.config)
    print("[step A] computing 39 fused class vectors...")
    class_vecs = compute_class_vectors(
        cfg, args.checkpoint, all_names, attr, args.device
    )  # [39, D] on device
    embed_dim = class_vecs.shape[1]
    encoder_name = (
        cfg.model.backbone.text_model.type.replace("PseudoHierarchical", "")
        .replace("LanguageBackbone", "")
        .lower()
    )
    print(f"[step A] done ({encoder_name} {embed_dim}d, {class_vecs.shape})")

    # Build full detector
    print("[step B] building full detector + loading ckpt...")
    detector = build_full_detector(cfg, args.checkpoint, args.device)

    # Sample GT instances
    print(f"[step D] sampling {args.n_per_class} GT/class from base + 3 novel anns...")
    base_samples = sample_gt_instances(
        args.base_test_ann, args.img_root, args.n_per_class, args.seed
    )
    novel_samples: List[dict] = []
    for nann in args.novel_test_anns:
        novel_samples.extend(
            sample_gt_instances(nann, args.img_root, args.n_per_class, args.seed)
        )
    # dedupe novel samples by ann_id (some classes appear in multiple split anns)
    seen = set()
    deduped_novel = []
    for s in novel_samples:
        if s["ann_id"] not in seen:
            seen.add(s["ann_id"])
            deduped_novel.append(s)
    print(
        f"[step D] sampled {len(base_samples)} base GT + "
        f"{len(deduped_novel)} novel GT (deduped from {len(novel_samples)})"
    )

    # Forward + extract features
    all_records = []  # (split: base|novel, gt_class_name, top1_class, top1_cos, gt_cos)
    for tag, samples in [("base", base_samples), ("novel", deduped_novel)]:
        for i, s in enumerate(samples):
            try:
                img_tensor, scale, pad = load_image_letterboxed(s["img_path"])
            except (FileNotFoundError, OSError) as e:
                continue
            cls_idx = name_to_idx.get(s["gt_class_name"])
            if cls_idx is None:
                continue
            features, _ = extract_image_feature_at_gt(
                detector,
                img_tensor,
                [s["bbox"]],
                [cls_idx],
                scale,
                pad,
            )
            features = features.to(class_vecs.device)
            cos = (features @ class_vecs.T).squeeze(0).cpu()  # [39]
            top1 = int(cos.argmax())
            all_records.append(
                dict(
                    split=tag,
                    gt_class=s["gt_class_name"],
                    gt_idx=cls_idx,
                    gt_cos=float(cos[cls_idx]),
                    top1_class=all_names[top1],
                    top1_idx=top1,
                    top1_cos=float(cos[top1]),
                    top1_is_base=(top1 < n_base),
                    cos_all=cos.tolist(),
                )
            )
            if (i + 1) % 50 == 0:
                print(f"[{tag}] processed {i+1}/{len(samples)}")
        print(f"[{tag}] total {sum(1 for r in all_records if r['split']==tag)}")

    # ────────────────────────────────────────────────────────────────
    # Aggregate stats
    # ────────────────────────────────────────────────────────────────
    base_recs = [r for r in all_records if r["split"] == "base"]
    novel_recs = [r for r in all_records if r["split"] == "novel"]

    def agg(recs: List[dict]) -> dict:
        if not recs:
            return {}
        gt_cos = np.array([r["gt_cos"] for r in recs])
        top1_cos = np.array([r["top1_cos"] for r in recs])
        correct = sum(1 for r in recs if r["top1_idx"] == r["gt_idx"])
        top1_is_base = sum(1 for r in recs if r["top1_is_base"])
        return dict(
            n=len(recs),
            gt_cos_mean=float(gt_cos.mean()),
            gt_cos_std=float(gt_cos.std()),
            gt_cos_max=float(gt_cos.max()),
            gt_cos_min=float(gt_cos.min()),
            top1_cos_mean=float(top1_cos.mean()),
            top1_acc=correct / len(recs),
            top1_is_base_rate=top1_is_base / len(recs),
        )

    base_agg = agg(base_recs)
    novel_agg = agg(novel_recs)

    # Per-class GT cos for novel
    per_novel_class = {}
    for cls in novel_names:
        cls_recs = [r for r in novel_recs if r["gt_class"] == cls]
        if cls_recs:
            per_novel_class[cls] = agg(cls_recs)

    summary = dict(
        encoder=encoder_name,
        embed_dim=embed_dim,
        checkpoint=args.checkpoint,
        n_per_class=args.n_per_class,
        n_base=n_base,
        n_novel=n_novel,
        base=base_agg,
        novel=novel_agg,
        per_novel_class=per_novel_class,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[json] wrote {out_dir/'summary.json'}")

    # ────────────────────────────────────────────────────────────────
    # Decision tree + markdown
    # ────────────────────────────────────────────────────────────────
    hits = []
    if novel_agg.get("top1_is_base_rate", 0) > 0.5:
        hits.append(
            f"**B confirmed (image encoder pulls novel toward base)**: "
            f"{novel_agg['top1_is_base_rate']*100:.1f}% of novel images' top-1 "
            f"prediction is a BASE class (out of {n_base + n_novel} candidates), "
            f"despite the GT being novel."
        )
    if novel_agg.get("gt_cos_mean", 1) < base_agg.get("gt_cos_mean", 0):
        diff = base_agg["gt_cos_mean"] - novel_agg["gt_cos_mean"]
        hits.append(
            f"**B partial: novel image features less aligned with GT class** "
            f"(novel gt-cos mean {novel_agg['gt_cos_mean']:.3f} < base "
            f"gt-cos mean {base_agg['gt_cos_mean']:.3f}, Δ={diff:+.3f})"
        )
    if novel_agg.get("top1_acc", 0) > 0.5:
        hits.append(
            f"Note: novel top-1 accuracy = {novel_agg['top1_acc']*100:.1f}% — "
            f"even when image encoder picks correct novel class, mAP collapse "
            f"may stem from detection-side issues (low confidence, NMS) rather "
            f"than image-encoder alignment."
        )
    if not hits:
        hits.append("Inconclusive — see per-class breakdown.")

    md = [
        f"# Phase 3.6 image encoder diagnostic — {encoder_name} ({embed_dim}d)",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Sampled GT bboxes: {base_agg.get('n', 0)} base + "
        f"{novel_agg.get('n', 0)} novel (target n_per_class={args.n_per_class})",
        "",
        "## Aggregate stats (cosine between image feature at GT bbox center and class vectors)",
        "",
        f"| metric | base GT bboxes | novel GT bboxes |",
        f"|---|---:|---:|",
        f"| n | {base_agg.get('n','?')} | {novel_agg.get('n','?')} |",
        f"| mean cosine to GT class | {base_agg.get('gt_cos_mean',0):.3f} | "
        f"{novel_agg.get('gt_cos_mean',0):.3f} |",
        f"| mean cosine to top-1 class | {base_agg.get('top1_cos_mean',0):.3f} | "
        f"{novel_agg.get('top1_cos_mean',0):.3f} |",
        f"| top-1 accuracy (predict correct class) | "
        f"{base_agg.get('top1_acc',0)*100:.1f}% | "
        f"{novel_agg.get('top1_acc',0)*100:.1f}% |",
        f"| top-1 is BASE class | "
        f"{base_agg.get('top1_is_base_rate',0)*100:.1f}% | "
        f"{novel_agg.get('top1_is_base_rate',0)*100:.1f}% |",
        "",
        "## Per-novel-class breakdown",
        "",
        f"| novel class | n | mean cos to GT | top-1 acc | top-1 is base |",
        f"|---|---:|---:|---:|---:|",
    ]
    for cls, a in per_novel_class.items():
        md.append(
            f"| {cls} | {a['n']} | {a['gt_cos_mean']:.3f} | "
            f"{a['top1_acc']*100:.1f}% | {a['top1_is_base_rate']*100:.1f}% |"
        )

    md.extend(["", "## Decision tree hits", ""])
    for h in hits:
        md.append(f"- {h}")

    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[md] wrote {out_dir/'summary.md'}")

    # ────────────────────────────────────────────────────────────────
    # Plot
    # ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    base_gt_cos = [r["gt_cos"] for r in base_recs]
    novel_gt_cos = [r["gt_cos"] for r in novel_recs]
    axes[0].hist(base_gt_cos, bins=30, alpha=0.6, label=f"base (n={len(base_gt_cos)})", color="#2ca02c")
    axes[0].hist(novel_gt_cos, bins=30, alpha=0.6, label=f"novel (n={len(novel_gt_cos)})", color="#d62728")
    axes[0].axvline(np.mean(base_gt_cos), ls="--", color="#2ca02c")
    axes[0].axvline(np.mean(novel_gt_cos), ls="--", color="#d62728")
    axes[0].set_xlabel("cosine(image feature @ GT center, GT class vector)")
    axes[0].set_ylabel("count")
    axes[0].set_title("How well image features align with their GT class vector")
    axes[0].legend()

    # top-1 distribution
    base_is_base = sum(1 for r in base_recs if r["top1_is_base"])
    novel_is_base = sum(1 for r in novel_recs if r["top1_is_base"])
    cats = ["base GT\ntop-1=base", "base GT\ntop-1=novel",
            "novel GT\ntop-1=base", "novel GT\ntop-1=novel"]
    vals = [
        base_is_base,
        len(base_recs) - base_is_base,
        novel_is_base,
        len(novel_recs) - novel_is_base,
    ]
    colors = ["#2ca02c", "#88ee88", "#d62728", "#ee8888"]
    axes[1].bar(cats, vals, color=colors)
    axes[1].set_ylabel("count of GT bboxes")
    axes[1].set_title("Top-1 class type: base vs novel")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 1, str(v), ha="center", fontsize=10)

    fig.suptitle(f"THAF {encoder_name} ({embed_dim}d) — image encoder alignment diagnostic")
    fig.tight_layout()
    fig.savefig(out_dir / "image_encoder_alignment.png", dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_dir/'image_encoder_alignment.png'}")


if __name__ == "__main__":
    main()
