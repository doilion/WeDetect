#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from wedetect.models.backbones.mm_backbone import XLMRobertaLanguageBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cached text embeddings for PseudoLanguageBackbone."
    )
    parser.add_argument("--texts", required=True, help="Class text JSON file.")
    parser.add_argument("--out", required=True, help="Output .pth file.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/wedetect_tiny.pth",
        help="WeDetect checkpoint containing backbone.text_model weights.",
    )
    parser.add_argument("--model-name", default="./xlm-roberta-base/")
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_text_model(args: argparse.Namespace) -> XLMRobertaLanguageBackbone:
    model = XLMRobertaLanguageBackbone(
        model_name=args.model_name,
        model_size=args.model_size,
        frozen_modules=("all",),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "backbone.text_model."
    text_state = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not text_state:
        raise RuntimeError(f"No {prefix} weights found in {args.checkpoint}")

    incompatible = model.load_state_dict(text_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected text-model load result: {incompatible}")

    model.to(args.device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    text_groups = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    texts = list(itertools.chain.from_iterable(text_groups))
    slash_texts = [text for text in texts if "/" in text]
    if slash_texts:
        raise ValueError(
            "Prompt text cannot contain slash because PseudoLanguageBackbone "
            f"uses split lookup. Examples: {slash_texts[:3]}"
        )
    if len(texts) != len(set(texts)):
        repeated = sorted({text for text in texts if texts.count(text) > 1})
        raise ValueError(f"Duplicate prompt strings would collide: {repeated}")

    model = load_text_model(args)
    with torch.no_grad():
        embeddings = model([texts]).squeeze(0).detach().cpu().float()

    embed_dict = {
        text: embeddings[index].contiguous()
        for index, text in enumerate(texts)
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embed_dict, out_path)
    print(f"Wrote {out_path} with {len(embed_dict)} embeddings")


if __name__ == "__main__":
    main()
