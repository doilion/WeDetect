#!/usr/bin/env python
"""Convert the 39-class 5-attribute dict JSON into the list-of-list formats
consumed by HierarchicalRandomLoadText (training) and the eval pipeline
(per-split class-name lookup against the THAF cache).

Input:
    data/texts/tct_ngc_fullnames_39_attr.json
        dict[class_name -> dict[attr_field -> str]]
        attr_field in {organ_specimen, diagnostic_code, cytomorphology,
                       background_and_immunoprofile, key_distinguishing_feature}

Outputs:
    data/texts/tct_ngc_fullnames_30_attr_train.json
        list-of-list, 30 entries by base cat_id, each = [organ, diag, morph,
        background, distinguish] in canonical order. HierarchicalRandomLoadText
        reads this directly during training.

    data/texts/tct_ngc_attr_<split>_eval.json   (× 4 splits)
        list-of-list, 5-attr inner lists in canonical order matching
        ATTR_FIELDS_ORDERED. Consumed by HierarchicalLoadText (the THAF
        test-time transform) via tools/eval_novel_thaf.py — the trained
        backbone's fusion module reads 5 attribute strings per class and
        produces a fused class vector at eval time.

        (Note: an earlier version of this tool emitted [[class_name]]
        1-element rows for v1 eval_novel_split.py replacement-backbone
        path. That format is incompatible with HierarchicalLoadText and
        was the root cause of bug B4.)

Canonical attribute order (matches attr_type_embed[0..4] in the fusion
module; reflects clinical reasoning flow from "where" to "what" to "how
distinguished"):
    0: organ_specimen
    1: diagnostic_code
    2: cytomorphology
    3: background_and_immunoprofile
    4: key_distinguishing_feature
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ATTR_FIELDS_ORDERED = (
    "organ_specimen",
    "diagnostic_code",
    "cytomorphology",
    "background_and_immunoprofile",
    "key_distinguishing_feature",
)

# (split_name, novel ann file relative to ann_dir)
NOVEL_SPLITS = (
    ("main_3", "instances_test_main_novel.json"),
    ("pseudo_2", "instances_test_pseudo_novel.json"),
    ("hard_4", "instances_hard_test.json"),
    ("full_5", "instances_test_novel.json"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--attr-json",
        default="data/texts/tct_ngc_fullnames_39_attr.json",
    )
    p.add_argument(
        "--base-ann",
        default="/home1/liwenjie/TCT_NGC_640/annotations/instances_train_dev_disjoint_dev30.json",
        help="dev30 base train ann; defines the 30 base class cat_id order",
    )
    p.add_argument(
        "--ann-dir",
        default="/home1/liwenjie/TCT_NGC/annotations/",
        help="directory holding the per-split novel ann files (test sets live "
        "under the original TCT_NGC tree, not the _640 cache)",
    )
    p.add_argument("--out-dir", default="data/texts/")
    p.add_argument(
        "--check-tokens",
        action="store_true",
        help="also tokenize each per-attr string via XLM-R and assert <256 tokens",
    )
    return p.parse_args()


def load_attr(path: Path) -> dict[str, dict[str, str]]:
    attr = json.loads(path.read_text(encoding="utf-8"))
    if len(attr) != 39:
        raise SystemExit(f"expected 39 entries in {path}, got {len(attr)}")
    for cls, fields in attr.items():
        missing = [f for f in ATTR_FIELDS_ORDERED if not fields.get(f, "").strip()]
        if missing:
            raise SystemExit(f"class {cls!r} missing/empty fields: {missing}")
    return attr


def class_names_in_cat_id_order(ann_path: Path) -> list[str]:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = sorted(data["categories"], key=lambda c: c["id"])
    return [c["name"] for c in cats]


def build_train_json(
    attr: dict[str, dict[str, str]],
    base_ann_path: Path,
) -> list[list[str]]:
    base_names = class_names_in_cat_id_order(base_ann_path)
    if len(base_names) != 30:
        raise SystemExit(
            f"base ann {base_ann_path} has {len(base_names)} categories, expected 30"
        )
    rows: list[list[str]] = []
    for name in base_names:
        if name not in attr:
            raise SystemExit(f"base class {name!r} missing from attr JSON")
        rows.append([attr[name][f].strip() for f in ATTR_FIELDS_ORDERED])
    return rows


def build_eval_json(
    novel_ann_path: Path,
    attr: dict[str, dict[str, str]],
) -> list[list[str]]:
    """Produce a list-of-list of 5 attribute strings per novel class, in cat_id
    order matching `novel_ann_path`. This is the format `HierarchicalLoadText`
    expects (5 attrs × num_classes). Used by tools/eval_novel_thaf.py to feed
    the trained THAF fusion module."""
    names = class_names_in_cat_id_order(novel_ann_path)
    for n in names:
        if n not in attr:
            raise SystemExit(f"novel class {n!r} missing from attr JSON")
    return [[attr[n][f].strip() for f in ATTR_FIELDS_ORDERED] for n in names]


def maybe_check_tokens(attr: dict[str, dict[str, str]]) -> None:
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(
        "/home/25_liwenjie/code/WeDetect/xlm-roberta-base"
    )
    max_per = 0
    max_concat = 0
    for cls, fields in attr.items():
        for f in ATTR_FIELDS_ORDERED:
            n = len(tk.encode(fields[f], add_special_tokens=True))
            if n > max_per:
                max_per = n
        n_concat = len(
            tk.encode(
                ". ".join(fields[f] for f in ATTR_FIELDS_ORDERED),
                add_special_tokens=True,
            )
        )
        if n_concat > max_concat:
            max_concat = n_concat
    if max_per >= 256:
        raise SystemExit(
            f"BiomedCLIP-PubMedBERT 256 limit may overflow on per-attr: max {max_per}"
        )
    if max_concat >= 512:
        raise SystemExit(
            f"XLM-R 512 limit may overflow on concat: max {max_concat}"
        )
    print(
        f"[token-check] max per-attr={max_per}, max concat={max_concat}; "
        f"BiomedCLIP 256 ✅, XLM-R 512 ✅"
    )


def main() -> None:
    args = parse_args()
    attr = load_attr(Path(args.attr_json))

    if args.check_tokens:
        maybe_check_tokens(attr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train JSON: 30 base classes × 5 attrs in canonical order
    train_rows = build_train_json(attr, Path(args.base_ann))
    train_path = out_dir / "tct_ngc_fullnames_30_attr_train.json"
    train_path.write_text(
        json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] wrote {train_path}  ({len(train_rows)} classes × {len(train_rows[0])} attrs)")

    # Per-split eval JSON: list-of-list of 5 attribute strings per class
    ann_dir = Path(args.ann_dir)
    for split, ann_name in NOVEL_SPLITS:
        ann_path = ann_dir / ann_name
        if not ann_path.exists():
            print(f"[skip] {split}: {ann_path} missing")
            continue
        rows = build_eval_json(ann_path, attr)
        out_path = out_dir / f"tct_ngc_attr_{split}_eval.json"
        out_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Show class names from the source ann (rows are now 5-attr strings,
        # not class names, so re-derive for log output).
        cls_names = class_names_in_cat_id_order(ann_path)
        print(
            f"[ok] wrote {out_path}  ({len(rows)} classes × {len(rows[0])} attrs: {cls_names})"
        )


if __name__ == "__main__":
    main()
