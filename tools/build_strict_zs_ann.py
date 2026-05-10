#!/usr/bin/env python
"""Build strict zero-shot ann files: test ann minus visproto holdout ann_ids.

Each visproto cache has 5 GT crops per class used to build the visual
prototype; those ann_ids leak into evaluation if we evaluate on the original
ann file. Strict ZS reports the same eval with those ann_ids removed.

Note: we only drop the *annotations*, keeping the images (the model still
sees them). This means the strict eval still tests whether the model
generalizes from the prototype, just without giving it credit for the 5
detections that match the exemplar source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = (
    ("main_3", "instances_test_main_novel.json"),
    ("pseudo_2", "instances_test_pseudo_novel.json"),
    ("hard_4", "instances_test_hard_novel.json"),  # may not exist
    ("full_5", "instances_test_novel.json"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir", default="/home1/liwenjie/TCT_NGC/annotations/")
    p.add_argument("--holdout-dir", default="data/texts/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    holdout_dir = Path(args.holdout_dir)

    for split, ann_name in SPLITS:
        ann_path = ann_dir / ann_name
        if not ann_path.exists():
            print(f"[skip] {split}: {ann_path} not found")
            continue

        holdout_path = holdout_dir / f"tct_ngc_novel_{split}_visproto_emb.holdout_anns.json"
        if not holdout_path.exists():
            print(f"[skip] {split}: holdout file missing {holdout_path}")
            continue

        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        bad_ids: set[int] = set()
        for ids in holdout.values():
            bad_ids.update(int(x) for x in ids)

        before = len(ann["annotations"])
        ann["annotations"] = [a for a in ann["annotations"] if int(a["id"]) not in bad_ids]
        after = len(ann["annotations"])
        kept_ids = {int(a["id"]) for a in ann["annotations"]}
        actually_dropped = bad_ids - (bad_ids - {a["id"] for a in ann["annotations"]})
        # the above is convoluted; simpler:
        actually_dropped = before - after

        out_name = ann_name.replace(".json", "_strict.json")
        out_path = ann_dir / out_name
        out_path.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
        print(
            f"[{split}] {ann_name} → {out_name}: "
            f"{before} anns → {after} (dropped {actually_dropped} of {len(bad_ids)} holdout)"
        )


if __name__ == "__main__":
    main()
