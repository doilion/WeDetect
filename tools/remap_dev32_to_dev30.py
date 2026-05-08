#!/usr/bin/env python
"""Remap a dev32 COCO ann JSON to dev30 by merging Urine 3 negative subclasses.

Merge rule:
  cat_id 16 (Urine-NILM) + 17 (Urine-Negative) + 20 (Urine-Negative Degeneration)
   → single cat_id 16 named "Urine-NHGUC" (Paris System NHGUC).

Other class names are preserved; cat_ids are renumbered to a contiguous [0..29]
because the dev32 source already had non-contiguous ids 0..15 / 16..20 / 23 / 31..40.

Sanity checks at runtime:
  - Source cat_ids must include {16, 17, 18, 19, 20, 23} (Urine block) AND
    {31..40} (TCT_CCD block) AND {0..15} (the rest).
  - Source class names at the merged ids must match expected (NILM / Negative /
    Negative Degeneration). This guards against a future ann-file regen that
    silently shuffled cat ordering.

Image-level fields (file_name, width, height, image_id) are unchanged.
Bbox coordinates are unchanged. Only `categories` and `annotations[*].category_id`
are touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# dev32 → dev30 cat_id remap.
# Identity for 0..15. Urine 16/17/20 → 16 (NHGUC). Then shift to fill 17..29.
REMAP_DEV32_TO_DEV30 = {
    # 0..15 unchanged
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 7, 8: 8,
    9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
    # Urine merge:
    16: 16,  # NILM             → NHGUC
    17: 16,  # Negative         → NHGUC
    20: 16,  # Negative Degen   → NHGUC
    18: 17,  # SHGUC
    19: 18,  # AUC
    23: 19,  # HGUC
    # TCT_CCD: shift down (31..40) → 20..29
    31: 20, 32: 21, 33: 22, 34: 23, 35: 24,
    36: 25, 37: 26, 38: 27, 39: 28, 40: 29,
}

# Sanity assertion targets: ALL 32 source cat_ids must match expected names.
# This guards against silent label corruption when a regenerated ann file
# shuffles cat order — even outside the merged Urine block, because
# NEW_CATEGORIES below is positionally hardcoded and the remap table assumes
# a specific dev32 cat_id → name mapping.
EXPECTED_DEV32_NAMES = {
    0: "respiratory tract-Neutrophil",
    1: "respiratory tract-Alveolar macrophages",
    2: "respiratory tract-Ciliated columnar epithelial cells",
    3: "respiratory tract-Lymphocyte",
    4: "respiratory tract-Impurity",
    5: "respiratory tract-Squamous epithelial cells",
    6: "respiratory tract-Diseased cells",
    7: "Serous effusion-Negative samples",
    8: "Serous effusion-Diseased cells",
    9: "Thyroid gland-PTC",
    10: "Thyroid gland-SPTC",
    11: "Thyroid gland-NS",
    12: "Thyroid gland-Macrophages",
    13: "Thyroid gland-AUC",
    14: "Thyroid gland-Negative samples",
    15: "Thyroid gland-FC",
    16: "Urine-NILM",
    17: "Urine-Negative",
    18: "Urine-SHGUC",
    19: "Urine-AUC",
    20: "Urine-Negative Degeneration",
    23: "Urine-HGUC",
    31: "TCT_CCD-normal",
    32: "TCT_CCD-ascus",
    33: "TCT_CCD-asch",
    34: "TCT_CCD-lsil",
    35: "TCT_CCD-hsil_scc_omn",
    36: "TCT_CCD-agc_adenocarcinoma_em",
    37: "TCT_CCD-vaginalis",
    38: "TCT_CCD-monilia",
    39: "TCT_CCD-dysbacteriosis_herpes_act",
    40: "TCT_CCD-ec",
}

# Final dev30 categories (id, name, supercategory).
NEW_CATEGORIES = [
    {"id":  0, "name": "respiratory tract-Neutrophil",                   "supercategory": "respiratory_tract"},
    {"id":  1, "name": "respiratory tract-Alveolar macrophages",         "supercategory": "respiratory_tract"},
    {"id":  2, "name": "respiratory tract-Ciliated columnar epithelial cells", "supercategory": "respiratory_tract"},
    {"id":  3, "name": "respiratory tract-Lymphocyte",                   "supercategory": "respiratory_tract"},
    {"id":  4, "name": "respiratory tract-Impurity",                     "supercategory": "respiratory_tract"},
    {"id":  5, "name": "respiratory tract-Squamous epithelial cells",    "supercategory": "respiratory_tract"},
    {"id":  6, "name": "respiratory tract-Diseased cells",               "supercategory": "respiratory_tract"},
    {"id":  7, "name": "Serous effusion-Negative samples",               "supercategory": "Serous_effusion"},
    {"id":  8, "name": "Serous effusion-Diseased cells",                 "supercategory": "Serous_effusion"},
    {"id":  9, "name": "Thyroid gland-PTC",                              "supercategory": "Thyroid_gland"},
    {"id": 10, "name": "Thyroid gland-SPTC",                             "supercategory": "Thyroid_gland"},
    {"id": 11, "name": "Thyroid gland-NS",                               "supercategory": "Thyroid_gland"},
    {"id": 12, "name": "Thyroid gland-Macrophages",                      "supercategory": "Thyroid_gland"},
    {"id": 13, "name": "Thyroid gland-AUC",                              "supercategory": "Thyroid_gland"},
    {"id": 14, "name": "Thyroid gland-Negative samples",                 "supercategory": "Thyroid_gland"},
    {"id": 15, "name": "Thyroid gland-FC",                               "supercategory": "Thyroid_gland"},
    {"id": 16, "name": "Urine-NHGUC",                                    "supercategory": "Urine"},
    {"id": 17, "name": "Urine-SHGUC",                                    "supercategory": "Urine"},
    {"id": 18, "name": "Urine-AUC",                                      "supercategory": "Urine"},
    {"id": 19, "name": "Urine-HGUC",                                     "supercategory": "Urine"},
    {"id": 20, "name": "TCT_CCD-normal",                                 "supercategory": "TCT_CCD"},
    {"id": 21, "name": "TCT_CCD-ascus",                                  "supercategory": "TCT_CCD"},
    {"id": 22, "name": "TCT_CCD-asch",                                   "supercategory": "TCT_CCD"},
    {"id": 23, "name": "TCT_CCD-lsil",                                   "supercategory": "TCT_CCD"},
    {"id": 24, "name": "TCT_CCD-hsil_scc_omn",                           "supercategory": "TCT_CCD"},
    {"id": 25, "name": "TCT_CCD-agc_adenocarcinoma_em",                  "supercategory": "TCT_CCD"},
    {"id": 26, "name": "TCT_CCD-vaginalis",                              "supercategory": "TCT_CCD"},
    {"id": 27, "name": "TCT_CCD-monilia",                                "supercategory": "TCT_CCD"},
    {"id": 28, "name": "TCT_CCD-dysbacteriosis_herpes_act",              "supercategory": "TCT_CCD"},
    {"id": 29, "name": "TCT_CCD-ec",                                     "supercategory": "TCT_CCD"},
]


def remap(in_path: Path, out_path: Path) -> dict:
    data = json.loads(in_path.read_text(encoding="utf-8"))

    src_cats = {c["id"]: c["name"] for c in data["categories"]}
    if len(src_cats) != 32:
        raise SystemExit(
            f"expected 32 source categories, got {len(src_cats)} in {in_path}")

    # Sanity check expected names at expected ids
    for cid, expected in EXPECTED_DEV32_NAMES.items():
        actual = src_cats.get(cid)
        if actual != expected:
            raise SystemExit(
                f"sanity fail: cat_id {cid} should be {expected!r} but is {actual!r} "
                f"in {in_path}. The dev32 source ann may have been regenerated "
                f"with shuffled cat ids; remap table needs to be re-derived.")

    # Sanity check all source ids are covered by REMAP
    missing = [cid for cid in src_cats if cid not in REMAP_DEV32_TO_DEV30]
    if missing:
        raise SystemExit(
            f"REMAP_DEV32_TO_DEV30 missing source cat_ids {missing} in {in_path}")

    # Replace categories
    data["categories"] = NEW_CATEGORIES

    # Remap annotations
    cnt_merged = 0
    for ann in data["annotations"]:
        old = ann["category_id"]
        new = REMAP_DEV32_TO_DEV30[old]
        if old in (17, 20):
            cnt_merged += 1
        ann["category_id"] = new

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "n_images": len(data["images"]),
        "n_annotations": len(data["annotations"]),
        "n_categories_in": len(src_cats),
        "n_categories_out": len(NEW_CATEGORIES),
        "n_anns_merged_into_NHGUC_from_17_20": cnt_merged,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()
    s = remap(Path(args.in_path), Path(args.out_path))
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
