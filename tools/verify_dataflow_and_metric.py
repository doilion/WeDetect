"""Independent end-to-end verification of organ-restricted dataflow + metric.

Runs without any of our new code (uses pycocotools directly) to check that
OrganRestrictedCocoMetric's reported numbers actually match what an
independent COCOeval pass produces.

Checks performed:
  1. Class-to-organ mask file consistency:
     - file_name path parsing agrees with taxonomy.classes[name].organ_id
     - mask matrix [C, O] has exactly one '1' per row
     - mask class order == ann.categories sorted by id
  2. OrganExtractor parsing on 100 random training images
     - organ_id matches class.organ for at least one GT in each image
       (since TCT_NGC is single-organ-per-image)
  3. Predictions integrity:
     - 0 cross-organ detections (mask worked at inference)
     - all category_ids are within ann.categories
  4. Per-organ AP independent recomputation:
     - For each organ, run vanilla COCOeval with catIds=organ_cls_ids,
       imgIds=all → AP. Compare to OrganRestrictedCocoMetric output.
     - macro_mAP = mean(per-organ AP) should match
  5. all-class flat: vanilla COCOeval over all kept cat_ids → AP. Compare.
  6. instance-weighted mAP: sum(AP_organ * n_inst_organ) / total_inst.

Usage:
    python tools/verify_dataflow_and_metric.py \\
        --ann /home1/liwenjie/TCT_NGC/annotations/instances_test_novel_merged_9.json \\
        --preds work_dirs/.../eval_organ_restricted_novel9/preds_organ.bbox.json \\
        --mask data/texts/tct_ngc_class_organ_mask_novel_merged.pt \\
        --taxonomy data/texts/tct_ngc_taxonomy.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_organ_from_path(file_name, organ_to_id):
    """Mirror of OrganExtractor logic: split on '/' and underscore-normalize."""
    segments = file_name.split('/')
    for seg in segments:
        organ = seg.replace('_', ' ')
        if organ in organ_to_id:
            return organ_to_id[organ], organ
    return -1, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ann', required=True, type=Path)
    p.add_argument('--preds', required=True, type=Path)
    p.add_argument('--mask', required=True, type=Path)
    p.add_argument('--taxonomy', required=True, type=Path)
    p.add_argument('--exclude-class-names', default=None,
                   help='comma-separated names to drop (for base 25 parity)')
    p.add_argument('--exclude-from-macro', action='store_true',
                   help='also exclude these from per-organ macro (else only flat is excluded)')
    args = p.parse_args()

    ann = json.loads(args.ann.read_text())
    tax = json.loads(args.taxonomy.read_text())
    mask_pkg = torch.load(args.mask, weights_only=False, map_location='cpu')
    organ_to_id = tax['organ_to_id']
    organ_names = tax['organs']
    tax_classes = tax['classes']

    # -- 1. Mask consistency --------------------------------------------
    print('=== (1) Mask consistency check ===')
    cats_sorted = sorted(ann['categories'], key=lambda c: c['id'])
    ann_class_names = [c['name'] for c in cats_sorted]
    ann_class_ids = [c['id'] for c in cats_sorted]

    if list(mask_pkg['class_names']) != ann_class_names:
        print(f'  FAIL: mask class_names != ann sorted-by-id class_names')
        print(f'    mask[:3]: {list(mask_pkg["class_names"])[:3]}')
        print(f'    ann[:3]:  {ann_class_names[:3]}')
        sys.exit(1)
    if list(mask_pkg['class_ids']) != ann_class_ids:
        print(f'  FAIL: mask class_ids != ann sorted ids')
        sys.exit(1)
    print(f'  OK  mask.class_names == ann.categories sorted by id ({len(ann_class_names)} classes)')

    M = mask_pkg['mask']
    row_sum = M.sum(dim=1)
    if not torch.all(row_sum == 1.0):
        print(f'  FAIL: not every row sums to 1 (row_sum unique values: {row_sum.unique().tolist()})')
        sys.exit(1)
    print(f'  OK  mask matrix [{M.shape[0]}, {M.shape[1]}] has exactly one 1 per row')

    # Check class → organ matches taxonomy
    mismatches = 0
    for i, name in enumerate(ann_class_names):
        tax_o = tax_classes[name]['organ_id']
        mask_o = int(M[i].argmax().item())
        if tax_o != mask_o:
            mismatches += 1
            print(f'  MISMATCH: class {name!r}  taxonomy.organ_id={tax_o}  mask.argmax={mask_o}')
    if mismatches:
        print(f'  FAIL: {mismatches} class→organ mismatches')
        sys.exit(1)
    print(f'  OK  all {len(ann_class_names)} class→organ map taxonomy<->mask')

    # -- 2. OrganExtractor parsing on 100 random images -----------------
    print()
    print('=== (2) OrganExtractor parsing vs class.organ ===')
    import random
    random.seed(20260513)
    sample_imgs = random.sample(ann['images'], min(100, len(ann['images'])))
    anns_by_img = defaultdict(list)
    for a in ann['annotations']:
        anns_by_img[a['image_id']].append(a['category_id'])
    cat_to_organ = {cat['id']: tax_classes[cat['name']]['organ_id'] for cat in ann['categories']}

    parse_fails = 0
    organ_mismatches = 0
    for img in sample_imgs:
        oid, oname = parse_organ_from_path(img['file_name'], organ_to_id)
        if oid < 0:
            parse_fails += 1
            print(f'  PARSE_FAIL: {img["file_name"]}')
            continue
        # Compare to GT class organs
        gt_organs = set(cat_to_organ[cid] for cid in anns_by_img[img['id']])
        if gt_organs and oid not in gt_organs:
            organ_mismatches += 1
            print(f'  ORGAN_MISMATCH img_id={img["id"]} parsed={oid} ({oname}) '
                  f'gt_organs={gt_organs}  file={img["file_name"][:60]}')
    if parse_fails or organ_mismatches:
        print(f'  FAIL: {parse_fails} parse fails, {organ_mismatches} GT mismatches')
        sys.exit(1)
    print(f'  OK  100/100 images: organ from file_name == organ of GT class')

    # -- 3. Predictions integrity ---------------------------------------
    print()
    print('=== (3) Predictions integrity (mask worked at inference) ===')
    preds = json.loads(args.preds.read_text())
    img_to_organ = {img['id']: parse_organ_from_path(img['file_name'], organ_to_id)[0]
                    for img in ann['images']}
    valid_cat_ids = set(c['id'] for c in ann['categories'])

    cross_organ = 0
    invalid_cat = 0
    score_zero = 0
    for p in preds:
        if p['category_id'] not in valid_cat_ids:
            invalid_cat += 1
            continue
        if cat_to_organ[p['category_id']] != img_to_organ.get(p['image_id'], -1):
            cross_organ += 1
        if p['score'] == 0:
            score_zero += 1
    print(f'  total preds: {len(preds)}')
    print(f'  cross-organ detections: {cross_organ}  (expected 0 if mask worked)')
    print(f'  invalid category_id:    {invalid_cat}   (expected 0)')
    print(f'  score == 0:             {score_zero}    (expected 0 — score_thr filters those)')
    if cross_organ or invalid_cat:
        print(f'  FAIL: mask did not work or category id misalignment')
        sys.exit(1)
    print(f'  OK  predictions are organ-clean')

    # -- 4. Per-organ AP independent recompute --------------------------
    print()
    print('=== (4) Per-organ AP independent recompute (vanilla COCOeval) ===')
    cocoGt = COCO(str(args.ann))
    cocoDt = cocoGt.loadRes(str(args.preds))

    excluded_names = set()
    if args.exclude_class_names:
        excluded_names = set(n.strip() for n in args.exclude_class_names.split(',') if n.strip())
    excluded_cat_ids = {c['id'] for c in ann['categories'] if c['name'] in excluded_names}
    kept_cat_ids = [c['id'] for c in cats_sorted if c['id'] not in excluded_cat_ids]

    # Per-organ
    organ_to_kept_cats = defaultdict(list)
    for cid in kept_cat_ids:
        organ_to_kept_cats[cat_to_organ[cid]].append(cid)

    n_inst_per_organ = defaultdict(int)
    for a in ann['annotations']:
        if a['category_id'] in excluded_cat_ids:
            continue
        n_inst_per_organ[cat_to_organ[a['category_id']]] += 1

    organ_aps = {}
    for oid in sorted(organ_to_kept_cats):
        cats = organ_to_kept_cats[oid]
        n_inst = n_inst_per_organ[oid]
        if n_inst == 0:
            print(f'  organ={organ_names[oid]:24s}  n_classes={len(cats):2d}  n_inst=     0  '
                  f'AP=NaN  (skipped, no GT)')
            continue
        e = COCOeval(cocoGt, cocoDt, 'bbox')
        e.params.catIds = cats
        e.evaluate()
        e.accumulate()
        # Suppress summarize() stdout (we only want stats[])
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            e.summarize()
        ap = float(e.stats[0])
        ap_50 = float(e.stats[1])
        ap_75 = float(e.stats[2])
        organ_aps[oid] = (ap, ap_50, ap_75, n_inst, len(cats))
        print(f'  organ={organ_names[oid]:24s}  n_classes={len(cats):2d}  '
              f'n_inst={n_inst:6d}  AP={ap:.4f}  AP50={ap_50:.4f}  AP75={ap_75:.4f}')

    macro = np.mean([v[0] for v in organ_aps.values()])
    inst_w_total = sum(v[3] for v in organ_aps.values())
    inst_weighted = sum(v[0] * v[3] for v in organ_aps.values()) / max(inst_w_total, 1)
    print(f'  overall macro:      {macro:.4f}')
    print(f'  instance-weighted:  {inst_weighted:.4f}')

    # -- 5. All-class flat -----------------------------------------------
    print()
    print('=== (5) All-class flat (vanilla COCOeval, all kept catIds) ===')
    e = COCOeval(cocoGt, cocoDt, 'bbox')
    e.params.catIds = kept_cat_ids
    e.evaluate()
    e.accumulate()
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        e.summarize()
    flat_ap = float(e.stats[0])
    flat_50 = float(e.stats[1])
    flat_75 = float(e.stats[2])
    print(f'  n_classes={len(kept_cat_ids)}  AP={flat_ap:.4f}  AP50={flat_50:.4f}  AP75={flat_75:.4f}')

    # Summary
    print()
    print('=== Summary ===')
    print(f'  overall macro      = {macro:.4f}')
    print(f'  all-class flat     = {flat_ap:.4f}')
    print(f'  instance-weighted  = {inst_weighted:.4f}')
    print()
    print('  Compare these to the OrganRestrictedCocoMetric table from the eval log.')
    print('  If they match → metric is correct.')


if __name__ == '__main__':
    main()
