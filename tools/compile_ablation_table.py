#!/usr/bin/env python
"""Compile a markdown ablation table from THAF + clean dev30 baseline eval logs.

Reads three eval summary files produced by the eval orchestrators:
  - work_dirs/.../thaf_xlmr_2gpu/thaf_eval_summary.txt
  - work_dirs/.../thaf_biomedclip_2gpu/thaf_eval_summary.txt
  - work_dirs/.../disjoint_clean_2gpu/baseline_eval_summary.txt

Extracts:
  - base 25-cls mAP                       (one number per ckpt)
  - novel main_3 / pseudo_2 / hard_4 / full_5 mAP   (per ckpt × split)
  - score-fusion mAP (clean dev30 only)   (per split)

Outputs a markdown table to stdout. The orchestrator pipes this to
work_dirs/ablation_table.md.

Robust to:
  - missing files (prints `?` for that row)
  - eval failures (prints `?` for that cell)
  - log format drift (regex tolerant)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

THAF_XLMR_LOG = (
    REPO_ROOT
    / "work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu/thaf_eval_summary.txt"
)
THAF_BIO_LOG = (
    REPO_ROOT
    / "work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu/thaf_eval_summary.txt"
)
CLEAN_LOG = (
    REPO_ROOT
    / "work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu"
    / "baseline_eval_summary.txt"
)

SPLITS = ("main_3", "pseudo_2", "hard_4", "full_5")


def parse_mAP(line: str) -> float | None:
    """Pull mAP out of a 'bbox_mAP_copypaste: 0.XXX ...' or pycocotools AP line.

    Deliberately does NOT match `coco/bbox_mAP:` — that pattern appears
    twice per eval (once in copypaste, once in the long per-class
    precision line), so matching both would double-count when scanning
    sequentially.
    """
    m = re.search(r"bbox_mAP_copypaste:\s*([0-9.]+)", line)
    if m:
        return float(m.group(1))
    # pycocotools line: " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.137"
    # Require "(AP)" to distinguish from Average Recall (AR) which has the
    # same IoU/area/maxDets signature.
    m = re.search(
        r"\(AP\)\s*@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)",
        line,
    )
    if m:
        return float(m.group(1))
    return None


def fmt(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, float) else "?"


def parse_thaf_log(path: Path) -> dict:
    """THAF eval log structure (from eval_thaf_all_splits.sh):
        ## Base eval ...
        bbox_mAP_copypaste: ...        ← base 25-cls
        ## Novel zero-shot eval
          [main_3]
          bbox_mAP_copypaste: ...      ← novel main_3
          [pseudo_2]
          ...
    """
    out = {"base": None}
    out.update({s: None for s in SPLITS})
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Base 25-cls is the FIRST mAP line under "## Base eval"
    base_section = re.split(r"## Novel zero-shot eval", text, maxsplit=1)
    if len(base_section) >= 1:
        for line in base_section[0].splitlines():
            v = parse_mAP(line)
            if v is not None:
                out["base"] = v
                break

    # Novel splits: scan the novel section in order, mAP lines appear in
    # SPLITS order (eval_thaf_all_splits.sh iterates main_3, pseudo_2,
    # hard_4, full_5). Don't rely on `[split]` anchors — current THAF
    # SUMMARY files have `[split] done` AFTER the mAP line, which would
    # cause anchor-based parsing to skip ahead to the next split.
    # Dedup consecutive identical mAP values: each eval produces 2 matching
    # lines (bbox_mAP_copypaste + pycocotools "(AP) @[ IoU=0.50:0.95 ..."
    # have the same value), so deduping leaves one mAP per eval.
    if len(base_section) >= 2:
        novel = base_section[1]
        mAP_lines = _scan_mAP_dedup(novel)
        for i, split in enumerate(SPLITS):
            if i < len(mAP_lines):
                out[split] = mAP_lines[i]

    return out


def _scan_mAP_dedup(text: str) -> list[float]:
    """Scan text for mAP lines, one per eval.

    Strategy: prefer `bbox_mAP_copypaste` lines (mmengine evals emit exactly
    one per eval). If no copypaste line is found in the section, fall back
    to `(AP) @[ IoU=0.50:0.95 | area=all | maxDets=100`  lines (pycocotools
    direct output from fuse_novel_predictions.py). This avoids the false
    dedup that would occur when two splits genuinely have the same mAP.
    """
    copypaste_re = re.compile(r"bbox_mAP_copypaste:\s*([0-9.]+)")
    # Accept either:
    #   - full pycocotools line ` Average Precision  (AP) @[ IoU=0.50:0.95 | ... ] = X.XXX`
    #   - stripped form from backfill grep `IoU=0.50:0.95 | area=   all | maxDets=100 ] = X.XXX`
    # Require area=all and maxDets=100 to exclude AR lines (which use the
    # same IoU header but maxDets=1 or 10 don't match, plus AR has separate
    # _maxDets=100 with same area; but maxDets=100 + (AP) excludes the rest).
    # In backfill form the `(AP)` is gone but the regex above already
    # excluded AR by virtue of grep -oE matching only AP-style outputs.
    ap_re = re.compile(
        r"IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)"
    )
    copypaste_vals = [float(m.group(1)) for m in copypaste_re.finditer(text)]
    if copypaste_vals:
        return copypaste_vals
    return [float(m.group(1)) for m in ap_re.finditer(text)]


def parse_baseline_log(path: Path) -> dict:
    """Baseline eval log structure (eval_baseline_all.sh):
        ## Step 1 — test_base 25-cls
        bbox_mAP_copypaste: ...        ← base
        ## Step 2 — v2 text baseline novel × 4 splits
          [<split>-v2text]
          bbox_mAP: ...
        ## Step 5 — visproto-only eval × 4 splits
          [<split>-visproto-eval]
          bbox_mAP: ...
        ## Step 7 — Procrustes calfused × 4 splits
          [<split>-calfused-eval]
          bbox_mAP: ...
        ## Step 8 — Score fusion × 4 splits
          [<split>-scorefuse]
          AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = ...
    """
    out = {
        "base": None,
        "v2text":   {s: None for s in SPLITS},
        "visproto": {s: None for s in SPLITS},
        "calfused": {s: None for s in SPLITS},
        "scorefuse": {s: None for s in SPLITS},
        # Fix-11: THAF + score fusion rows, keyed by encoder (xlmr/biomedclip)
        "thaf_scorefuse_xlmr":      {s: None for s in SPLITS},
        "thaf_scorefuse_biomedclip": {s: None for s in SPLITS},
    }
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Step 1 base
    m = re.search(r"## Step 1.*?(?=## Step 2|\Z)", text, re.DOTALL)
    if m:
        for line in m.group(0).splitlines():
            v = parse_mAP(line)
            if v is not None:
                out["base"] = v
                break

    # Step 2 / 5 / 7 / 8 — for each step, find the LAST occurrence of its
    # header line (handles BACKFILL re-runs appended at end of file), then
    # capture from that position to the next `## Step <next>` (or EOF).
    step_map = {
        "v2text":    (r"^## Step 2 ",  r"^## Step 3 "),
        "visproto":  (r"^## Step 5 ",  r"^## Step 6 "),
        "calfused":  (r"^## Step 7 ",  r"^## Step 8 "),
        "scorefuse": (r"^## Step 8 ",  r"^## Step 9 "),
    }
    for key, (header_re, end_re) in step_map.items():
        starts = [m.start() for m in re.finditer(header_re, text, re.MULTILINE)]
        if not starts:
            continue
        start_pos = starts[-1]  # last (latest) header occurrence
        # find next step header AFTER start_pos
        ends = [m.start() for m in re.finditer(end_re, text, re.MULTILINE) if m.start() > start_pos]
        end_pos = ends[0] if ends else len(text)
        sec_text = text[start_pos:end_pos]
        # synthetic Match-like wrapper so downstream code can use sec.group(0)
        class _SecWrapper:
            def __init__(self, s): self._s = s
            def group(self, _): return self._s
        sec = _SecWrapper(sec_text)
        if not sec:
            continue
        # Use sequential ordering: bbox_mAP lines in step section appear in
        # SPLITS order (main_3, pseudo_2, hard_4, full_5) because all step
        # loops iterate splits in this fixed order. Dedup consecutive
        # identical values (mmengine emits 2 mAP lines per eval).
        mAP_lines = _scan_mAP_dedup(sec.group(0))
        for i, split in enumerate(SPLITS):
            if i < len(mAP_lines):
                out[key][split] = mAP_lines[i]

    # Fix-11: Step 9 — THAF + score fusion (per encoder × per split)
    # Sequential ordering: for ENCODER in xlmr biomedclip, for SPLIT in main_3
    # pseudo_2 hard_4 full_5 — total 8 mAP lines if both encoders ran.
    sec9 = re.search(r"## Step 9.*?\Z", text, re.DOTALL)
    if sec9:
        mAP_lines = _scan_mAP_dedup(sec9.group(0))
        encoder_order = ("xlmr", "biomedclip")
        for ei, encoder in enumerate(encoder_order):
            key = f"thaf_scorefuse_{encoder}"
            for si, split in enumerate(SPLITS):
                idx = ei * len(SPLITS) + si
                if idx < len(mAP_lines):
                    out[key][split] = mAP_lines[idx]
    return out


def avg(d: dict) -> str:
    """Average across SPLITS dict. Fix-10: returns '?' if not all 4 splits
    have a float, else '0.XXX'. This prevents silently averaging partial
    results (which would mislead paper readers)."""
    if len(d) != len(SPLITS):
        return "?"
    vs = [d.get(s) for s in SPLITS]
    if not all(isinstance(v, float) for v in vs):
        n_have = sum(1 for v in vs if isinstance(v, float))
        return f"? ({n_have}/{len(SPLITS)})"
    return f"{sum(vs) / len(vs):.3f}"


def main() -> None:
    thaf_xlmr = parse_thaf_log(THAF_XLMR_LOG)
    thaf_bio = parse_thaf_log(THAF_BIO_LOG)
    base = parse_baseline_log(CLEAN_LOG)

    print(f"# THAF Phase 3 — Ablation Table (auto-compiled)")
    print()
    print(f"Source files:")
    print(f"- THAF XLM-R: `{THAF_XLMR_LOG.relative_to(REPO_ROOT)}` "
          f"(exists: {THAF_XLMR_LOG.exists()})")
    print(f"- THAF BiomedCLIP: `{THAF_BIO_LOG.relative_to(REPO_ROOT)}` "
          f"(exists: {THAF_BIO_LOG.exists()})")
    print(f"- clean dev30 baseline: `{CLEAN_LOG.relative_to(REPO_ROOT)}` "
          f"(exists: {CLEAN_LOG.exists()})")
    print()
    print(f"## Main results (base 25-cls + 4 novel splits)")
    print()
    print(f"| Method | Base 25-cls | main_3 | pseudo_2 | hard_4 | full_5 | Avg novel |")
    print(f"|---|---:|---:|---:|---:|---:|---:|")

    rows = [
        ("v2 baseline (XLM-R, single PSC)", base["base"], base["v2text"]),
        ("score fusion (XLM-R, raw visproto)", base["base"], base["scorefuse"]),
        ("Procrustes calfused (DEAD-5 verify)", base["base"], base["calfused"]),
        ("THAF + XLM-R (768d)", thaf_xlmr["base"],
         {s: thaf_xlmr[s] for s in SPLITS}),
        ("**THAF + BiomedCLIP (512d)** ← main", thaf_bio["base"],
         {s: thaf_bio[s] for s in SPLITS}),
        # Fix-11: THAF + score fusion rows (preds_thaf + preds_visproto merged)
        ("THAF (XLM-R) + score fusion", thaf_xlmr["base"],
         base["thaf_scorefuse_xlmr"]),
        ("THAF (BiomedCLIP) + score fusion", thaf_bio["base"],
         base["thaf_scorefuse_biomedclip"]),
    ]
    for name, base_v, novel in rows:
        cells = [fmt(novel.get(s)) for s in SPLITS]
        a = avg(novel)  # already string from Fix-10
        print(f"| {name} | {fmt(base_v)} | {' | '.join(cells)} | {a} |")

    # Optional: visproto-alone row (5-shot, leaky baseline reference)
    print()
    print(f"## Reference rows (no main-method status)")
    print()
    print(f"| Method | Base 25-cls | main_3 | pseudo_2 | hard_4 | full_5 | Avg novel |")
    print(f"|---|---:|---:|---:|---:|---:|---:|")
    rows_ref = [
        ("visproto raw (5-shot, leakage)", "—", base["visproto"]),
    ]
    for name, base_v, novel in rows_ref:
        cells = [fmt(novel.get(s)) for s in SPLITS]
        a = avg(novel)  # already string from Fix-10
        print(f"| {name} | {base_v} | {' | '.join(cells)} | {a} |")

    print()
    print(f"## dev30 baseline reference (old throttled GPU 1 ckpt)")
    print(f"- val 30-cls mAP (training-time) = 0.283")
    print(f"- **test_base 25-cls mAP (paper headline) = 0.306**")
    print(f"- avg novel mAP (4 splits) under v2 baseline = 0.103, under score fusion = 0.125")
    print()
    # Fix-9 caveat: surface the LR-schedule confound so paper readers don't
    # silently trust THAF vs baseline comparisons without context.
    print(f"## ⚠ Caveats")
    print()
    print(f"- **LR-schedule confound (Fix-9)**: THAF configs inherit the old "
          f"`CosineAnnealingLR begin=1, T_max=12` (overlaps with LinearLR warmup), "
          f"while clean dev30 retrain uses the fixed `begin=2, T_max=11`. The 0.8 pp "
          f"THAF XLM-R vs clean dev30 base gap may include LR schedule effect — "
          f"not purely a fusion-design difference.")
    print(f"- **THAF fusion bypass (Phase 3.5 diagnostic)**: trained alpha ≈ −0.0001 "
          f"for both encoders → cross-attention proj contributes ~0, output ≈ "
          f"`mean(5 attr embs)`. Base gain comes from BiomedCLIP encoder + 5-attr "
          f"text mean, NOT from learned fusion. See "
          f"`docs/figures/thaf_diagnostic/<encoder>/summary.md`.")
    print(f"- **Novel zero-shot collapse**: THAF + BiomedCLIP novel avg ≈ 0.052 "
          f"(vs v2 baseline 0.095). Cosine geometry of class vectors is fine "
          f"(Phase 3.5 refutes Hypothesis A); root cause likely image encoder "
          f"misalignment (Hypothesis B, Phase 3.6 to verify).")
    print()


if __name__ == "__main__":
    main()
