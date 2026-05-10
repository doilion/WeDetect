#!/usr/bin/env python
"""Print a per-epoch mAP table from a THAF training log.

Parses the `Epoch(val) [N][809/809]    coco/bbox_mAP: 0.XXXX  coco/bbox_mAP_50: ...`
lines that mmengine writes after each epoch's val pass and emits a clean
markdown table. Use it during long training runs or after completion.

Usage:
    python tools/summarize_thaf_training_progress.py work_dirs/.../2026*/2026*.log
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


VAL_RE = re.compile(
    r"Epoch\(val\)\s+\[(\d+)\].*coco/bbox_mAP:\s*([\d\.\-]+)"
    r"\s+coco/bbox_mAP_50:\s*([\d\.\-]+)"
    r"\s+coco/bbox_mAP_75:\s*([\d\.\-]+)"
    r"\s+coco/bbox_mAP_s:\s*([\d\.\-]+)"
    r"\s+coco/bbox_mAP_m:\s*([\d\.\-]+)"
    r"\s+coco/bbox_mAP_l:\s*([\d\.\-]+)"
)
TIME_RE = re.compile(r"^(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("log", help="path to mmengine training log")
    p.add_argument(
        "--baseline",
        nargs="*",
        default=[],
        help="optional dev30 baseline mAP per epoch (space-separated, e.g. "
        "--baseline 0.07 0.13 0.21 0.23 ...)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        sys.exit(f"log not found: {log_path}")

    rows: list[dict] = []
    last_time = ""
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tm = TIME_RE.match(line)
        if tm:
            last_time = tm.group(1)
        m = VAL_RE.search(line)
        if not m:
            continue
        ep, mAP, m50, m75, ms, mm, ml = m.groups()
        rows.append(
            {
                "epoch": int(ep),
                "time": last_time,
                "mAP": float(mAP),
                "mAP50": float(m50),
                "mAP75": float(m75),
                "mAP_s": float(ms) if ms != "-1.000" else None,
                "mAP_m": float(mm) if mm != "-1.000" else None,
                "mAP_l": float(ml) if ml != "-1.000" else None,
            }
        )

    if not rows:
        sys.exit("no Epoch(val) bbox_mAP lines found in log")

    print(f"# THAF training progress — {log_path.name}\n")
    print(f"Parsed **{len(rows)}** epoch(s).\n")

    best = max(rows, key=lambda r: r["mAP"])
    print(f"Best so far: **ep{best['epoch']} mAP={best['mAP']:.4f}** at {best['time']}\n")

    headers = ["ep", "wall time", "mAP", "mAP50", "mAP75", "Δ vs prev"]
    if args.baseline:
        headers.append("dev30 baseline")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---:" if h not in ("wall time",) else "---" for h in headers]) + "|")

    prev = None
    baseline = [float(x) for x in args.baseline] if args.baseline else None
    for r in rows:
        delta = f"{r['mAP'] - prev:+.4f}" if prev is not None else "—"
        cells = [
            str(r["epoch"]),
            r["time"],
            f"{r['mAP']:.4f}",
            f"{r['mAP50']:.4f}",
            f"{r['mAP75']:.4f}",
            delta,
        ]
        if baseline:
            idx = r["epoch"] - 1
            if 0 <= idx < len(baseline):
                cells.append(f"{baseline[idx]:.3f}")
            else:
                cells.append("—")
        print("| " + " | ".join(cells) + " |")
        prev = r["mAP"]


if __name__ == "__main__":
    main()
