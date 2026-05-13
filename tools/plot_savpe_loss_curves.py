#!/usr/bin/env python
"""Plot SAVPE-v1 / SAVPE-v2 training loss curves.

Parses the custom log format from `tools/train_savpe_cell_contrastive.py`
(v1) and `tools/train_savpe_v2_aligned.py` (v2 with align/cross losses).

v1 line format:
  [ep 1/3] step 50/2418  loss=0.0078  avg=0.0134  valid_cls=112/1920  pos_rate=0.0212  lr=4.00e-03

v2 line format:
  [ep 1/3] step 50/2418  total=0.2561  align=0.1965  cell=0.0047  cross=0.5495  lr=4.00e-03

Outputs a single PNG plotting one curve per loss component (and cos(vis,text)
when v2 align is present, derived from cos = 1 − L_align/2).

Usage:
  python tools/plot_savpe_loss_curves.py \\
      --log work_dirs/savpe_v2_aligned_v1_lambda10_collapse/train.log \\
      --label "SAVPE-v2 λ=1.0" \\
      --out docs/tct_ngc_experiment_journey_figures/12_savpe_v2_lambda10_loss.png

  # Side-by-side comparison:
  python tools/plot_savpe_loss_curves.py \\
      --log work_dirs/savpe_v2_aligned_v1_lambda10_collapse/train.log \\
            work_dirs/savpe_v2_aligned_lambda03/train.log \\
      --label "λ=1.0" "λ=0.3" \\
      --out docs/tct_ngc_experiment_journey_figures/13_savpe_v2_compare.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


V2_RE = re.compile(
    r"\[ep\s+(?P<ep>\d+)/(?P<eps>\d+)\]\s+step\s+(?P<step>\d+)/(?P<total>\d+)\s+"
    r"total=(?P<total_loss>[\d.]+)\s+align=(?P<align>[\d.]+)\s+"
    r"cell=(?P<cell>[\d.]+)\s+cross=(?P<cross>[\d.]+)\s+lr=(?P<lr>[\d.eE+-]+)"
)

V1_RE = re.compile(
    r"\[ep\s+(?P<ep>\d+)/(?P<eps>\d+)\]\s+step\s+(?P<step>\d+)/(?P<total>\d+)\s+"
    r"loss=(?P<loss>[\d.]+)\s+avg=(?P<avg>[\d.]+).*?lr=(?P<lr>[\d.eE+-]+)"
)


def parse_log(log_path: Path) -> dict:
    steps = []
    rows = {}
    is_v2 = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m2 = V2_RE.search(line)
        if m2:
            is_v2 = True
            steps.append(int(m2["step"]))
            rows.setdefault("total", []).append(float(m2["total_loss"]))
            rows.setdefault("align", []).append(float(m2["align"]))
            rows.setdefault("cell", []).append(float(m2["cell"]))
            rows.setdefault("cross", []).append(float(m2["cross"]))
            rows.setdefault("lr", []).append(float(m2["lr"]))
            continue
        m1 = V1_RE.search(line)
        if m1:
            is_v2 = False
            steps.append(int(m1["step"]))
            rows.setdefault("loss", []).append(float(m1["loss"]))
            rows.setdefault("avg", []).append(float(m1["avg"]))
            rows.setdefault("lr", []).append(float(m1["lr"]))
    return dict(steps=steps, rows=rows, is_v2=is_v2)


def plot_one(log_path: Path, label: str, ax_loss, ax_cos: Optional[plt.Axes]) -> None:
    data = parse_log(log_path)
    steps = data["steps"]
    rows = data["rows"]
    if not steps:
        print(f"[skip] {log_path}: no parseable rows")
        return

    if data["is_v2"]:
        ax_loss.plot(steps, rows["total"], label=f"{label} total", lw=2.0)
        ax_loss.plot(steps, rows["align"], label=f"{label} L_align", lw=1.2, ls="--")
        ax_loss.plot(steps, rows["cell"], label=f"{label} L_cell", lw=1.2, ls=":")
        ax_loss.plot(steps, rows["cross"], label=f"{label} L_cross", lw=1.2, ls="-.")
        if ax_cos is not None:
            cos = [1.0 - a / 2.0 for a in rows["align"]]
            ax_cos.plot(steps, cos, label=f"{label} cos(vis, text)", lw=2.0)
    else:
        ax_loss.plot(steps, rows["loss"], label=f"{label} per-step loss", lw=1.0, alpha=0.4)
        ax_loss.plot(steps, rows["avg"], label=f"{label} running avg", lw=2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", required=True,
                    help="One or more train.log paths")
    ap.add_argument("--label", nargs="+", required=True,
                    help="One label per --log (same order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="SAVPE training loss")
    args = ap.parse_args()

    if len(args.log) != len(args.label):
        raise SystemExit(f"--log count {len(args.log)} != --label count {len(args.label)}")

    has_v2 = any(parse_log(Path(p))["is_v2"] for p in args.log)

    if has_v2:
        fig, (ax_loss, ax_cos) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    else:
        fig, ax_loss = plt.subplots(1, 1, figsize=(11, 4))
        ax_cos = None

    for log_path, label in zip(args.log, args.label):
        plot_one(Path(log_path), label, ax_loss, ax_cos)

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(args.title)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right", fontsize=8, ncol=2)

    if ax_cos is not None:
        ax_cos.set_xlabel("step")
        ax_cos.set_ylabel("cos(vis_emb, text_emb)")
        ax_cos.set_title("Cross-modal alignment: cos = 1 − L_align/2 (1.0 = collapsed, ≤0.9 = healthy)")
        ax_cos.axhline(0.95, color="red", ls="--", alpha=0.5, label="collapse threshold 0.95")
        ax_cos.axhline(0.5, color="green", ls="--", alpha=0.5, label="weak alignment 0.5")
        ax_cos.grid(True, alpha=0.3)
        ax_cos.legend(loc="lower right", fontsize=8)
        ax_cos.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
