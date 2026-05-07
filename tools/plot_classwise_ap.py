#!/usr/bin/env python
"""Read MMEngine classwise eval log and plot per-class AP as a horizontal bar chart."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROW_RE = re.compile(
    r"^\|\s+(?P<name>[^|]+?)\s+\|\s+(?P<map>[0-9.]+|nan)\s+\|"
    r"\s+(?P<map50>[0-9.]+|nan)\s+\|\s+(?P<map75>[0-9.]+|nan)\s+\|"
)


SUPER_COLOR = {
    "respiratory tract": "#4C9AFF",
    "Serous effusion": "#7AC274",
    "Thyroid gland": "#F2A93B",
    "Urine": "#E0584C",
    "TCT_CCD": "#9E7BD0",
}


def supercategory(name: str) -> str:
    for prefix in SUPER_COLOR:
        if name.startswith(prefix):
            return prefix
    return "other"


def parse_log(path: Path) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if name == "category":
            continue
        try:
            ap = float(m.group("map"))
            ap50 = float(m.group("map50"))
        except ValueError:
            continue
        rows.append((name, ap, ap50))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="MMEngine eval .log with classwise table")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Per-class AP (epoch 12 best, exclude-negative)")
    args = parser.parse_args()

    rows = parse_log(Path(args.log))
    if not rows:
        raise SystemExit(f"No classwise rows parsed from {args.log}")

    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    aps = [r[1] for r in rows]
    aps50 = [r[2] for r in rows]
    colors = [SUPER_COLOR.get(supercategory(n), "#888") for n in names]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * len(rows))))
    y = range(len(rows))
    ax.barh(y, aps50, color=colors, alpha=0.35, label="mAP_50")
    ax.barh(y, aps, color=colors, alpha=0.95, label="mAP")
    for i, (ap, ap50) in enumerate(zip(aps, aps50)):
        ax.text(ap + 0.005, i, f"{ap:.3f}", va="center", fontsize=8)
        ax.text(ap50 + 0.005, i, f"({ap50:.2f})", va="center", fontsize=7, color="#555")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("AP")
    ax.set_xlim(0, max(max(aps50) + 0.10, 1.0))
    ax.set_title(args.title)
    ax.axvline(sum(aps) / len(aps), color="black", linestyle="--", linewidth=0.8,
               label=f"mean mAP = {sum(aps) / len(aps):.3f}")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in SUPER_COLOR.items()
    ]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=f"mean mAP = {sum(aps) / len(aps):.3f}"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out} ({len(rows)} classes, mean mAP={sum(aps) / len(aps):.4f})")


if __name__ == "__main__":
    main()
