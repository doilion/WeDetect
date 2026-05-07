#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FLOAT_RE = r"(?:[-+]?(?:nan|inf)|[0-9.eE+-]+)"


TRAIN_RE = re.compile(
    r"Epoch\(train\)\s+\[(?P<epoch>\d+)\]\[\s*(?P<iter>\d+)/(?P<iters>\d+)\].*?"
    rf"lr:\s*(?P<lr>{FLOAT_RE}).*?"
    rf"time:\s*(?P<time>{FLOAT_RE}).*?"
    rf"loss:\s*(?P<loss>{FLOAT_RE}).*?"
    rf"loss_cls:\s*(?P<loss_cls>{FLOAT_RE}).*?"
    rf"loss_bbox:\s*(?P<loss_bbox>{FLOAT_RE}).*?"
    rf"loss_dfl:\s*(?P<loss_dfl>{FLOAT_RE})"
)

VAL_RE = re.compile(
    r"Epoch\(val\).*?coco/bbox_mAP:\s*(?P<map>[0-9.eE+-]+).*?"
    r"coco/bbox_mAP_50:\s*(?P<map50>[0-9.eE+-]+)"
)


def parse_train_log(path: Path) -> tuple[list[dict], list[dict]]:
    train_by_position: dict[tuple[int, int], dict] = {}
    val_rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRAIN_RE.search(line)
        if match:
            row = match.groupdict()
            epoch = int(row["epoch"])
            it = int(row["iter"])
            iters = int(row["iters"])
            row["epoch"] = epoch
            row["iter"] = it
            row["iters"] = iters
            for key in ("lr", "time", "loss", "loss_cls", "loss_bbox", "loss_dfl"):
                row[key] = float(row[key])
            train_by_position[(epoch, it)] = row
            continue

        match = VAL_RE.search(line)
        if match:
            val_rows.append(
                {
                    "epoch": len(val_rows) + 1,
                    "bbox_mAP": float(match.group("map")),
                    "bbox_mAP_50": float(match.group("map50")),
                }
            )
    train_rows = sorted(
        train_by_position.values(), key=lambda row: (row["epoch"], row["iter"])
    )
    epoch_lengths: dict[int, int] = {}
    for row in train_rows:
        epoch_lengths[row["epoch"]] = max(epoch_lengths.get(row["epoch"], 0), row["iters"])
    epoch_offsets: dict[int, int] = {}
    offset = 0
    for epoch in sorted(epoch_lengths):
        epoch_offsets[epoch] = offset
        offset += epoch_lengths[epoch]
    for row in train_rows:
        row["global_iter"] = epoch_offsets[row["epoch"]] + row["iter"]
    return train_rows, val_rows


def read_val_loss(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            {
                key: int(value) if key == "epoch" else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(f)
        ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-loss-csv")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = parse_train_log(log_path)
    val_loss_rows = read_val_loss(Path(args.val_loss_csv) if args.val_loss_csv else None)

    write_csv(out_dir / "train_loss_from_log.csv", train_rows)
    write_csv(out_dir / "val_map_from_log.csv", val_rows)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    ax = axes[0]
    if train_rows:
        x = [row["global_iter"] for row in train_rows]
        ax.plot(x, [row["loss"] for row in train_rows], label="train loss", lw=1.8)
        ax.plot(x, [row["loss_cls"] for row in train_rows], label="train loss cls", lw=1.0)
        ax.plot(x, [row["loss_bbox"] for row in train_rows], label="train loss bbox", lw=1.0)
        ax.plot(x, [row["loss_dfl"] for row in train_rows], label="train loss dfl", lw=1.0)
    ax.set_title("Training Loss")
    ax.set_xlabel("global iter")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if val_loss_rows:
        epochs = [row["epoch"] for row in val_loss_rows]
        ax.plot(epochs, [row["loss"] for row in val_loss_rows], marker="o", label="val loss")
        ax.plot(epochs, [row["loss_cls"] for row in val_loss_rows], marker="o", label="val loss cls")
        ax.plot(epochs, [row["loss_bbox"] for row in val_loss_rows], marker="o", label="val loss bbox")
        ax.plot(epochs, [row["loss_dfl"] for row in val_loss_rows], marker="o", label="val loss dfl")
    if val_rows:
        ax2 = ax.twinx()
        epochs = [row["epoch"] for row in val_rows]
        ax2.plot(epochs, [row["bbox_mAP"] for row in val_rows], color="black", marker="x", label="bbox mAP")
        ax2.set_ylabel("mAP")
        ax2.legend(loc="lower right")
    ax.set_title("Validation Loss And mAP")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=180)
    print(f"wrote {out_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
