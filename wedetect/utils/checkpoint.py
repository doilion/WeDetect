from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_latest_checkpoint(
    checkpoint: Optional[str],
    work_dir: str | Path,
) -> str:
    """Return ``checkpoint`` if given, else newest ``best_*`` (or ``epoch_*``) ckpt under ``work_dir``."""
    if checkpoint:
        return checkpoint

    work_path = Path(work_dir)
    candidates = sorted(
        work_path.glob("best_coco_bbox_mAP_epoch_*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            work_path.glob("epoch_*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {work_path}; pass --checkpoint explicitly."
        )
    return str(candidates[0])
