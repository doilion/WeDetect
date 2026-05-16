from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_latest_checkpoint(
    checkpoint: Optional[str],
    work_dir: str | Path,
) -> str:
    """Return ``checkpoint`` if given, else newest ``best_*`` (or ``epoch_*``)
    ckpt under ``work_dir``.

    Search priority (each tier sorted newest-first by mtime):
      1. ``best_coco_overall_macro_mAP_epoch_*.pth`` (corrected-protocol best,
         emitted by trainings whose CheckpointHook uses
         ``save_best='coco/overall/macro_mAP'``: M1, M2, axisstruct, ICF, ...)
      2. ``best_coco_bbox_mAP_epoch_*.pth`` (legacy CocoMetric best,
         emitted by older trainings like clean dev30, noTHAF, THAF, PCW)
      3. ``epoch_*.pth`` (any per-epoch dump as last resort)
    """
    if checkpoint:
        return checkpoint

    work_path = Path(work_dir)
    for pattern in (
        "best_coco_overall_macro_mAP_epoch_*.pth",
        "best_coco_bbox_mAP_epoch_*.pth",
        "epoch_*.pth",
    ):
        candidates = sorted(
            work_path.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError(
        f"No checkpoint found under {work_path}; pass --checkpoint explicitly."
    )
