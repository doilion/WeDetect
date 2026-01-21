# Experiments Index (TCT_NGC V2)

This file summarizes the five TCT_NGC V2 experiments recorded in
`docs/TCT_NGC_V2_实验总结.md`.

| # | Experiment | Config | Best Epoch | Base mAP (exclude negatives) | Novel mAP | Checkpoint | Notes |
|---|------------|--------|-----------:|------------------------------:|----------:|------------|-------|
| 1 | Baseline | `config/wedetect_tiny_tct.py` | 9 | 32.1% | 6.8% | missing | Trained on RTX 5880, checkpoint not copied here |
| 2 | LR down | `config/wedetect_tiny_tct.py` | 9 | 30.7% | 7.2% | missing | Trained on RTX 5880, checkpoint not copied here |
| 3 | Freeze backbone | `config/wedetect_tiny_tct_exp2.py` | 8 | 28.4% | 8.1% | `work_dirs/wedetect_tiny_tct_exp2/best_coco_bbox_mAP_epoch_8.pth` | Frozen first two stages |
| 4 | Loss tuning | `config/wedetect_tiny_tct_exp3.py` | 9 | 29.6% | 9.1% | `work_dirs/wedetect_tiny_tct_exp3/best_coco_bbox_mAP_epoch_9.pth` | cls=1.0, bbox=5.0 |
| 5 | Freeze + loss | `config/wedetect_tiny_tct_exp4.py` | 7 | 24.3% | 9.0% | `work_dirs/wedetect_tiny_tct_exp4/best_coco_bbox_mAP_epoch_7.pth` | Combination strategy |

## Eval outputs

- `work_dirs/eval/test_exclude_negative/`
- `work_dirs/eval/test_novel_exp2/`
- `work_dirs/eval/test_novel_exp3/`
- `work_dirs/eval/test_novel_exp4/`
- `work_dirs/eval/eval_exp4_base.log`
- `work_dirs/eval/eval_exp4_novel.log`

## Training logs

- `work_dirs/logs/exp2_train.log`
- `work_dirs/logs/exp3_train.log`
- `work_dirs/logs/exp4_train.log`
