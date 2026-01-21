# work_dirs layout

This directory holds experiment outputs. The layout is standardized as:

- `work_dirs/experiments/`
  - `exp2_freeze_backbone` -> `../wedetect_tiny_tct_exp2`
  - `exp3_loss_tuning` -> `../wedetect_tiny_tct_exp3`
  - `exp4_freeze_loss_combo` -> `../wedetect_tiny_tct_exp4`
- `work_dirs/eval/`: evaluation artifacts and logs
- `work_dirs/logs/`: consolidated training logs

Raw experiment outputs remain in:

- `work_dirs/wedetect_tiny_tct_exp2/`
- `work_dirs/wedetect_tiny_tct_exp3/`
- `work_dirs/wedetect_tiny_tct_exp4/`
