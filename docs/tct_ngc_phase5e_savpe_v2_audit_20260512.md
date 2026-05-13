# Phase 5e SAVPE-v2 实现思路审查（2026-05-12）

启动 λ_align=1.0 训练之前的一次 review。目的是把潜在风险落地、把可修的代码 bug 修掉，
设计层面的风险（坍缩到 text-only）留到实测之后再决策。

## 🔴 设计风险（不修，留作实测对照）

### R1. L_align 与 L_cell 同向叠加 → vis_emb 完全坍缩到 text_emb 的风险
- 头是 `score = BN(cell) · L2(class_vec) · scale + bias`。训练时 class_vec=text_emb。
- L_cell（focal_bce）让 cell-inside-bbox score 高 → 隐式逼 vis_emb → text_emb 方向。
- L_align 是显式 MSE 拉 vis_emb → text_emb。两个力同向叠加。
- 一旦 cos(vis_emb, text_emb) → 1.0，novel 退化为 text-only baseline (avg 0.004)。
- inference-only baseline 0.105 之所以能用，恰因 vis_emb ≠ text_emb，保留 image-side info。

**对照实验设计**：先用 λ_align=1.0 跑完 3 epoch，跑 cos diagnostic + 全套 eval 看坍缩程度。
- cos ∈ [0.6, 0.9] → λ 合适
- cos < 0.5 → λ 应该升
- cos > 0.95 → λ 应该降到 0.3 重训

### R2. SAVPE 只见 base 30 类，novel 类的训练信号为零
- valid_mask 只在 base 类 = 1，SAVPE 收敛后对 novel 类只能用 share 的 sem/act/attn conv 泛化。
- 风险：SAVPE 收敛成 "FPN-feat → text_emb lookup table" → novel 上输出 "最近 base 类的 text_emb"。
- L_cross λ=0.1 是 mitigation，但权重小。
- **post-train 必查**：vis_emb 30 类之间 cos 直方图，应该散开（中位 < 0.5）。

### R3. 冻结的 cls_preds 让 L_cell 的优化 landscape 变 hostile
- SAVPE 实际上是在 invert 一个固定的高维 BNContrastiveHead 映射。
- 跟 YOLOE 不同（YOLOE 是 head + SAVPE 一起 train）。
- 风险体现：L_cell 可能很快卡 plateau，L_align 主导。
- **观察点**：训练曲线 L_cell 在 epoch 1 后是否还下降。

## 🟡 实现细节问题（可修，本次修复）

### B1. ✅ FIXED — Pad math 已验证 pixel-perfect
跟 `wedetect/datasets/transformers/transforms.py:237-240` 一致。

### B2. CUDA_VISIBLE_DEVICES 必须强制 = 0,1
- yoloe Claude session 在 GPU 2+3（19614/19050 MiB occupied）
- 默认 torchrun + LOCAL_RANK 0..1 会占用 cuda:0、cuda:1
- 但 init_detector 在 `--sanity-only` mode 也会默认 `--device cuda:0`
- **launcher 必须在 env 里写死 `CUDA_VISIBLE_DEVICES=0,1`**

### B3. 缺 post-training cos 诊断工具
- 训练完无法快速判断是不是坍缩了
- 必须有 `tools/diagnose_savpe_v2_alignment.py`：
    - 对 base 30 类的 visproto 跑 cos(vis, text) 直方图
    - 输出 mean / median / per-class
    - 跑 cos(vis_i, vis_j) 30×30 矩阵看类间是否散开

### B4. VRAM 风险（不强制修）
- 主要瓶颈：SAVPE 内部 `[B*Q, 16, 80, 80]` 的 score_map softmax + cls logits `[B, 30, 80, 80] × 3`
- batch=64, Q=30, FP32 估算 ~3-5 GB activation，加 ConvNext frozen forward + 反传 vis_emb ≈ 15-18 GB
- 24 GB 卡应该够，但留 watch 余地。OOM 就回退 batch=32。

### B5. Drop_last + small train set 可能丢比例较大
- train ann 约 6k 张，batch=64 × 2 GPU effective=128，drop_last 丢 0-1 个 batch（~128 张）
- 占 train set 2% — 可接受

## 实测验证清单（训练后立即跑）

```bash
# 1. cos diagnostic
PYTHONPATH=. python tools/diagnose_savpe_v2_alignment.py \
    --savpe-ckpt work_dirs/savpe_v2_aligned_v1/savpe_final.pth \
    --base-config config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py \
    --base-ckpt work_dirs/.../best_*.pth \
    --visproto-pth data/texts/tct_ngc_base30_savpe_v2_aligned.pth \
    --text-cache data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth \
    --fullnames-json data/texts/tct_ngc_fullnames_30.json

# 2. 如果 cos 看着 OK (median 0.5-0.9)，跑完整 4 novel + base 25-cls eval
# 3. 如果 cos 直接 → 1.0，写下来"v2 λ=1.0 collapse confirmed"，重训 λ=0.3
```

## 退出标准

| 结果 | 行动 |
|---|---|
| cos median ∈ [0.5, 0.9] AND novel ≥ baseline + 0.02 | ✅ v2 success，写 paper |
| cos median > 0.95 AND novel < baseline | ❌ collapse confirmed，重训 λ=0.3 或加 reconstruction loss |
| cos median < 0.3 | L_align 没起作用，加大 λ 或换 loss 形式 |
| cos OK 但 novel < baseline | DEAD-7 image encoder 限制，升级 Phase 5f |
