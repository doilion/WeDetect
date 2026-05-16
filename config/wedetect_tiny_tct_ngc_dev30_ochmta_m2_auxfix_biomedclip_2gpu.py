_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m2_biomedclip_2gpu.py"]

# Row 4d-fix v2: M1+M2-完整方法 + 修复后的 aux loss
#
# 实证发现（P0.2）：M2 baseline 的 5 个 aux loss 在 12 ep 训练里全部失效：
#   - loss_pool_entropy 恒定 0.0311（实际 entropy 1.555 顶满 log5=1.609）
#       → λ=0.02 太小，被 loss_cls (~290) 完全淹没
#   - loss_proj_drift = 0（drift > 0.05 阈值，relu guard 永远不启动）→ 正常，不动
#   - loss_gate_entropy = 0.0032（也顶满但量级小，且被 gate_prior_strength=5 主导）
#   - loss_rank_norm = 0（norm > 0.05 阈值）→ 正常，不动
#   - loss_ord ≈ 0.0003（MSE 在 axis-mean 归一化下压死，比 loss_cls 小 1e6 倍）
#
# 修复策略：两个真正生效的改动（v2 conservative，v1 1.0/1.0 数值不稳崩在 iter 1）
#   1. λ_pool_entropy: 0.02 → 0.3 (15×)
#      → 强制 stage 1 attention entropy 下降，让 attention 真正选择性
#   2. OrganOrdinalLoss: normalization='sum' (替代 baseline 默认 'mean')
#      → 6 active axes 累加而不平均，让 loss_ord 量级跳 6×
#      → ord loss_weight 仍保留 0.3，配合 sum 后 loss_ord 初始 ~5-10
#        (loss_cls ~300，ratio ~1:30，比 baseline 1:300000 提升 4 个数量级)
#
# Stage 1 proj_drift / Stage 3 rank_norm 保持原值（guard，正常 0 不需要 boost）。
# Stage 2 gate_entropy 保持（被 gate_prior_strength=5 主导，调 λ 没用）。
#
# 训练 12 ep，从 noTHAF ckpt（同 M2），fair 对比。
#
# 实证 (post-train ep12)：
#   - base val 0.3136 vs M2 baseline 0.3085 (+0.51pp) ✓ aux loss 真正生效
#   - stage 1 entropy 从 ~1.51 (ep1) 压到 ~0.82 (ep12)，attention 学会选择性 ✓
#   - loss_ord 从 ~5 量级稳定下降到 ~1，rank head 学到 ladder ✓
#   - novel macro 0.0557 vs M2 baseline 0.1051 (−4.9pp) ⚠ adapter 过拟合
#     base manifold，novel zero-shot 走丢（routing 越选择性 novel 越死）

model = dict(
    backbone=dict(
        text_model=dict(
            adapter=dict(
                # 关键改动 1：lambda_pool_entropy 从 0.02 翻 15× 到 0.3
                lambda_pool_entropy=0.3,
                # 其他保持
                lambda_proj_drift=0.001,
                lambda_gate_entropy=0.02,
                lambda_rank_norm=0.01,
                # 不开 Row 5 的 axis 结构损失（独立 ablation）
                lambda_axis_attract=0.0,
                lambda_cross_organ_repel=0.0,
            ),
        ),
    ),
    ordinal_loss=dict(
        # 关键改动 2：normalization='sum' 让 loss_ord 量级 6× 跳起来
        # 6 active axes × 平均 MSE ~5 = ~30 → ×0.3 = ~9 → loss_cls~300，ratio 1:33
        # 训练完 ep12 loss_ord 收敛到 ~1（ladder 学好后 MSE 自然下降）
        loss_weight=0.3,
        monotonicity_weight=0.5,
        normalization='sum',
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m2_auxfix_biomedclip_2gpu"
