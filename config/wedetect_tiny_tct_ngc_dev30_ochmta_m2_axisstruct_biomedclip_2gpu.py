_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m2_biomedclip_2gpu.py"]

# Row 5: OCHMTA + axis 结构损失
#
# 在 M1+M2-完整方法 之上加 (organ, axis)-conditional structure loss：
#   - 同 (organ, axis) 拉近：penalize cos < 0.5
#   - 跨 organ 推开：penalize cos > 0.1
#   - 同 organ 异 axis 不管：no penalty（保留 cross-organ-kin 类的 cos）
#
# 动机：M1+M2-完整方法 在 val 上 macro 0.3085 < M1-1PSC 0.3137，per-class delta
# 显示 M2 砍掉了"跨组织共享形态学"的类（甲状腺-巨噬细胞 −0.077，呼吸道-淋巴
# 细胞 −0.029 等）。原因是 Stage 2 organ MoE 把它们强行路由到不同 expert，
# 输出 cos ≈ 0.2，导致语义近邻信息丢失。
#
# 设计避开 P1.1：不强迫 within-organ 整体拉近（那会把 PTC 和巨噬细胞这种"同
# organ 但完全不同 axis"的类强制接近，更糟）；只拉近同 (organ, axis) 内的
# 类（PTC↔FC 这种同 Malignant axis 的 rank ladder），跨 organ 推开（无关类
# 分离），同 organ 异 axis 留给架构自己决定（保留 organ 内细分能力）。
#
# 训练 12 ep，从 noTHAF ckpt（同 M1/M1+M2），不在 M2 ckpt 上 fine-tune
# 以保证跟 M1+M2 fair 对比。

model = dict(
    backbone=dict(
        text_model=dict(
            adapter=dict(
                # 保留 M2 原有 5 个 aux loss 配置
                lambda_pool_entropy=0.02,
                lambda_proj_drift=0.001,
                lambda_gate_entropy=0.02,
                lambda_rank_norm=0.01,
                # 新增 Row 5 结构损失（默认 0，这里开启）
                lambda_axis_attract=0.3,
                lambda_cross_organ_repel=0.3,
                axis_attract_target=0.5,
                cross_organ_repel_target=0.1,
            ),
        ),
    ),
    # In-flight 训练 (started 2026-05-14 12:06) 跑的是 OrganOrdinalLoss
    # working-tree 版本 (SUM normalization)。现在 normalization 已加 flag
    # 默认 'mean'，这里显式标 'sum' 以复现 in-flight ckpt 的训练动力学。
    ordinal_loss=dict(
        normalization='sum',
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m2_axisstruct_biomedclip_2gpu"
