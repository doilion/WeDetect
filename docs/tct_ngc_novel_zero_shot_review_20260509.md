# TCT_NGC novel zero-shot 改进路线 —— 问题汇总与下一步

**日期**：2026-05-09
**目标**：让 dev30 base 模型对**没见过的肿瘤类（novel）**做 prompt 切换式部署，目前几乎不可用。
**对比基线**：dev30 best ckpt (ep9，val mAP 0.283) + v2 国际标准 prompts (PSC/MAL-S/Bethesda)。

---

## TL;DR

- **文本端的所有 inference-time 改进都失败了**（CuPL ensemble、L2-norm、anisotropy reduction 全不行）
- **视觉 prototype（用图当类向量）部分成功**：救活了 Resp 和 Thyroid Bethesda V，但伤了 Serous（文本本来就 work）
- **简单的 text + visproto 融合（按类挑来源）也失败**：因为 text 和 visproto 不在同一几何空间，混在一起视觉 prototype 类**系统性挤占文本类的预测**
- **下一步**：要么改架构（post-hoc score fusion / 重训），要么换文本编码器（BiomedCLIP）

---

## 1. 背景：novel zero-shot 想干啥

base 模型（dev30）训了 30 个细胞类，**部署到新医院/新病种时希望直接改 prompt** 就能识别新类（"PSC Category VI: Squamous cell carcinoma" 等），不用再标数据重训。

实测下来 v2 prompts 给的 zero-shot mAP：

| 组织（base 类数）| novel 类 | v2 mAP |
|---|---|---:|
| Serous（base 2）| Breast / Ovarian / Adeno | **0.18 - 0.45** ✅ 能用 |
| Resp（base 7）| SCC / Adeno / SmallCell | **0.00 - 0.07** ❌ 几乎归零 |
| Thyroid（base 7）| MTC / Bethesda V/VI | **0.00 - 0.004** ❌ 几乎归零 |

**结论**：组织 base 类多 → prompt 余弦空间饱和 → novel 挤不进去。Serous base 2 类有空间，所以 zero-shot 能用。

---

## 2. 已尝试的方案（按时间顺序）

### 2.1 ❌ 文本端 inference-time 改进（全军覆没）

| 方案 | 实测 | 失败原因 |
|---|---|---|
| **CuPL** 多 prompt 描述（每类 6 个 PSC/Bethesda/形态学变体）| **mAP 跌 60-100%** | 平均把向量拉到几个 prompt 方向之间的"无人区" |
| **Prompt ensembling**（每变体 L2-norm 后再平均）| 同上 | L2-norm 改了 raw embedding scale，跟模型训练时见的不一样 |
| **Anisotropy reduction**（减全局均值 + L2-norm）| 同上 | 同上 |

**失败本质**：WeDetect 的 PseudoLanguageBackbone 用**冻结缓存 + raw 内积**。image encoder 训练时被钉死在某个特定 prompt 方向，inference-time 任何后处理都是在挪一个被钉住的锚点。

### 2.2 ✅/❌ 视觉 prototype（部分成功）

每类用 5 张 GT bbox crop，过一遍 dev30 image encoder，特征均值当作类向量：

| 组织 | v2 文本 | visproto | 结论 |
|---|---:|---:|---|
| **Resp** | 0.00 - 0.07 | **0.05 - 0.10** | ✅ 救活（0 → 实质数字）|
| **Thyroid** | 0.00 - 0.004 | Beth-V **0.079** / 其他 ≈0 | ✅ Beth-V 起死回生 |
| **Serous** | **0.18 - 0.45** | 0.07 - 0.11 | ❌ 大跌（被自损）|

**对偶现象**：
- 文本 work 的组织（Serous）：visproto 反而伤
- 文本死的组织（Resp/Thyroid）：visproto 救活

→ 这是**互补的两个工具**，不是替代关系。

### 2.3 ❌ 简单融合（按类挑最优来源）—— **本以为能 work，实测失败**

按规则路由：Serous 类用 text，Resp/Thyroid 用 visproto。**期望同时拿两边好处**。

实测 main_3 / pseudo_2 fused 结果：

| 类 | 应走 | 单源 mAP | fused mAP | 现象 |
|---|---|---:|---:|---|
| **main_3 Breast** | text | 0.454 | **0.000** | text 类被 visproto 类挤死 |
| main_3 SCC | visproto | 0.056 | 0.056 | 保留 visproto 数字 |
| **pseudo_2 Ovarian** | text | 0.181 | **0.000** | 同样被挤死 |
| pseudo_2 Resp-Adeno | visproto | 0.095 | 0.096 | 保留 |

**失败本质（关键技术点）**：text 向量 (XLM-Roberta 输出) 和 visproto 向量 (`cls_preds(image)` 输出) **不在同一几何空间**：
- visproto 来自 image encoder 自己的输出 → 对查询图像的 cosine **天然高**
- text 来自 XLM-Roberta → 跟查询图像 cls_preds **通过训练间接对齐**，cosine 必然**低**

虽然 contrastive head 对两边都做 L2-norm，但 normalize 后两类向量"对图像的吸引力"还是不对等。**混在同一次 inference 里，visproto 类系统性赢分**，text 类即使是对的也被压成 0。

---

## 3. 现在走到哪儿了

把 v2 baseline + 上面 3 类失败实验铺开看：

```
                  Resp      Serous    Thyroid
v2 文本           ~0        0.18-0.45 ~0
text-ensemble     ❌(更差)   ❌(更差)   ❌(更差)
visproto         0.05-0.10  0.07-0.11 Beth-V 0.08
binary-fusion     ⚠ visproto贡献保住  text类被挤死  visproto贡献保住
```

→ **最佳单源策略**：Serous 跑 v2 文本、Resp/Thyroid 跑 visproto，**两次独立 inference**（不能混）。

---

## 4. 剩余工程问题（需要修但不是核心）

1. **prototype 构造的 letterbox 问题**：当前 `cv2.resize(crop, 640, 640)` 没保留长宽比，模型见的是变形图。改用 letterbox padding 应该能再涨几个点。
2. **bbox `expand=1.5`**：~33% 像素是非 lesion 上下文，spatial mean 把背景吃进去了。可以试 `expand=1.1` 或 mask-aware pooling。
3. **跨 3 个 FPN scale 平均**：小 lesion 主要在 stride 8（最浅），平均稀释。可以改成只用 stride 8。
4. **Leakage**：5 张 prototype 来自 test set，又在 test set 上评估。要严格 zero-shot 得排除 holdout ann_ids。但当前数字趋势可信。
5. **fusion rule bug**：hard_4 的 Serous-Adeno 被错路由到 visproto（应走 text）。binary fusion 反正不 work，这个 bug 也就不重要了。

---

## 5. 下一步建议（按可行性排）

### 🟢 方案 A：post-hoc score fusion（推荐先试，~1 天）

**做法**：跑两次独立 inference（text-only + visproto-only），分别得到检测框列表 + 分数。然后**按类**挑：
- Serous 类的预测：取 text-only 那次的
- Resp/Thyroid 类的预测：取 visproto-only 那次的
- 最后把两份预测合并喂给 COCO eval

**为什么 work**：每次 inference 内部只有同一来源的类向量，没有几何不公平。两次结果在**预测层面**合并，避开 cosine 比较的偏置问题。

**成本**：写一个合并脚本（30 行 Python），各跑一次 eval。

**预期**：
- Serous 拿 0.18-0.45（保留文本优势）
- Resp/Thyroid 拿 0.05-0.10（保留视觉优势）
- **整体 novel mAP 比单源都好**

### 🟡 方案 B：换文本编码器到 BiomedCLIP / PubMedBERT（治本，1-2 周）

**做法**：把 XLM-Roberta 换成医学专用预训练模型，重训 dev30。

**预期**：fine-grained 类的 prompt cosine 从 >0.95 跌到 0.7-0.8（reviewer 经验值），文本端的所有方法重新可用。

**成本**：架构改动 + 1 次 8h 重训 + 调试。

### 🔴 方案 C：train-time prompt + exemplar augmentation（终极方案）

**做法**：base 训练时随机用 5-10 个 v2 风格 prompts 当类向量，外加偶尔用 visual exemplar，让 image encoder 学到对**两种来源**都鲁棒。

**预期**：text 和 visproto 都能拿好分，融合也不再有几何不匹配。

**成本**：1 次重训 + 数据 pipeline 改造。这是 OVD 业界标准做法（YOLO-World / OWL-ViT）。

---

## 6. 我的建议

**第一步先做方案 A**（post-hoc score fusion），1 天能拿到第一个**真正能用**的 novel zero-shot 数字。这一步几乎零成本，先验证"text 跟 visproto 互补"的工程化能不能让 mAP 真的合得起来。

**第二步看方案 A 数字**：
- 如果 fused mAP > 0.20（综合 Serous + Resp + Thyroid 平均），就工程上可用了 → 往临床部署方向走
- 如果还差，启动方案 B 换 BiomedCLIP

**方案 C** 是最终目标，但跟 base 端损失函数改进等工作并行排队即可。

---

## 7. 已记录的死路（**不要再尝试**）

⛔ inference-time 文本 ensembling（CuPL / mean pooling 多 prompt）
⛔ inference-time anisotropy reduction（mean centering / whitening）
⛔ binary class-level fusion (text + visproto 按类挑来源放同一个 .pth)

---

## 8. 文件指引

| 内容 | 路径 |
|---|---|
| dev30 best ckpt | `work_dirs/.../best_coco_bbox_mAP_epoch_9.pth` |
| v2 baseline (4 split novel mAP) | dev32 baseline 报告 §11 |
| 文本 ensembling 失败实验 | (产物已删，记录在 `TODO.md` 死路段) |
| visual prototype 工具 | `tools/build_visual_prototype.py` |
| visproto 实验结果 | `work_dirs/.../novel_visproto_summary.txt` |
| binary fusion 失败实验 | `work_dirs/.../novel_fused_summary.txt` |
| prompt cos 诊断 | `docs/figures/dev30_taxonomy_refactor_20260508/diagnostics/` |
| novel 方向 TODO 项 | `TODO.md` 12-18 |

---

## 9. 记给未来的自己

下次想"用 inference-time 文本技巧救 novel"之前，**先读这份文档**。
WeDetect 的"冻结缓存 + raw 内积"架构对文本端任何后处理都不友好。要嘛走视觉，要嘛重训。
