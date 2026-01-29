# 🧪 实验计划: 文本描述与数据增强优化 (Text & Augmentation Optimization)

## 📅 实验目标

验证 **英文全称文本描述** 和 **强数据增强** 对 TCT_NGC 开放词汇检测性能的影响。

## 📊 实验矩阵 (2×2 消融实验)

基于 **exp3 (Loss调整)** 配置作为基准：
- **Loss Cls**: 1.0
- **Loss BBox**: 5.0
- **Backbone**: 不冻结
- **Epochs**: 12

| 实验ID | 文本描述 | 数据增强 | GPU | 预期效果 |
|--------|---------|---------|-----|---------|
| **exp3** (基准) | 中文简称 | 弱 (仅RandomFlip) | - | Base: 29.6%, Novel: 9.1% |
| **exp5** | **英文全称 + 多别名** | 弱 (仅RandomFlip) | GPU 0 | Novel 提升 (预期 >10%) |
| **exp6** | 中文简称 | **强 (颜色+几何)** | GPU 1 | Base 提升 (预期 >31%) |
| **exp7** | **英文全称 + 多别名** | **强 (颜色+几何)** | GPU 3 | 双重提升 (叠加效应) |

---

## 📝 详细配置

### 1. 文本描述 (Text Description)

#### 中文简称 (exp3 / exp6)
- 文件: `data/texts/tct_ngc_v2_base_class_texts.json`
- 示例: `["正常细胞", "正常"]`, `["非典型鳞状细胞", "ASCUS"]`

#### 英文全称 + 多别名 (exp5 / exp7)
- 文件: `data/texts/tct_ngc_v2_base_class_texts_en.json` (新创建)
- 示例: 
  - `["cervical normal cells", "normal cervical epithelial cells", "benign cervical cells"]`
  - `["cervical atypical squamous cells of undetermined significance", "cervical ASCUS", "atypical squamous cells"]`

### 2. 数据增强 (Data Augmentation)

#### 弱增强 (exp3 / exp5)
```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='WeDetectKeepRatioResize', scale=(640, 640)),
    dict(type='WeDetectLetterResize', scale=(640, 640), ...),
    dict(type='RandomFlip', prob=0.5),  # 仅水平翻转
    dict(type='LoadText'),
    dict(type='PackDetInputs', ...),
]
```

#### 强增强 (exp6 / exp7)
```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    
    # === 颜色增强 (新增) ===
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2),
         hue_delta=10),
    
    dict(type='WeDetectKeepRatioResize', scale=(640, 640)),
    dict(type='WeDetectLetterResize', scale=(640, 640), ...),
    
    # === 几何增强 (新增垂直翻转) ===
    dict(type='RandomFlip', prob=0.5),                        # 水平翻转
    dict(type='RandomFlip', prob=0.5, direction='vertical'),  # 垂直翻转
    
    dict(type='LoadText'),
    dict(type='PackDetInputs', ...),
]
```

---

## 🚀 执行步骤

1. **准备文本文件**
   - 创建 `data/texts/tct_ngc_v2_base_class_texts_en.json`
   - 创建 `data/texts/tct_ngc_v2_class_texts_en.json`

2. **准备配置文件**
   - `config/wedetect_tiny_tct_exp5.py`
   - `config/wedetect_tiny_tct_exp6.py`
   - `config/wedetect_tiny_tct_exp7.py`

3. **并行训练**
   ```bash
   # Terminal 1 (GPU 0)
   CUDA_VISIBLE_DEVICES=0 python train.py config/wedetect_tiny_tct_exp5.py
   
   # Terminal 2 (GPU 1)
   CUDA_VISIBLE_DEVICES=1 python train.py config/wedetect_tiny_tct_exp6.py
   
   # Terminal 3 (GPU 3)
   CUDA_VISIBLE_DEVICES=3 python train.py config/wedetect_tiny_tct_exp7.py
   ```

4. **评估与对比**
   - 运行 `test_exclude_negative.py` 评估 Base mAP
   - 运行 `eval_novel_exp*.py` 评估 Novel mAP
   - 汇总结果到 `docs/EXPERIMENTS_TABLE.md`

