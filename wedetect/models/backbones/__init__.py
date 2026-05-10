# Copyright (c) Tencent Inc. All rights reserved.
# YOLO Multi-Modal Backbone (Vision Language)
# Vision: YOLOv8 CSPDarknet
# Language: CLIP Text Encoder (12-layer transformer)
from .mm_backbone import (
    MultiModalYOLOBackbone,
    HuggingVisionBackbone,
    HuggingCLIPLanguageBackbone,
    PseudoLanguageBackbone,
    XLMRobertaLanguageBackbone,
    ConvNextVisionBackbone,
    HuggingCLIPVisionBackbone,
    )
from .hierarchical_mm_backbone import (
    HierarchicalXLMRLanguageBackbone,
    PseudoHierarchicalXLMRLanguageBackbone,
)
from .biomedclip_backbone import (
    HierarchicalBiomedCLIPLanguageBackbone,
    PseudoHierarchicalBiomedCLIPLanguageBackbone,
)

__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'PseudoLanguageBackbone',
    'ConvNextVisionBackbone',
    'XLMRobertaLanguageBackbone',
    'HuggingCLIPVisionBackbone',
    'HierarchicalXLMRLanguageBackbone',
    'PseudoHierarchicalXLMRLanguageBackbone',
    'HierarchicalBiomedCLIPLanguageBackbone',
    'PseudoHierarchicalBiomedCLIPLanguageBackbone',
]
