# Copyright (c) Tencent Inc. All rights reserved.
from .dynamic_loss import CoVMSELoss
from .iou_loss import mmyoloIoULoss
from .organ_ordinal_loss import OrganOrdinalLoss

__all__ = ['CoVMSELoss', 'OrganOrdinalLoss']
