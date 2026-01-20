"""
src/models.py
This module defines the data models used in the application.
BoMeyering 2026
"""

# Import statements
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork


class SwinTFeatureExtractor(nn.Module):
    def __init__(self, weights=Swin_T_Weights.IMAGENET1K_V1, trainable=True):
        super().__init__()
        m = swin_t(weights=weights)

        self.body = create_feature_extractor(
            m,
            return_nodes={
                "features.1": "c2",
                "features.3": "c3",
                "features.5": "c4",
                "features.7": "c5",
            }
        )

    def forward(self, x):
        feats = self.body(x)  # dict of BHWC tensors
        out = OrderedDict()

        # BHWC -> NCHW for downstream FPN/detectors
        for k in ("c2", "c3", "c4", "c5"):
            v = feats[k]
            out[k] = v.permute(0, 3, 1, 2).contiguous()
        return out

class SwinFPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.backbone = SwinTFeatureExtractor()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[96, 192, 384, 768],
            out_channels=out_channels,
        )
        self.out_channels = out_channels

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.fpn(feats)

        return feats

def retinanet_swin(num_classes=2, backbone_out_channels=256):
    backbone = SwinFPN(out_channels=backbone_out_channels)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),   # one per level
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    model = RetinaNet(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
    )

    model.detections_per_img = 100

    return model
