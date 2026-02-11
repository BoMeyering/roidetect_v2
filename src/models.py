"""
src/models.py
This module defines the data models used in the application.
BoMeyering 2026
"""

# Import statements
import torch
import torchvision
import torch.nn as nn
from sys import exception
from collections import OrderedDict
from omegaconf import OmegaConf
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import create_model
from effdet.bench import DetBenchTrain, DetBenchPredict
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.detection import RetinaNet, fasterrcnn_resnet50_fpn_v2
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

def retinanet_swin(conf: OmegaConf) -> RetinaNet:
    """
    Creates a RetinaNet model with a Swin Transformer backbone.
    """

    backbone = SwinFPN(out_channels=conf.backbone_out_channels)

    anchor_generator = AnchorGenerator(
        sizes=((4,), (8,), (16,), (32,)),   # one per level
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    model = RetinaNet(
        backbone,
        num_classes=conf.num_classes,
        anchor_generator=anchor_generator,
        detections_per_img=conf.detections_per_img,
    )

    return model

def create_fasterrcnn(conf: OmegaConf) -> torch.nn.Module:
    """
    Creates a Faster R-CNN model with a Swin Transformer backbone.
    """

    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,)),   # one per level
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
        num_classes=conf.num_classes,
        anchor_generator=anchor_generator,
        box_detections_per_img=conf.detections_per_img,
    )

    return model

def create_effdet(conf: OmegaConf) -> torch.nn.Module:
    """
    Placeholder for EfficientDet model creation.
    Currently not implemented in torchvision.
    """
    try:
        config = get_efficientdet_config(conf.effdet.architecture)
    except KeyError:
        efficientdet_model_param_dict[conf.effdet.architecture] = {
            'name': conf.effdet.architecture,
            'backbone_name': conf.effdet.architecture,
            'backbone_args': dict(drop_path_rate=0.2),
            'image_size': conf.images.resize,
            'num_classes': conf.model.num_classes,
            'url': None,
        }
        config = get_efficientdet_config(conf.effdet.architecture)
        config.update({'num_classes': conf.model.num_classes})
        config.update({'image_size': conf.images.resize})

        net = EfficientDet(config, pretrained_backbone=True)
        net.class_net = HeadNet(
            config, 
            num_outputs=config.num_classes,
        )

        return DetBenchTrain(net, config)
    
class EffDetWrapper(nn.Module):
    def __init__(self, conf: OmegaConf, device: torch.device):
        super().__init__()
        try:
            self.model = create_model(
                model_name=conf.effdet.architecture,
                pretrained=True,
                num_classes=conf.model.num_classes,
                image_size=(conf.images.resize, conf.images.resize),
                max_det_per_image=conf.model.detections_per_img,
            ).to(device)
        except Exception as e:
            print(f"Error creating model: {e}")
            raise

        # Keep an explicit config for benches
        self.config = get_efficientdet_config(conf.effdet.architecture)
        self.config.num_classes = conf.model.num_classes
        self.config.image_size = conf.images.resize

        self.train_bench = DetBenchTrain(self.model).to(device)
        self.eval_bench  = DetBenchPredict(self.model).to(device)

    def train_mode(self):
        self.train_bench.train()
        self.eval_bench.train()

    def eval_mode(self):
        self.train_bench.eval()
        self.eval_bench.eval()

    def forward_train(self, images, targets):

        return self.train_bench(images, targets)
    
    @torch.no_grad()
    def predict(self, images):

        if images.ndim == 3:
            images = images.unsqueeze(0)

        return self.eval_bench(images)

if __name__ == "__main__":
    conf = OmegaConf.create({
        "effdet": {
            "architecture": "tf_efficientdet_d0",
        },
        "model": {
            "num_classes": 2,
        },
        "images": {
            "resize": 512,
        },
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effdet_wrapper = EffDetWrapper(conf, device)
    print("EfficientDet model created successfully.")

    img = torch.randn(3, conf.images.resize, conf.images.resize).to(device)
    outputs = effdet_wrapper.predict(img)
    print("Prediction completed successfully.")
    print(outputs)