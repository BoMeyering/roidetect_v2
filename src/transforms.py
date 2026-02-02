"""
src/transforms.py
===========================
This module defines image transformation utilities for object detection tasks.
BoMeyering 2026
"""

import omegaconf
import logging
import os
import json
import albumentations as A
from omegaconf import OmegaConf
from albumentations.pytorch import ToTensorV2
from typing import Iterable
from src.utils.config import Norm

def get_train_transforms(rgb_means: Iterable[float]=(0.485, 0.456, 0.406), rgb_stds: Iterable[float]=(0.229, 0.224, 0.225), resize: int=512):
    return A.Compose(
        [   
            A.Resize(height=resize, width=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.5),
            # A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.15), rotate=(-15, 15), p=0.5),
            # A.ToFloat(max_value=255.0),
            A.Normalize(mean=rgb_means, std=rgb_stds, max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )

def get_val_transforms(rgb_means: Iterable[float]=(0.485, 0.456, 0.406), rgb_stds: Iterable[float]=(0.229, 0.224, 0.225), resize: int=512):
    return A.Compose(
        [
            A.Resize(height=resize, width=resize),
            A.Normalize(mean=rgb_means, std=rgb_stds, max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )

def get_inference_transforms(rgb_means: Iterable[float]=(0.485, 0.456, 0.406), rgb_stds: Iterable[float]=(0.229, 0.224, 0.225), resize: int=512):
    return A.Compose(
        [
            A.Resize(height=resize, width=resize),
            A.Normalize(mean=rgb_means, std=rgb_stds, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

def set_normalization_values(conf: OmegaConf):
    """_summary_

    Args:
        conf (OmegaConf): _description_
    """

    logger = logging.getLogger()
    if not isinstance(conf, omegaconf.dictconfig.DictConfig):
        raise ValueError(f"Argument 'conf' should be of type 'omegaconf.dictconfig.DictConfig'.")
    
    if 'metadata' in conf:
        if (conf.metadata is not None) and ('norm_path' in conf.metadata):
            if os.path.exists(conf.metadata.norm_path):
                with open(conf.metadata.norm_path, 'r') as f:
                    norm_dict = json.load(f)
                    conf.metadata.norm = Norm(means=norm_dict['means'], std=norm_dict['std'])
            else:
                raise ValueError(f"Path to normalization values ({conf.metadata.norm_path}) does not exist")
        else:
            raise KeyError(f"Key 'norm_path' not found in 'conf.metadata'. Please set 'conf.metadata.norm_path' to the normalization value JSON path.")
    else:
        raise KeyError(f"Key 'metadata.norm_path' not found in 'conf'. Please set 'conf.metadata.norm_path' to the normalization value JSON path.")