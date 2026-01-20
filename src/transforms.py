"""
src/transforms.py
===========================
This module defines image transformation utilities for object detection tasks.
BoMeyering 2026
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Iterable

def get_train_transforms(rgb_means: Iterable[float]=(0.485, 0.456, 0.406), rgb_stds: Iterable[float]=(0.229, 0.224, 0.225), resize: Iterable[int]=(512, 512)):
    return A.Compose(
        [   
            A.Resize(height=resize[0], width=resize[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.15), rotate=(-15, 15), p=0.5),
            A.Normalize(mean=rgb_means, std=rgb_stds),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )

def get_valid_transforms(rgb_means: Iterable[float]=(0.485, 0.456, 0.406), rgb_stds: Iterable[float]=(0.229, 0.224, 0.225), resize: Iterable[int]=(512, 512)):
    return A.Compose(
        [
            A.Resize(height=resize[0], width=resize[1]),
            A.Normalize(mean=rgb_means, std=rgb_stds),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )