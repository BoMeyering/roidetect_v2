"""
src/datasets.py
===========================
This module defines dataset utilities for object detection tasks.
BoMeyering 2026
"""

import os
import torch
import cv2
import json
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from omegaconf import OmegaConf
import albumentations as A

class TrainDataset(Dataset):
    def __init__(self, conf: OmegaConf, transforms: A.Compose):
        self.conf = conf
        self.images_dir = self.conf.directories.train_dir
        self.annotations_file = self.conf.directories.annotations_file
        self.transforms = transforms

        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = [id for id in glob("*", root_dir=self.images_dir) if id.endswith(('jpeg', 'jpg'))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)

        targets = {}
        boxes = np.array(self.annotations[image_id]['boxes'], dtype=np.float32)
        labels = np.array(self.annotations[image_id]['labels'], dtype=np.int64)

        transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
        image = transformed['image']

        targets['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        targets['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)

        # if targets["boxes"].numel() == 0:
        #     return None

        return image_id, image, targets
    

class ValDataset(Dataset):
    def __init__(self, conf: OmegaConf, transforms: A.Compose):
        self.conf = conf
        self.images_dir = self.conf.directories.val_dir
        self.annotations_file = self.conf.directories.annotations_file
        self.transforms = transforms

        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = [id for id in glob("*", root_dir=self.images_dir) if id.endswith(('jpeg', 'jpg'))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_id)
        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)

        targets = {}
        boxes = np.array(self.annotations[image_id]['boxes'], dtype=np.float32)
        labels = np.array(self.annotations[image_id]['labels'], dtype=np.int64)

        transformed = self.transforms(image=raw_image, bboxes=boxes, labels=labels)
        image = transformed['image']

        targets['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        targets['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)

        # if targets["boxes"].numel() == 0:
        #     return None
        
        return image_id, image, targets, raw_image

class InferenceDataset(Dataset):
    def __init__(self, conf: OmegaConf, transforms: A.Compose):
        self.conf = conf
        self.images_dir = self.conf.directories.inference_dir
        self.transforms = transforms

        self.image_ids = [id for id in glob("*", root_dir=self.images_dir) if id.endswith(('jpeg', 'jpg'))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_id)
        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)

        transformed = self.transforms(image=raw_image)
        image = transformed['image']

        return image_id, image, raw_image
    
class EffdetDataset(Dataset):
    def __init__(self, conf: OmegaConf, transforms: A.Compose, type: str='train'):
        self.conf = conf
        self.images_dir = self.conf.directories.image_dir
        if type == 'train':
            self.annotations_file = self.conf.directories.train_annotations_file
        else:
            self.annotations_file = self.conf.directories.val_annotations_file
        self.transforms = transforms

        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = [id for id in glob("*", root_dir=self.images_dir) if id.endswith(('jpeg', 'jpg'))]

    def __len__(self):
        return len(self.annotations.keys())

    def __getitem__(self, idx):
        image_id = list(self.annotations.keys())[idx]
        image_path = os.path.join(self.images_dir, image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)

        sample = {
            "image": image,
            "bboxes": np.array(self.annotations[image_id]['boxes'], dtype=np.float32),
            "labels": np.array(self.annotations[image_id]['labels'], dtype=np.int64)
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"], dtype=np.float32)
        image = sample['image']
        labels = sample['labels']

        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]  # xyxy to yxyx

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0])
        }

        return image, target, image_id

class EffdetInferenceDataset(Dataset):
    def __init__(self, conf: OmegaConf, transforms: A.Compose):
        self.conf = conf
        self.images_dir = self.conf.directories.image_dir
        self.annotations_file = self.conf.directories.val_annotations_file
        self.transforms = transforms

        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = [id for id in glob("*", root_dir=self.images_dir) if id.endswith(('jpeg', 'jpg'))]

    def __len__(self):
        return len(self.annotations.keys())

    def __getitem__(self, idx):
        image_id = list(self.annotations.keys())[idx]
        image_path = os.path.join(self.images_dir, image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)

        sample = {
            "image": image,
        }

        sample = self.transforms(**sample)
        image = sample['image']

        return image, image_id