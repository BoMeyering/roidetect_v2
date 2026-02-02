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