"""
train_retinanet.py
==========================
This script trains a RetinaNet model using a Swin Transformer backbone for object detection tasks.
BoMeyering 2026
"""

from src.models import retinanet_swin
from src.datasets import TrainDataset
from src.transforms import get_train_transforms
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

conf_dict = {
    "directories": {
        "train_dir": "data/processed/train/labeled",
        "valid_dir": "data/processed/valid/images",
        "annotations_file": "data/formatted_bboxes.json"
    },
    "training": {
        "batch_size": 8,
        "num_epochs": 50,
        "learning_rate": 0.001
    }
}

conf = OmegaConf.create(conf_dict)

ds = TrainDataset(conf, get_train_transforms(resize=(1024, 1204)))
dl = DataLoader(ds, batch_size=conf.training.batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# model = retinanet_swin(num_classes=3)  # background + 2 cla

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=conf.training.learning_rate)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(conf.training.num_epochs):
    # model.train()
    epoch_loss = 0.0

    for image_ids, images, targets in dl:
        print(type(image_ids))
        print(type(images))
        print(type(targets))
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # loss_dict = model(images, targets)
        # losses = sum(loss for loss in loss_dict.values())
        # print(losses)

        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()

        # epoch_loss += losses.item()

    # lr_scheduler.step()
    # print(f"Epoch [{epoch+1}/{conf.training.num_epochs}], Loss: {epoch_loss/len(dl):.4f}")