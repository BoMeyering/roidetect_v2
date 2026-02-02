"""
scripts/prelabel_images.py
Pre-label images using a trained model for faster annotation.
BoMeyering, 2026
"""

from cProfile import label
import torch
import torchvision
from tqdm import tqdm
from src.transforms import get_inference_transforms
from src.datasets import InferenceDataset
from src.models import create_fasterrcnn
from src.utils.loggers import setup_loggers, rank_log
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import os
from glob import glob
import cv2
import json
import numpy as np
import logging
import uuid
from PIL import Image
import requests
import base64
import labelbox as lb
import labelbox.types as lb_types
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("LB_API_KEY")
PROJECT_ID = os.getenv("LB_PROJECT_ID")

print(API_KEY, PROJECT_ID)

client = lb.Client(api_key=API_KEY)

UNLABELED_DIR = "data/processed/train/images"

conf = {
    "model_run": "labelbox_prelabeling_run",
    "backbone_out_channels": 256,
    "num_classes": 3,
    "detections_per_img": 100,
    "resize": (1024, 1024),
    "directories": {
        "inference_dir": UNLABELED_DIR,
        "log_dir": "logs/labelbox_logs"
    },
    "logging_level": "INFO"
}

mapping = {
    1: "quadrat_bb",
    2: "marker_bb"
}

conf = OmegaConf.create(conf)

setup_loggers(conf)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_names = [f for f in glob("*", root_dir=UNLABELED_DIR) if f.endswith(('jpeg', 'jpg'))]

model = create_fasterrcnn(conf).to(device)
model_states = torch.load("model_checkpoints/fasterrcnn_resnet50_2026-01-26_22.32.46/fasterrcnn_resnet50_2026-01-26_22.32.46_epoch_38_vloss-0.081062.pth", map_location=device)
model.load_state_dict(
    model_states['ema_state_dict'], 
    strict=True
)
model.eval()


def create_annotations_object(boxes, classes, type='prediction'):
    """
    Upload predicted labels to Labelbox
    """
    annotations = []

    for box, class_ in zip(boxes, classes):
        x1, y1, x2, y2 = box
        feature_name = mapping.get(class_, None)
        bbox_annotation = lb_types.ObjectAnnotation(
            name=feature_name,
            value=lb_types.Rectangle(
                start=lb_types.Point(x=x1, y=y1),  # x = left, y = top
                end=lb_types.Point(x=x2, y=y2),  # x= left + width , y = top + height
            ))
        annotations.append(bbox_annotation)

    return annotations

# def upload_label(label_list, id_list, type='prediction'):

#     if type == 'ground_truth':
#         upload_job = lb.LabelImport.create_from_objects(
#             client = client,
#             project_id = PROJECT_ID,
#             name="mal_job"+str(uuid.uuid4()),
#             labels=label_list
#         )
#     elif type == 'prediction':
#         upload_job = lb.MALPredictionImport.create_from_objects(
#             client = client,
#             project_id = PROJECT_ID,
#             name="mal_job"+str(uuid.uuid4()),
#             predictions=label_list
#         )

#     return upload_job

# Upload ground truth labels
# with open("data/formatted_bboxes.json", "r") as f:
#     gt_data = json.load(f)

# pbar = tqdm(enumerate(gt_data.items()), colour="blue", desc='Uploading GT labels', total=len(gt_data))
# label_list = []

# for i, (external_id, labels) in pbar:
#     pbar.set_description(f"Processing item {external_id}")
#     id_dict = client.get_data_row_ids_for_external_ids([external_id])
#     data_row_id = id_dict.get(external_id)[0]

#     boxes = labels['boxes']
#     classes = labels['labels']

#     if len(boxes) == 0:
#         logger.info(f"No GT boxes for data row {data_row_id} - external_id {external_id}")
#         continue

#     annotations = create_annotations_object(boxes, classes, type='ground_truth')
#     label_list.append(
#         lb_types.Label(
#             data={"uid": data_row_id},
#             annotations=annotations,
#             is_benchmark_reference = True
#         )
#     )

#     if len(label_list) == 500:
#         upload_job = lb.LabelImport.create_from_objects(
#             client = client,
#             project_id = PROJECT_ID,
#             name="label_import_job"+str(uuid.uuid4()),
#             labels=label_list
#         )

#         logger.error(f"Errors: {upload_job.errors}", )
#         logger.info(f"Uploaded batch of {len(label_list)} GT labels, job ID: {upload_job.uid}")

#         label_list = []

# if len(label_list) > 0:
#     upload_job = lb.LabelImport.create_from_objects(
#         client = client,
#         project_id = PROJECT_ID,
#         name="label_import_job"+str(uuid.uuid4()),
#         labels=label_list
#     )
#     logger.info(f"Uploaded final batch of {len(label_list)} GT labels, job ID: {upload_job.uid}")


# Run predictions on unlabeled data
transforms = get_inference_transforms(resize=(conf.resize))

ds = InferenceDataset(
    conf=conf,
    transforms=transforms
)

pbar = tqdm(enumerate(ds), colour="blue", desc='Running', total=len(ds))

# Create an empty list to hold all labels to upload in batch
label_list = []

for i, (img_id, img, raw_img) in pbar:
    pbar.set_description(f"Processing item {img_id}")
    id_dict = client.get_data_row_ids_for_external_ids([img_id])
    data_row_id = id_dict.get(img_id)[0]
    

    h, w = raw_img.shape[:2]
    img = img.to(device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
    
    outputs = {k: v.to('cpu') for k, v in outputs[0].items()}
    boxes = outputs['boxes'].numpy()
    scores = outputs['scores'].numpy()
    labels = outputs['labels'].numpy()

    # Filter out low confidence detections
    conf_threshold = 0.85
    keep_idxs = np.where(scores >= conf_threshold)

    boxes = boxes[keep_idxs]
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]

    if len(boxes) == 0:
        logger.info(f"No detections above confidence threshold for data row {data_row_id} - external_id {img_id}")
        continue

    # Resize boxes back to original image size
    resize_h, resize_w = conf.resize
    scale_x = w / resize_w
    scale_y = h / resize_h
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
    boxes = boxes.astype(int).tolist()
    scores = scores.tolist()
    labels = labels.tolist()

    annotations = create_annotations_object(boxes, labels)
    
    label_list.append(
        lb_types.Label(
            data={"uid": data_row_id},
            annotations=annotations
        )
    )

    if len(label_list) == 500:
        upload_job = lb.MALPredictionImport.create_from_objects(
            client = client,
            project_id = PROJECT_ID,
            name="mal_job"+str(uuid.uuid4()),
            predictions=label_list
        )

        logger.error(f"Errors: {upload_job.errors}", )
        logger.info(f"Uploaded batch of {len(label_list)} predictions, job ID: {upload_job.uid}")

        label_list = []

upload_job = lb.MALPredictionImport.create_from_objects(
    client = client,
    project_id = PROJECT_ID,
    name="mal_job"+str(uuid.uuid4()),
    predictions=label_list
)

logger.error(f"Errors: {upload_job.errors}", )
logger.info(f"Uploaded final batch of {len(label_list)} predictions, job ID: {upload_job.uid}")