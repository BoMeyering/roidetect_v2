"""
scripts/format_bboxes.py
==========================
This script formats bounding box annotations for object detection tasks.
BoMeyering 2026
"""

import os
import json
import ndjson
from tqdm import tqdm
from shutil import copy2
from glob import glob
from scipy.stats import uniform

PROJECT_ID = 'cml1weuzx0m3b073571jv9m22'

with open('metadata/class_mapping.json', 'r') as f:
    class_map = json.load(f)

with open('data/annotations.ndjson', 'r') as f:
    annotations = ndjson.load(f)

# with open('data/formated_bboxes.json', 'w') as f:
formatted_bboxes = {}
TRAIN_DIR = 'data/processed/train'
VAL_DIR = 'data/processed/val'
RAW_DIR = 'data/raw'

TRAIN_SPLIT = 0.8

raw_img_names = [name for name in glob("*", root_dir=RAW_DIR) if name.endswith(('.jpg', '.jpeg'))]




def main():
    train_labels = {}
    val_labels = {}

    train_counter = 0
    val_counter = 0 

    for _, image in tqdm(enumerate(annotations), colour='blue', desc='Formatting BBoxes'):
        
        split = 'train' if uniform.rvs() <= TRAIN_SPLIT else 'val'

        image_id = image['data_row']['external_id']
        formatted_bboxes[image_id] = {'boxes': [], 'labels': []}

        labels = image['projects'][PROJECT_ID]['labels'][0]['annotations']['objects']
        
        for label in labels:
            if label['annotation_kind'] != 'ImageBoundingBox':
                continue
            bbox_class = label['name']
            bbox = label['bounding_box']

            y1, x1, h, w = list(bbox.values())
            y2 = y1 + h
            x2 = x1 + w

            formatted_bboxes[image_id]['boxes'].append([x1, y1, x2, y2])
            formatted_bboxes[image_id]['labels'].append(class_map[bbox_class])
        
        if split == 'train':
            train_labels[image_id] = formatted_bboxes[image_id]
            train_counter += 1
        else:
            val_labels[image_id] = formatted_bboxes[image_id]
            val_counter += 1

    with open('data/train_labels.json', 'w') as f:
        json.dump(train_labels, f)

    with open('data/val_labels.json', 'w') as f:
        json.dump(val_labels, f)

    print(train_counter, "images in training set")
    print(val_counter, "images in validation set")

if __name__ == "__main__":
    main()