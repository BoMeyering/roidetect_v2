"""
scripts/format_bboxes.py
==========================
This script formats bounding box annotations for object detection tasks.
BoMeyering 2026
"""

import json
import ndjson
from tqdm import tqdm

PROJECT_ID = 'cma74xh22061607ysdyf4gudl'

with open('metadata/class_mapping.json', 'r') as f:
    class_map = json.load(f)

with open('data/annotations.ndjson', 'r') as f:
    annotations = ndjson.load(f)

# with open('data/formated_bboxes.json', 'w') as f:
formatted_bboxes = {}

def main():
    for _, image in tqdm(enumerate(annotations), colour='blue', desc='Formatting BBoxes'):
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

    with open('data/formatted_bboxes.json', 'w') as f:
        json.dump(formatted_bboxes, f)

if __name__ == "__main__":
    main()