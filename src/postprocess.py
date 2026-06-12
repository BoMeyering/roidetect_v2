"""
src/postprocess.py
Post-processing for raw EfficientDet ONNX outputs.

Replicates the DetBenchPredict pipeline exactly:
  _post_process  → top-K selection (5000 anchors)
  decode_box_outputs → anchor delta → xyxy boxes
  batched_nms    → IoU 0.5, keep top max_det_per_image
  zero-pad       → fixed output shape [B, max_det_per_image, 6]

Output tensor columns: [x1, y1, x2, y2, score, class]
  - coords are in the model input image space (e.g. 1024×1024)
  - class is 1-indexed (0 = background / empty row)
"""

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from effdet import get_efficientdet_config
from effdet.anchors import Anchors


# ---------------------------------------------------------------------------
# Anchor grid (cached per (architecture, image_size) pair)
# ---------------------------------------------------------------------------

_anchor_cache: dict = {}


def get_anchor_boxes(architecture: str, image_size: int) -> torch.Tensor:
    """Return anchor boxes [N, 4] in (y1, x1, y2, x2) for the given model."""
    key = (architecture, image_size)
    if key not in _anchor_cache:
        config = get_efficientdet_config(architecture)
        config.image_size = (image_size, image_size)
        _anchor_cache[key] = Anchors.from_config(config).boxes
    return _anchor_cache[key]


# ---------------------------------------------------------------------------
# Box decoding
# ---------------------------------------------------------------------------

def decode_boxes(regressors: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Decode anchor deltas to absolute (x1, y1, x2, y2) boxes.

    Args:
        regressors: [N, 4]  — (ty, tx, th, tw) network outputs
        anchors:    [N, 4]  — (y1, x1, y2, x2) anchor boxes

    Returns:
        boxes: [N, 4] in (x1, y1, x2, y2) format
    """
    cy_a = (anchors[:, 0] + anchors[:, 2]) / 2
    cx_a = (anchors[:, 1] + anchors[:, 3]) / 2
    ha   =  anchors[:, 2] - anchors[:, 0]
    wa   =  anchors[:, 3] - anchors[:, 1]

    ty, tx, th, tw = regressors.unbind(dim=1)

    h       = torch.exp(th) * ha
    w       = torch.exp(tw) * wa
    ycenter = ty * ha + cy_a
    xcenter = tx * wa + cx_a

    x1 = xcenter - w / 2
    y1 = ycenter - h / 2
    x2 = xcenter + w / 2
    y2 = ycenter + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


# ---------------------------------------------------------------------------
# Main post-processing function
# ---------------------------------------------------------------------------

def postprocess_detections(
    class_logits: torch.Tensor,
    box_regressors: torch.Tensor,
    anchors: torch.Tensor,
    max_detection_points: int = 5000,
    max_det_per_image: int = 100,
    nms_iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> torch.Tensor:
    """Convert raw ONNX outputs to final detections.

    Exactly replicates DetBenchPredict (no img_scale rescaling — coordinates
    are in the model input image space, e.g. 1024×1024).

    Args:
        class_logits:   [B, N, num_classes]  raw logits from ONNX
        box_regressors: [B, N, 4]            raw deltas from ONNX
        anchors:        [N, 4]               (y1,x1,y2,x2) from get_anchor_boxes()
        max_detection_points: top-K anchors to consider before NMS (default 5000)
        max_det_per_image:    max detections to return per image (default 100)
        nms_iou_threshold:    NMS IoU threshold (default 0.5)
        score_threshold:      discard detections below this sigmoid score (default 0.0)

    Returns:
        detections: [B, max_det_per_image, 6]
            columns: [x1, y1, x2, y2, score, class]
            class is 1-indexed; rows with class=0 are zero-padded empties.
    """
    device = class_logits.device
    B, N, C = class_logits.shape
    anchors = anchors.to(device)

    batch_detections = []

    for i in range(B):
        cls_i = class_logits[i]   # [N, C]
        box_i = box_regressors[i] # [N, 4]

        # --- top-K selection (mirrors _post_process) ---
        flat_scores = cls_i.reshape(-1)
        k = min(max_detection_points, flat_scores.shape[0])
        _, topk_flat = torch.topk(flat_scores, k=k)

        anchor_idx = topk_flat // C
        class_idx  = topk_flat  % C

        cls_topk = cls_i[anchor_idx, class_idx]   # [k]
        box_topk = box_i[anchor_idx]               # [k, 4]
        anc_topk = anchors[anchor_idx]             # [k, 4]

        # --- box decoding ---
        boxes = decode_boxes(box_topk, anc_topk)   # [k, 4] xyxy

        # --- sigmoid scores + optional threshold ---
        scores = torch.sigmoid(cls_topk)           # [k]
        if score_threshold > 0:
            keep_thresh = scores > score_threshold
            boxes      = boxes[keep_thresh]
            scores     = scores[keep_thresh]
            class_idx  = class_idx[keep_thresh]

        # --- batched NMS ---
        if boxes.shape[0] > 0:
            keep = batched_nms(
                boxes.float(), scores.float(),
                class_idx.to(torch.int64),
                iou_threshold=nms_iou_threshold,
            )
            keep = keep[:max_det_per_image]
            det_boxes   = boxes[keep]
            det_scores  = scores[keep].unsqueeze(1)
            det_classes = (class_idx[keep] + 1).float().unsqueeze(1)  # 1-indexed
        else:
            det_boxes   = boxes
            det_scores  = scores.unsqueeze(1)
            det_classes = class_idx.float().unsqueeze(1)

        # --- concat and zero-pad to fixed length ---
        num_det = det_boxes.shape[0]
        det = torch.cat([det_boxes, det_scores, det_classes], dim=1)  # [num_det, 6]
        if num_det < max_det_per_image:
            pad = torch.zeros(
                max_det_per_image - num_det, 6, device=device, dtype=det.dtype
            )
            det = torch.cat([det, pad], dim=0)

        batch_detections.append(det)

    return torch.stack(batch_detections, dim=0)  # [B, max_det, 6]
