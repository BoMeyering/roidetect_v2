#!/usr/bin/env python3
"""
scripts/onnx_inference.py
Run ONNX inference with a trained EfficientDet model.

Decodes raw class_logits + box_regressors via anchor grid and batched NMS,
then optionally scales boxes back to original image resolution.

Usage:
    python scripts/onnx_inference.py \\
        --model outputs/onnx/effdet_d2_epoch10.onnx \\
        --images data/raw/img1.jpg data/raw/img2.jpg \\
        [--score-threshold 0.3] [--nms-iou 0.5] [--max-det 100] \\
        [--device cpu|cuda] [--visualize] [--output-dir outputs/predictions]

    # or pass a directory of images:
    python scripts/onnx_inference.py --model ... --images data/raw/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.postprocess import postprocess_detections, get_anchor_boxes

# ---------------------------------------------------------------------------
# Constants (match dataset normalization used during training)
# ---------------------------------------------------------------------------

ARCHITECTURE = "tf_efficientdet_d2"
IMG_SIZE     = 1024

MEANS = (0.45607941452457035, 0.4613301098895315, 0.3276268550568099)
STDS  = (0.25295059190761293, 0.24145019419169375, 0.2300590928582059)

CLASS_NAMES = {1: "quadrat_bb", 2: "marker_bb"}
CLASS_COLORS = {1: (0, 200, 80), 2: (255, 80, 0)}  # BGR


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img_path: Path, img_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image, resize, and normalise to a float32 [1, 3, H, W] batch."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normed  = resized.astype(np.float32) / 255.0
    normed  = (normed - np.array(MEANS, dtype=np.float32)) / np.array(STDS, dtype=np.float32)
    batch   = normed.transpose(2, 0, 1)[np.newaxis]          # [1, C, H, W]
    return batch, (orig_h, orig_w)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def infer(
    onnx_path: str,
    image_paths: list[Path],
    architecture: str   = ARCHITECTURE,
    img_size: int       = IMG_SIZE,
    score_threshold: float = 0.3,
    nms_iou: float      = 0.5,
    max_det: int        = 100,
    device: str         = "cpu",
) -> list[dict]:
    """
    Returns a list of dicts, one per image:
        {
          "image": str,
          "detections": np.ndarray  # [N, 6] — x1 y1 x2 y2 score class (1-indexed)
        }
    Boxes are in the original image coordinate space.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit(
            "onnxruntime is required:\n"
            "  pip install onnxruntime-gpu   # GPU support\n"
            "  pip install onnxruntime       # CPU-only"
        )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    anchors = get_anchor_boxes(architecture, img_size)  # [N, 4] yx format, torch.Tensor

    results = []
    for img_path in image_paths:
        batch, (orig_h, orig_w) = preprocess(img_path, img_size)

        cls_np, box_np = sess.run(None, {input_name: batch})

        dets = postprocess_detections(
            torch.from_numpy(cls_np),   # [1, N, C]
            torch.from_numpy(box_np),   # [1, N, 4]
            anchors,
            max_detection_points=5000,
            max_det_per_image=max_det,
            nms_iou_threshold=nms_iou,
            score_threshold=score_threshold,
        )  # [1, max_det, 6]

        dets_img = dets[0]                       # [max_det, 6]
        valid    = dets_img[:, 5] > 0            # class == 0 → zero-padded empty row
        dets_img = dets_img[valid].clone()

        # Scale from model input space back to original image space
        dets_img[:, [0, 2]] *= orig_w / img_size
        dets_img[:, [1, 3]] *= orig_h / img_size

        results.append({"image": str(img_path), "detections": dets_img.cpu().numpy()})

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_results(results: list[dict]) -> None:
    for r in results:
        dets = r["detections"]
        print(f"\n{Path(r['image']).name}  —  {len(dets)} detection(s)")
        for d in dets:
            x1, y1, x2, y2, score, cls = d.tolist()
            name = CLASS_NAMES.get(int(cls), f"class_{int(cls)}")
            print(f"  [{name:12s}]  score={score:.3f}  box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")


def visualize_and_save(results: list[dict], output_dir: Path | None) -> None:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        img = cv2.imread(r["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            continue

        for d in r["detections"]:
            x1, y1, x2, y2, score, cls = d.tolist()
            color = CLASS_COLORS.get(int(cls), (0, 200, 255))
            name  = CLASS_NAMES.get(int(cls), f"class_{int(cls)}")
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{name} {score:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (int(x1), int(y1) - lh - 6), (int(x1) + lw, int(y1)), color, -1)
            cv2.putText(img, label, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if output_dir:
            out_path = output_dir / Path(r["image"]).name
            cv2.imwrite(str(out_path), img)
            print(f"  Saved → {out_path}")
        else:
            cv2.imshow("detections", img)
            cv2.waitKey(0)

    if not output_dir:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="ONNX inference for EfficientDet (with anchor decoding + NMS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  required=True, help="Path to .onnx file.")
    p.add_argument(
        "--images", required=True, nargs="+",
        help="Image file(s) or a directory (jpg/jpeg/png/tif/tiff).",
    )
    p.add_argument("--architecture", default=ARCHITECTURE, help="EfficientDet architecture name.")
    p.add_argument("--img-size", type=int, default=IMG_SIZE, help="Model input size (must match export).")
    p.add_argument("--score-threshold", type=float, default=0.3, help="Minimum confidence score.")
    p.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold.")
    p.add_argument("--max-det", type=int, default=100, help="Max detections per image.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--visualize", action="store_true", help="Draw boxes on each image.")
    p.add_argument("--output-dir", default=None, help="Save visualizations here (requires --visualize).")
    p.add_argument("--save-json", default=None, help="Save detection results as JSON file.")
    args = p.parse_args()

    # Collect image paths
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths: list[Path] = []
    for src in args.images:
        src_p = Path(src)
        if src_p.is_dir():
            image_paths.extend(sorted(f for f in src_p.iterdir() if f.suffix.lower() in exts))
        elif src_p.is_file():
            image_paths.append(src_p)
        else:
            print(f"Warning: {src} not found, skipping.")

    if not image_paths:
        raise SystemExit("No valid images found.")

    print(f"Running inference on {len(image_paths)} image(s) ...")

    results = infer(
        onnx_path     = args.model,
        image_paths   = image_paths,
        architecture  = args.architecture,
        img_size      = args.img_size,
        score_threshold = args.score_threshold,
        nms_iou       = args.nms_iou,
        max_det       = args.max_det,
        device        = args.device,
    )

    print_results(results)

    if args.visualize:
        visualize_and_save(results, Path(args.output_dir) if args.output_dir else None)

    if args.save_json:
        serialisable = [
            {"image": r["image"], "detections": r["detections"].tolist()}
            for r in results
        ]
        with open(args.save_json, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nDetections saved → {args.save_json}")


if __name__ == "__main__":
    main()
