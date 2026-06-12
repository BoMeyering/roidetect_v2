#!/usr/bin/env python3
"""
export_onnx.py
Export a trained EfficientDet checkpoint to ONNX format.

The NMS head inside DetBenchPredict is a TorchScript ScriptFunction and cannot
be traced to ONNX.  This script exports only the backbone + BiFPN + class/box
heads (raw predictions, no NMS, no anchor decoding):

  class_logits   [B, num_anchors_total, num_classes]  — raw, pre-sigmoid
  box_regressors [B, num_anchors_total, 4]             — raw delta regressors

Post-processing (anchor decoding + NMS) must be handled by the deployment runtime
using the anchor grid from effdet.anchors.Anchors.from_config(model.config).

Usage:
    python export_onnx.py \\
        --config  configs/effdet_d2.yaml \\
        --weights model_checkpoints/<run>/<checkpoint>.pth \\
        --name    effdet_d2_best \\
        [--img-size 1024] [--output-dir outputs/onnx] [--opset 18] \\
        [--batch-size 1]  [--verify]
"""

import argparse
import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# The dynamo ONNX exporter scans all registered torchvision ops and emits
# harmless annotation warnings for ROI ops not used in EfficientDet.
logging.getLogger("torch.onnx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")

from src.models import EffDetWrapper


# ---------------------------------------------------------------------------
# ONNX-friendly wrapper
# ---------------------------------------------------------------------------

class _EfficientDetExporter(nn.Module):
    """Flattens the FPN-level list outputs of EfficientDet into contiguous tensors.

    EfficientDet.forward returns two Python lists (one per FPN level).  Python
    list iteration is traced statically, so all concat ops are inlined into the
    graph.  The output is:

      class_logits   [B, N, C]  where N = sum of (H_i * W_i * anchors_per_loc)
      box_regressors [B, N, 4]
    """

    def __init__(self, effdet_model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.model = effdet_model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        class_out, box_out = self.model(x)
        # class_out: list of [B, anchors_per_loc * num_classes, H_i, W_i]
        # box_out  : list of [B, anchors_per_loc * 4,           H_i, W_i]
        B = x.shape[0]
        cls = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes) for c in class_out],
            dim=1,
        )
        box = torch.cat(
            [b.permute(0, 2, 3, 1).reshape(B, -1, 4) for b in box_out],
            dim=1,
        )
        return cls, box


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_wrapper(conf: OmegaConf, weights_path: str, device: torch.device) -> EffDetWrapper:
    """Create EffDetWrapper and load weights from a checkpoint or plain state dict."""
    wrapper = EffDetWrapper(conf=conf, device=device)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "ema_state_dict" in checkpoint:
            state_dict = checkpoint["ema_state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Already a plain {param_name: tensor} mapping
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    wrapper.load_state_dict(state_dict)
    wrapper.eval_mode()
    return wrapper


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(
    config_path: str,
    weights_path: str,
    img_size: int | None,
    output_dir: str,
    name: str,
    opset: int,
    batch_size: int,
    verify: bool,
) -> None:
    conf = OmegaConf.load(config_path)
    if img_size is not None:
        conf.images.resize = img_size

    img_size_resolved: int = (
        conf.images.resize
        if isinstance(conf.images.resize, int)
        else conf.images.resize[0]
    )
    num_classes: int = conf.model.num_classes

    print(f"Config        : {config_path}")
    print(f"Weights       : {weights_path}")
    print(f"Image size    : {img_size_resolved}x{img_size_resolved}")
    print(f"Num classes   : {num_classes}")
    print(f"Batch size    : {batch_size}")
    print(f"ONNX opset    : {opset}")

    # Export on CPU for broadest runtime compatibility.
    # The pretrained backbone weights are downloaded on first run (timm cache).
    device = torch.device("cpu")
    print("\nLoading model ...")
    wrapper = _load_wrapper(conf, weights_path, device)

    exporter = _EfficientDetExporter(wrapper.model, num_classes).eval()

    dummy = torch.zeros(batch_size, 3, img_size_resolved, img_size_resolved)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / f"{name}.onnx"

    print(f"\nExporting → {out_path}")
    batch_dim = torch.export.Dim("batch_size")
    with torch.no_grad():
        torch.onnx.export(
            exporter,
            dummy,
            str(out_path),
            opset_version=opset,
            input_names=["images"],
            output_names=["class_logits", "box_regressors"],
            dynamic_shapes={"x": {0: batch_dim}},
        )

    # The dynamo exporter may split large weights into a <name>.data sidecar file.
    data_file = Path(str(out_path) + ".data")
    total_bytes = out_path.stat().st_size + (data_file.stat().st_size if data_file.exists() else 0)
    size_mb = total_bytes / 1024**2
    print(f"Saved ({size_mb:.1f} MB): {out_path}")
    if data_file.exists():
        print(f"  + external weights : {data_file}  (deploy alongside the .onnx)")

    if verify:
        _verify(str(out_path), dummy)

    architecture: str = conf.effdet.architecture
    print(
        "\nOutputs are raw (NMS-free):\n"
        "  class_logits   [B, num_anchors, num_classes]  — sigmoid → class scores\n"
        "  box_regressors [B, num_anchors, 4]             — decode with anchor grid\n"
        "\nAnchor grid:\n"
        "  from effdet.anchors import Anchors\n"
        "  from effdet import get_efficientdet_config\n"
        f"  config = get_efficientdet_config('{architecture}')\n"
        f"  config.image_size = {img_size_resolved}\n"
        "  anchors = Anchors.from_config(config).boxes  # [N, 4] in yx format"
    )


# ---------------------------------------------------------------------------
# Verification (optional, requires onnx + onnxruntime)
# ---------------------------------------------------------------------------

def _verify(onnx_path: str, dummy_input: torch.Tensor) -> None:
    try:
        import onnx
        model_proto = onnx.load(onnx_path)
        onnx.checker.check_model(model_proto)
        print("ONNX graph check : PASSED")
    except ImportError:
        print("onnx not installed — skipping graph check.")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
        outputs = sess.run(None, inputs)
        print(
            f"OnnxRuntime check: PASSED — "
            f"output shapes {[list(o.shape) for o in outputs]}"
        )
    except ImportError:
        print("onnxruntime not installed — skipping inference check.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Export a trained EfficientDet checkpoint to ONNX (no NMS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to model YAML config.")
    p.add_argument("--weights", required=True, help="Path to .pth checkpoint.")
    p.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override image size from config (square input, e.g. 1024).",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/onnx",
        help="Directory to write the .onnx file.",
    )
    p.add_argument(
        "--name",
        required=True,
        help="Output filename without .onnx extension.",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18, minimum supported by dynamo exporter).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for the dummy export input.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Run onnxruntime on the exported model to verify correctness.",
    )

    args = p.parse_args()
    export(
        config_path=args.config,
        weights_path=args.weights,
        img_size=args.img_size,
        output_dir=args.output_dir,
        name=args.name,
        opset=args.opset,
        batch_size=args.batch_size,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
