"""
src.metrics.py
Torchmetrics for image prediction
BoMeyering 2025
"""

import random
from collections import deque
import numpy as np
import logging
import torch
from torchmetrics import MetricCollection, MeanMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou, generalized_box_iou, complete_box_iou, distance_box_iou
from typing import Union, List, Optional, Sequence, Iterable
from numbers import Real
from abc import ABC, abstractmethod

logger = logging.getLogger()

class Meter(ABC):
    """Abstract base class for metric accumulators."""

    def __init__(self) -> None:
        self._values = []

    @abstractmethod
    def update(self, val: Real, *args, **kwargs) -> None:
        """Update the meter with a new value (and optional count)."""
        ...

    def reset(self) -> None:
        """ Reset internal state """
        self._values.clear()

    @property
    def mean(self) -> Optional[Real]:
        """ Mean of tracked values (or None if empty) """
        if not self._values:
            return None
        return float(sum(self._values) / len(self._values))

    @property
    def min(self) -> Optional[Real]:
        """Minimum of tracked values (or None if empty)."""
        if not self._values:
            return None
        return min(self._values)

    @property
    def max(self) -> Optional[Real]:
        """Maximum of tracked values (or None if empty)."""
        if not self._values:
            return None
        return max(self._values)

    @property
    def values(self) -> Sequence[Real]:
        """All tracked values."""
        return self._values

class ValueMeter(Meter):
    """ A class to handle any numerical values """

    def __init__(self):
        super().__init__()

    def update(self, val: Real, n: int=1):
        """Append `val` to the list `n` times."""
        if not isinstance(val, (float, int)):
            raise ValueError(
                f"Argument 'val' must be a numeric data type; got {type(val)} instead."
            )
        if not isinstance(n, int) or n < 1:
            raise ValueError(
                f"Argument 'n' must be a positive integer; got {type(n)} instead."
            )
        self._values.extend([val]*n)
    
    def __str__(self):
        """ Implement str format """
        return f"Mean: {self.mean} - Min: {self.min} - Max: {self.max}"
    
    def __repr__(self):
        """ Implement object representation """
        if len(self._values) <= 10:
            return f"ValueMeter(values={self._values}, len={len(self._values)})"
        else:
            first = ", ".join(map(str, self._values[:3]))
            last = ", ".join(map(str, self._values[-3:]))
            return f"ValueMeter(values=[{first}, ..., {last}], len={len(self._values)})"
    
class RunningAvgMeter(Meter):
    def __init__(self, window_length: int=10):
        """Initialize the RunningAvgMeter

        parameters:
        -----------
            window_length : int
                The number of numeric elements to include in the running average. Defaults to 10.        
        """
        super().__init__()
        if not isinstance(window_length, (float, int)):
            raise ValueError(
                f"Argument 'window_length' must be a scalar numeric data type"
            )
        # Clip window_length between 2 and 100
        wl = int(window_length)
        wl = max(2, min(wl, 100))
        # Create the deque
        self._values = deque(maxlen=wl)

    def update(self, val: Real, *args):
        """ Update the deque with a new value """
        if not isinstance(val, (float, int)):
            raise ValueError(
                f"Argument 'val' must be a numeric data type; got {type(val)} instead."
            )
        self._values.append(val)

    def __str__(self):
        """ Implement str format """
        return f"Mean: {self.mean} - Min: {self.min} - Max: {self.max}"
    
    def __repr__(self):
        """ Implement object representation """
        if len(self._values) <= 10:
            return f"RunningAvgMeter(values={self._values}, len={len(self._values)})"
        else:
            first = ", ".join(map(str, list(self._values)[:3]))
            last = ", ".join(map(str, list(self._values)[-3:]))
            return f"RunningAvgMeter(values=[{first}, ..., {last}], len={len(self._values)})"

class MeterSet:
    """ MeterSet manages a group of abstract Meter instances """
    def __init__(self, meter_dict: dict[str, Meter]):
        
        if not isinstance(meter_dict, dict):
            raise ValueError(
                f"Parameter 'meter_dict' must be a dictionary with meter names as keys and values of type abstract class Meter"
            )
        
        for k, v in meter_dict.items():
            if not isinstance(v, (Meter, ValueMeter, RunningAvgMeter)):
                raise ValueError(
                    f"All values in parameter 'meter_dict' must be of base type Meter"
                )
        
        self.meters = meter_dict
    
    def __getitem__(self, name: str):
        try:
            return self.meters[name]
        except KeyError:
            raise KeyError(f"No meter named '{name}'. Existing: {list(self.meters)}")
        
    def _add_one_meter(self, name: str, meter_type: str, **kwargs):
        """ Add another meter to the MeterSet """
        if name in self.meters.keys():
            raise KeyError(
                f"A meter named {name} is already present in the MeterSet. Please choose a different key."
            )
        wl = kwargs['window_length'] if 'window_length' in kwargs else 10

        if meter_type == 'value':
            meter = ValueMeter()
        elif meter_type == 'running_avg':
            meter = RunningAvgMeter(window_length=wl)
        else:
            raise ValueError(
                f"Parameter 'meter_type' must be one of ['value', 'running_avg']. Got {type(meter_type)} instead."
            )
        
        self.meters[name] = meter

    def _update_one_meter(self, name: str, val: float, n: int = 1):
        # assumes ValueMeter.update(value, n=1) exists
        self.meters[name].update(val, n)
    
    def _delete_one_meter(self, name: str):
        """ Delete one meter from the set """

        if name in self.meters.keys():
            del self.meters[name]
        else:
            logger.warning(f"Key {name} not found in self.meters.keys().")

    def update(self, val_dict: dict):
        if not isinstance(val_dict, dict):
            raise ValueError("'val_dict' must be a valid dictionary")
        for k, v in val_dict.items():
            self._update_one_meter(name=k, val=v.get('val'), n=v.get('n', 1))

    def reset(self, name: Optional[str] = None):
        """ Reset all Meters in the set """
        if name is not None:
            self[name].reset()
        else:
            for meter in self.meters.values():
                meter.reset()

    def clear(self):
        """ Remove all Meters from the MeterSet """
        self.meters = {}

    def values(self, name: Optional[str] = None, postfix: str = ""):
        if name is not None:
            return {f"{name}_values{('_' + postfix) if postfix else ''}": self[name].values}
        return {f"{n}_values{('_' + postfix) if postfix else ''}": m.values for n, m in self.meters.items()}

    def mins(self, name: Optional[str] = None, postfix: str = ""):
        if name is not None:
            return {f"{name}_min{('_' + postfix) if postfix else ''}": self[name].min}
        return {f"{n}_min{('_' + postfix) if postfix else ''}": m.min for n, m in self.meters.items()}

    def maxs(self, name: Optional[str] = None, postfix: str = ""):
        if name is not None:
            return {f"{name}_max{('_' + postfix) if postfix else ''}": self[name].max}
        return {f"{n}_max{('_' + postfix) if postfix else ''}": m.max for n, m in self.meters.items()}

    def means(self, name: Optional[str] = None, postfix: str = ""):
        if name is not None:

            pass
            # return {f"{name}_mean{(,
                # HausdorffDistance(num_classes=num_classes, input_format='index').to(device)'_' + postfix) if postfix else ''}": self[name].mean}
        # return {f"{n}_mean{('_' + postfix) if postfix else ''}": m.mean for n, m in self.meters.items()}

    def __str__(self):
        lines = [f"{name}: {vm}" for name, vm in sorted(self.meters.items())]
        return "\n".join(lines)

    def __repr__(self):
        pairs = ", ".join(f"{name}: {repr(vm)}" for name, vm in sorted(self.meters.items()))
        return f"ValueMeterSet(meters={{ {pairs} }})"

class MetricLogger:
    """
    Wrapper that computes detection metrics correctly:
    - IoU/GIoU/CIoU/DIoU: computed via torchvision functional ops on
      Hungarian-matched (aligned) pred-target pairs, accumulated with
      MeanMetric. This avoids the torchmetrics class-based detection
      IoU metrics which average the full N×M pairwise matrix (including
      off-diagonal cross-object entries) and produce misleading results.
    - MeanAveragePrecision: uses the torchmetrics class-based API which
      does its own internal greedy matching, so it receives the full
      (unmatched) predictions and targets.
    """
    # Map of metric name -> torchvision functional op
    _IOU_FNS = {
        'iou': box_iou,
        'giou': generalized_box_iou,
        'ciou': complete_box_iou,
        'diou': distance_box_iou,
    }

    def __init__(self, name: str, num_classes: int, device: str):
        """
        Initialize the MetricLogger

        Parameters:
        -----------
            num_classes : int
                The total number of classes to track
            device : torch.device
                The computational device for metric calculation
        """
        self.name = name
        self.device = device
        self.num_classes = num_classes

        # --- Average IoU metrics (across all classes) ---
        self.iou_meters = {
            k: MeanMetric().to(device) for k in self._IOU_FNS
        }

        # --- Per-class IoU metrics ---
        # Dict of {iou_name: {class_idx: MeanMetric}}
        self.iou_class_meters = {
            k: {c: MeanMetric().to(device) for c in range(num_classes)}
            for k in self._IOU_FNS
        }

        # MeanAveragePrecision — class-based, does its own matching internally
        self.map_metric = MeanAveragePrecision(
            iou_type='bbox',
            class_metrics=True,
            average='macro',
        ).to(device)

        self.results = {}

    def update(
        self,
        matched_preds: list[dict[str, torch.Tensor]],
        matched_targets: list[dict[str, torch.Tensor]],
        all_preds: Optional[list[dict[str, torch.Tensor]]] = None,
        all_targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ):
        """
        Update metrics with a batch of predictions and targets.

        Parameters:
        -----------
            matched_preds : list[dict]
                Hungarian-matched predictions (aligned with matched_targets).
            matched_targets : list[dict]
                Hungarian-matched targets (aligned with matched_preds).
            all_preds : list[dict], optional
                Full (unmatched) predictions for MAP. If None, uses matched_preds.
            all_targets : list[dict], optional
                Full (unmatched) targets for MAP. If None, uses matched_targets.
        """
        # --- IoU variants on matched pairs ---
        for pred_dict, tgt_dict in zip(matched_preds, matched_targets):
            p_boxes = pred_dict['boxes']
            t_boxes = tgt_dict['boxes']
            p_labels = pred_dict['labels']
            t_labels = tgt_dict['labels']
            if p_boxes.numel() == 0 or t_boxes.numel() == 0:
                continue
            n = min(len(p_boxes), len(t_boxes))
            for name, fn in self._IOU_FNS.items():
                # Compute full pairwise matrix, take diagonal (matched pairs only)
                iou_matrix = fn(p_boxes[:n], t_boxes[:n])
                diag_scores = iou_matrix.diag()
                # Update average meter
                self.iou_meters[name].update(diag_scores.mean(), weight=n)
                # Update per-class meters (group by target label)
                for c in range(self.num_classes):
                    mask = t_labels[:n] == c
                    if mask.any():
                        class_scores = diag_scores[mask]
                        self.iou_class_meters[name][c].update(
                            class_scores.mean(), weight=int(mask.sum())
                        )

        # --- MAP on full (unmatched) preds/targets ---
        map_preds = all_preds if all_preds is not None else matched_preds
        map_targets = all_targets if all_targets is not None else matched_targets
        self.map_metric.update(map_preds, map_targets)

    def compute(self):
        try:
            results = {}
            # Average IoU metrics
            for name, meter in self.iou_meters.items():
                results[name] = meter.compute()
            # Per-class IoU metrics
            for name in self._IOU_FNS:
                for c, meter in self.iou_class_meters[name].items():
                    # MeanMetric returns 0 if no updates; check weight to skip empty classes
                    try:
                        results[f"{name}/cl_{c}"] = meter.compute()
                    except Exception:
                        results[f"{name}/cl_{c}"] = torch.tensor(float('nan'))
            # MAP metrics (includes per-class MAP via class_metrics=True)
            results.update(self.map_metric.compute())
            self.results = results
        except Exception as e:
            logger.error(f"Encountered error when computing metrics. Error: {e}")
            self.results = None

    def __str__(self) -> str:
        """
        Nicely format the metric results for printing.
        """
        lines = [self.name + ":"]

        if self.results is None or len(self.results) == 0:
            lines.append("   Results: None")
        else:
            for k, v in self.results.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        v = f"{v.item():.5f}"
                    else:
                        v = [round(x, 5) for x in v.flatten().tolist()]
                if isinstance(v, float):
                    lines.append(f"   {k:<20s}: {v:.4f}")
                else:
                    lines.append(f"   {k:<20s}: {v}")

        return "\n" + "\n".join(lines)

    def reset(self):
        """ Reset all metrics """
        for meter in self.iou_meters.values():
            meter.reset()
        for class_meters in self.iou_class_meters.values():
            for meter in class_meters.values():
                meter.reset()
        self.map_metric.reset()
