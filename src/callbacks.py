"""
src.callbacks.py
Callback Functions
BoMeyering 2025
"""

import torch
import os
import logging
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from src.utils.loggers import rank_log

class CheckpointManager:
    def __init__(self, conf: OmegaConf, top_k: int = None):
        self.checkpoint_dir = Path(conf.directories.checkpoint_dir)
        self.model_run_name = conf.model_run
        self.logger = logging.getLogger()
        self.conf = conf

        cp = getattr(conf, 'checkpoint', None)
        self.top_k = top_k if top_k is not None else (getattr(cp, 'top_k', 5) if cp else 5)
        self.w_loss = getattr(cp, 'w_loss', 0.1) if cp else 0.1
        self.w_iou  = getattr(cp, 'w_iou',  0.3) if cp else 0.3
        self.w_giou = getattr(cp, 'w_giou', 0.4) if cp else 0.4
        self.w_diou = getattr(cp, 'w_diou', 0.3) if cp else 0.3

        self.top_checkpoints = []       # list of (fitness, filepath), sorted descending
        self.most_recent_checkpoint = []

        self.checkpoint_sub_dir = Path(self.checkpoint_dir) / self.model_run_name
        if not os.path.exists(self.checkpoint_sub_dir):
            rank_log(self.conf.is_main, self.logger.info, f"Creating new checkpoint dir at '{self.checkpoint_sub_dir}'.")
            if self.conf.is_main:
                os.makedirs(self.checkpoint_sub_dir, exist_ok=True)

    @staticmethod
    def _to_scalar(v):
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.item()
        return float(v)

    def _compute_fitness(self, val_loss, iou, giou, diou) -> float:
        """Composite fitness score — higher is better.

        fitness = w_iou * iou + w_giou * giou + w_diou * diou - w_loss * val_loss
        Missing IoU metrics default to 0.0 so the score degrades gracefully when
        predictions don't overlap any ground-truth boxes early in training.
        """
        loss_val = val_loss if val_loss is not None else float('inf')
        iou_val  = iou  if iou  is not None else 0.0
        giou_val = giou if giou is not None else 0.0
        diou_val = diou if diou is not None else 0.0
        return (self.w_iou * iou_val
                + self.w_giou * giou_val
                + self.w_diou * diou_val
                - self.w_loss * loss_val)

    def __call__(self, logs=None):
        """Evaluate fitness and save checkpoint if it ranks in the top-k."""
        epoch    = logs.pop('epoch', None)
        val_loss = self._to_scalar(logs.pop('val_loss', None))
        iou      = self._to_scalar(logs.pop('iou',  None))
        giou     = self._to_scalar(logs.pop('giou', None))
        diou     = self._to_scalar(logs.pop('diou', None))

        fitness = self._compute_fitness(val_loss, iou, giou, diou)

        rank_log(
            self.conf.is_main, self.logger.info,
            f"Epoch {epoch} - fitness={fitness:.6f} "
            f"(val_loss={val_loss}, iou={iou}, giou={giou}, diou={diou})"
        )

        if not self.conf.is_main:
            return None

        # Persist the run config once
        config_filename = self.checkpoint_sub_dir / f"{self.model_run_name}_config.yaml"
        if not os.path.exists(config_filename):
            OmegaConf.save(self.conf, config_filename)

        # Always keep one "latest" checkpoint (rolling — delete previous)
        if self.most_recent_checkpoint:
            prev_path = self.most_recent_checkpoint.pop()[1]
            if os.path.exists(prev_path):
                os.remove(prev_path)
                rank_log(self.conf.is_main, self.logger.info, f"Removed previous latest checkpoint: {prev_path}")

        latest_filename = self.checkpoint_sub_dir / f"{self.model_run_name}_latest_epoch_{epoch}_fitness-{fitness:.6f}.pth"
        torch.save(logs, latest_filename)
        rank_log(self.conf.is_main, self.logger.info, f"Epoch {epoch} saved as latest checkpoint: {latest_filename}")
        self.most_recent_checkpoint.append((epoch, latest_filename))

        # Save to top-k if this fitness beats the current worst in the pool
        worst_fitness = self.top_checkpoints[-1][0] if self.top_checkpoints else float('-inf')
        should_save = len(self.top_checkpoints) < self.top_k or fitness > worst_fitness

        if should_save:
            topk_filename = self.checkpoint_sub_dir / f"{self.model_run_name}_epoch_{epoch}_fitness-{fitness:.6f}.pth"
            torch.save(logs, topk_filename)
            rank_log(self.conf.is_main, self.logger.info,
                     f"Epoch {epoch} - fitness={fitness:.6f} is in top-{self.top_k}. Saved to {topk_filename}")

            self.top_checkpoints.append((fitness, topk_filename))
            self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)  # best first

            if len(self.top_checkpoints) > self.top_k:
                evicted_fitness, evicted_path = self.top_checkpoints.pop()
                if os.path.exists(evicted_path):
                    os.remove(evicted_path)
                    rank_log(self.conf.is_main, self.logger.info,
                             f"Evicted checkpoint with fitness={evicted_fitness:.6f} (no longer in top-{self.top_k}): {evicted_path}")
        else:
            rank_log(self.conf.is_main, self.logger.info,
                     f"Epoch {epoch} - fitness={fitness:.6f} did not improve top-{self.top_k}. Skipping top-k save.")
