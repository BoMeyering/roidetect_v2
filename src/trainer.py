"""
src.trainer.py
Base Trainers Classes
BoMeyering 2025
"""

import torch
import os
import sys
import json
import wandb
import time
import cv2
from glob import glob
import uuid
import logging
import argparse
import numpy as np
from typing import Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
import numpy as np

from typing import Union, Optional, Any, Tuple, List
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from src.models import EffDetWrapper
from src.parameters import EMA, apply_ema
from src.metrics import MetricLogger
from src.distributed import is_main_process
from src.utils.loggers import rank_log
from src.callbacks import CheckpointManager

class Trainer(ABC):
    """Abstract Trainer Class"""

    def __init__(self, name: str, tb_writer: SummaryWriter=None):
        super().__init__()
        self.name = name

    @abstractmethod
    def _train_step(self, batch) -> Tuple[Any, Any]:
        """Implement the train step for one batch"""
        ...

    @abstractmethod
    def _val_step(self, batch) -> Tuple[Any, Any]:
        """Implement the val step for one batch"""
        ...

    @abstractmethod
    def _train_epoch(self, epoch) -> Any:
        """Implement the training method for one epoch"""
        ...

    @abstractmethod
    def _val_epoch(self, epoch) -> Any:
        """Implement the validation method for one epoch"""
        ...

    @abstractmethod
    def train(self):
        """Implement the whole training loop"""
        ...


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        name: str,
        tb_writer: SummaryWriter,
        conf: OmegaConf,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_manager=Optional[CheckpointManager],
        sanity_check: bool=True,
        ema: Optional[EMA]=None,
    ):
        super().__init__(name=name, tb_writer=tb_writer) # Initialize the name and AverageMeterSet
        self.trainer_id = "_".join([name, str(uuid.uuid4())])
        self.conf = conf
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.logger = logging.getLogger()
        self.sanity_check = sanity_check
        self.checkpoint_manager = checkpoint_manager
        self.train_loss_meter = MeanMetric().to(self.conf.device)
        self.val_loss_meter = MeanMetric().to(self.conf.device)

        # Load in target mapping
        # if self.conf.metadata.target_mapping_path:
        #     with open(self.conf.metadata.target_mapping_path, 'r') as f:
        #         self.map_dict = json.load(f)
        #     map_arr = np.zeros((len(self.map_dict), 3)).astype(np.uint8)
        #     for k, v in self.map_dict.items():
        #         idx = v['class_idx']
        #         map_arr[idx] = v['rgb'][::-1]

        #     self.map_arr = map_arr

        # Set up metrics class
        # self.train_metrics = MetricLogger(
        #     name='Train Metrics',
        #     num_classes=self.conf.model.config.classes, 
        #     device=self.conf.device
        # )
        # self.val_metrics = MetricLogger(
        #     name='Validation Metrics',
        #     num_classes=self.conf.model.config.classes, 
        #     device=self.conf.device
        # )

    def _train_step(self, batch: Tuple[Tuple, Tuple, Tuple]) -> Tuple[torch.Tensor, int]:
        """Train over one batch
        
        parameters:
        -----------
            batch : Tuple[torch.Tensor, torch.Tensor]
                A batch of images and targets from the training DataLoader.
        """
        # Unpack batch and send to device
        img_ids, images, targets = batch
        images = list(image.to(self.conf.device, non_blocking=True) for image in images)
        targets = [{k: v.to(self.conf.device, non_blocking=True) for k, v in t.items()} for t in targets]
        # Forward pass through model
        loss_dict = self.model(images, targets)

        # Compute the training loss
        loss = sum(loss for loss in loss_dict.values())

        return loss, len(images)

    def _train_epoch(self, epoch: int):
        """ Traing over one epoch """
        # Put model in training mode and reset meters
        self.model.train()
        self.train_loss_meter.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            colour='yellow', 
            disable=not is_main_process()
        )

        # Iterate through the batches
        for batch_idx, batch in p_bar:
            if batch is None: # Guard against empty batches
                continue
            # Zero the optimizer
            self.optimizer.zero_grad(set_to_none=True)

            # Train one batch and backpropagate the errors
            loss, batch_size = self._train_step(batch)
            loss.backward()

            # Add training loss to MeanMetric (for unified validation loss over all ranks in DDP)
            self.train_loss_meter.update(loss.detach(), weight=batch_size)

            # Step optimizer and update parameters for EMA
            self.optimizer.step()

            if self.ema is not None:
                self.ema.update_params()

            # Update progress bar
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch,
                    epochs=self.conf.training.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.train_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss.item()
                )
            )

            # Tensorboard batch writing
            batch_step = ((epoch-1) * len(self.train_loader)) + batch_idx
            if dist.get_rank() == 0 and batch is not None:
                self.tb_writer.add_scalar(
                    tag="batch_loss/train", scalar_value=loss.item(), global_step=batch_step
                )
        
        # ddp barrier
        dist.barrier()

        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.train_loss_meter.compute().item()

        # Compute epoch metrics and loss
        # self.train_metrics.compute()
        # rank_log(self.conf.is_main, self.logger.info, self.train_metrics)
        
        # Tensorboard epoch logging
        # if dist.get_rank() == 0:
        #     self.tb_writer.add_scalar(
        #         tag="epoch_loss/train", scalar_value=avg_loss, global_step=epoch
        #     )

            # self._tb_log_metrics(
            #     self.train_metrics.results, 
            #     main_tag="train_metrics", 
            #     global_step=epoch, 
            #     exclude_idx=self.conf.tb_exclude_classes
            # )

        return avg_loss

    @torch.no_grad()
    def _val_step(self, batch: Tuple) -> Tuple[torch.Tensor, int]:
        """ Validate over one batch """

        # Unpack batch and send to device
        img_ids, images, targets, _ = batch
        images = list(image.to(self.conf.device, non_blocking=True) for image in images)
        targets = [{k: v.to(self.conf.device, non_blocking=True) for k, v in t.items()} for t in targets]
        # Forward pass through model
        output = self.model(images, targets)

        self.model.train()
        loss_dict = self.model(images, targets)
        self.model.eval()

        # Compute the training loss
        loss = sum(loss for loss in loss_dict.values())

        # Update the validation metrics
        # self.val_metrics.update(preds=preds, targets=targets)

        return loss, output, len(images)
    
    @torch.no_grad()
    def _val_epoch(self, epoch: int):
        """ Validate over one epoch """

        # Put model in eval mode and reset meters
        self.model.eval()
        self.val_loss_meter.reset()

        with apply_ema(self.ema):
            # Set progress bar and unpack batches
            p_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), colour='blue', disable=not is_main_process())

            out_dir = Path(self.conf.directories.output_dir) / self.conf.model_run / "_".join(["epoch", str(epoch)])
            if self.conf.local_rank == 0:
                os.makedirs(out_dir)

            # Iterate through the batches
            with torch.inference_mode():  
                for batch_idx, batch in p_bar:
                    if batch is None: # Guard against empty batches
                        continue
                    # Validate one batch
                    loss, output, batch_size = self._val_step(batch)

                    # Add validation loss to MeanMetric (for unified validation loss over all ranks in DDP)
                    self.val_loss_meter.update(loss.detach(), weight=batch_size)

                    # Update the progress bar
                    p_bar.set_description(
                        "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                            epoch=epoch,
                            epochs=self.conf.training.epochs,
                            batch=batch_idx + 1,
                            iter=len(self.val_loader),
                            lr=self.scheduler.get_last_lr()[0],
                            loss=loss.item()
                        )
                    )
                    
                    self._sanity_check(batch, output, epoch)

                    # Tensorboard batch writing
                    # batch_step = ((epoch-1) * len(self.val_loader)) + batch_idx
                    # if dist.get_rank() == 0:
                    #     self.tb_writer.add_scalar(
                    #         tag="batch_loss/val", scalar_value=loss.item(), global_step=batch_step
                    #     )
        # ddp barrier
        dist.barrier()

        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.val_loss_meter.compute().item()

        # Compute epoch metrics
        # self.val_metrics.compute()
        # rank_log(self.conf.is_main, self.logger.info, self.val_metrics)

        # Tensorboard epoch logging
        # if dist.get_rank() == 0:
        #     self.tb_writer.add_scalar(
        #         tag="epoch_loss/val", scalar_value=avg_loss, global_step=epoch
        #     )
        #     self._tb_log_metrics(
        #         self.val_metrics.results, 
        #         main_tag="val_metrics", 
        #         global_step=epoch,
        #         exclude_idx=self.conf.tb_exclude_classes
        #     )

        return avg_loss

    def _sanity_check(self, batch, output, epoch):
        """ Run a sanity check for the model """
        
        image_ids, _, _, raw_images = batch

        batch_images = [cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (1024, 1024)) for img in raw_images]
        batch_boxes = [o['boxes'].detach().cpu().numpy() for o in output]
        batch_scores = [s['scores'].detach().cpu().numpy() for s in output]
        batch_labels = [l['labels'].detach().cpu().numpy() for l in output]

        image_sizes = [img.shape[:2] for img in batch_images]  # (height, width)

        for img_id, img, boxes, scores, labels in zip(image_ids, batch_images, batch_boxes, batch_scores, batch_labels):
            for box, score, label in zip(boxes, scores, labels):
                if score >= 0.15:  # Only draw boxes with confidence >= 0.5
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            out_dir = Path(self.conf.directories.output_dir) / self.conf.model_run / "_".join(["epoch", str(epoch)])
            if self.conf.local_rank == 0:
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(str(Path(out_dir) / f"{Path(img_id).stem}_outbox.jpg"), img)

        # Iterate through the batches
        # with torch.inference_mode():  
        #     for batch_idx, batch in p_bar:
        #         if batch_idx % 10 == 0:
        #             # Unpack batch and send to device
        #             img, targets, img_keys, raw_images = batch
        #             inputs = img.to(self.conf.device, non_blocking=True)
        #             targets = targets.long().to(self.conf.device, non_blocking=True)

        #             # Forward pass through model
        #             logits = self.model(inputs)

        #             maps = torch.argmax(logits, dim=1)

        #             # for i, img in enumerate(maps):
        #             for img_key, img in zip(img_keys, maps):
        #                 img = img.detach().cpu().numpy().astype(np.uint8)
        #                 if getattr(self, 'map_arr', None) is not None:
        #                     img = self.map_arr[img]
        #                 else:
        #                     img *= 20 # Scale outputs to make class distinction clear

        #                 cv2.imwrite(str(Path(out_dir) / f"{Path(img_key).stem}_outmap.png"), img)
                    
        #             # Update the progress bar
        #             p_bar.set_description(
        #                 "Sanity Check: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}.".format(
        #                     epoch=epoch,
        #                     epochs=self.conf.training.epochs,
        #                     batch=batch_idx + 1,
        #                     iter=len(self.val_loader)
        #                 )
        #             )

    # def _tb_log_metrics(self, metric_dict: dict, main_tag: str, global_step: int, exclude_idx: Optional[List[int]]=None):
    #     """ Log metrics from a metric dictionary to TensorBoard """
    #     for type, v in metric_dict.items(): # type is 'avg' or 'mc'
    #         if type == 'avg':
    #             for mk, mv in v.items(): # mk is the metric name, mv is the metric value as a torch.Tensor
    #                 self.tb_writer.add_scalar(f"{main_tag}/avg_{mk}", mv.item(), global_step=global_step)
    #         elif type == 'mc':
    #             metric_map = {data['class_idx']: cname for cname, data in self.map_dict.items()} # Create a mapping from class index (int) to class name (str)
    #             for mk, mv in v.items():
    #                 scalar_dict = {metric_map.get(i): t.item() for i, t in enumerate(mv) if i not in exclude_idx} # Map the tensor values to a new dict with class names as keys
    #                 for sk, sv in scalar_dict.items():
    #                     self.tb_writer.add_scalar(f"{main_tag}/{sk}_{mk}", sv, global_step=global_step)

    def train(self):
        """ Train the model """
        rank_log(self.conf.is_main, self.logger.info, f"Training {self.trainer_id} for {self.conf.training.epochs} epochs.")

        for epoch in range(1, self.conf.training.epochs + 1):
            # Train and validate one epoch
            rank_log(self.conf.is_main, self.logger.info, f"TRAINING EPOCH {epoch}")
            train_loss = self._train_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            rank_log(self.conf.is_main, self.logger.info, f"VALIDATING EPOCH {epoch}")
            val_loss = self._val_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            # Logger Logging
            rank_log(
                self.conf.is_main,
                self.logger.info, 
                f"Epoch {epoch} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
            )

            with apply_ema(self.ema):
                # Create checkpoint logs
                ema_state_dict = self.model.module.state_dict()

            chkpt_logs = {
                "epoch": epoch,
                "val_loss": torch.tensor(val_loss),
                "model_state_dict": self.model.module.state_dict(),
                "ema_state_dict": ema_state_dict,
            }

            self.checkpoint_manager(logs=chkpt_logs)

            # Step LR scheduler
            if self.scheduler:
                self.scheduler.step()


class EffdetTrainer(Trainer):
    def __init__(
        self,
        name: str,
        conf: OmegaConf,
        model: EffDetWrapper,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_manager=Optional[CheckpointManager],
        sanity_check: bool=True,
        ema: Optional[EMA]=None,
    ):
        super().__init__(name=name) # Initialize the name and AverageMeterSet
        self.trainer_id = "_".join([name, str(uuid.uuid4())])
        self.conf = conf
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.logger = logging.getLogger()
        self.sanity_check = sanity_check
        self.checkpoint_manager = checkpoint_manager
        self.train_loss_meter = MeanMetric().to(self.conf.device)
        self.val_loss_meter = MeanMetric().to(self.conf.device)
        self.val_metric_logger = MetricLogger(
            name='Validation Metrics',
            num_classes=self.conf.model.num_classes,
            device=self.conf.device
        )

        # Initialize Weights and Biases for experiment tracking - only on main process to avoid duplicates
        if conf.is_main:
            self.run = wandb.init(
                project="roidetect_v2",
                entity="bomeyering-the-land-institute",
                name=conf.model_run,
                config=OmegaConf.to_container(conf, resolve=True),
                sync_tensorboard=True,
            )
        else:
            self.run = None
        
    def _unwrap_ddp(self):
        return self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

    def _order_preds_by_targets(self, pred_boxes, target_boxes):
        """ Order a set of predicted boxes by their best matching target box using the Hungarian algorithm. """
        
        # Compute bbox IOU matrix
        box_iou_matrix = box_iou(pred_boxes, target_boxes)

        # Linear sum assignment to find the best matching between preds and targets
        row_ind, col_ind = linear_sum_assignment(-box_iou_matrix.cpu().numpy())

        return row_ind, col_ind
    
    def _format_preds_and_targets(self, preds, targets):
        """
        Format preds and targets into a unified format for metric logging.

        Returns:
            matched_preds, matched_targets: Hungarian-matched pairs for IoU metrics.
            all_preds, all_targets: Full (unmatched) lists for MeanAveragePrecision.
        """
        matched_preds = []
        matched_targets = []
        all_preds = []
        all_targets = []

        for pred_array, target_boxes, target_labels in zip(preds, targets['bbox'], targets['cls']):
            # Extract prediction arrays
            pred_boxes = pred_array[:, :4].detach().float()
            pred_scores = pred_array[:, 4].detach().float()
            pred_labels = pred_array[:, 5].detach().int() - 1  # Convert to zero-based class indices
            
            # Extract and format the targets
            target_boxes = target_boxes[:, [1, 0, 3, 2]].float()  # Convert from yxyx to xyxy
            target_labels = target_labels.int() - 1  # Convert to zero-based class indices

            # Always append full (unmatched) preds/targets for MAP
            all_preds.append({
                'boxes': pred_boxes.to(self.conf.device),
                'labels': pred_labels.to(self.conf.device),
                'scores': pred_scores.to(self.conf.device),
            })
            all_targets.append({
                'boxes': target_boxes.to(self.conf.device),
                'labels': target_labels.to(self.conf.device),
            })

            # If an image has no ground truth boxes, append empty matched entries
            if len(target_boxes) == 0:
                matched_preds.append({
                    'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.conf.device),
                    'labels': torch.zeros((0,), dtype=torch.int32, device=self.conf.device),
                    'scores': torch.zeros((0,), dtype=torch.float32, device=self.conf.device),
                })
                matched_targets.append({
                    'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.conf.device),
                    'labels': torch.zeros((0,), dtype=torch.int32, device=self.conf.device),
                })
                continue
            
            # Reorder preds and targets according to the best matching
            row_ind, col_ind = self._order_preds_by_targets(pred_boxes, target_boxes)

            # Append Hungarian-matched preds and targets
            matched_preds.append({
                'boxes': pred_boxes[row_ind].to(self.conf.device),
                'labels': pred_labels[row_ind].to(self.conf.device),
                'scores': pred_scores[row_ind].to(self.conf.device),
            })
            matched_targets.append({
                'boxes': target_boxes[col_ind].to(self.conf.device),
                'labels': target_labels[col_ind].to(self.conf.device),
            })

        return matched_preds, matched_targets, all_preds, all_targets

    def _train_step(self, batch) -> Tuple[Any, Any]:
        """Train over one batch
        
        parameters:
        -----------
            batch : Tuple[torch.Tensor, torch.Tensor]
                A batch of images and targets from the training DataLoader.
        """
        
        images, targets, img_ids = batch
        images = images.to(self.conf.device, non_blocking=True)
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v.to(self.conf.device, non_blocking=True)
            elif isinstance(v, list):
                targets[k] = [item.to(self.conf.device, non_blocking=True) for item in v]

        loss_dict = self._unwrap_ddp().forward_train(images, targets)

        loss = sum(loss for loss in loss_dict.values())

        return loss, len(images)

    def _train_epoch(self, epoch) -> Any:
        """Train over one epoch
        
        parameters:
        -----------
            epoch : int
                The current epoch number.
        """

        self._unwrap_ddp().train_mode()
        self.train_loss_meter.reset()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            colour='yellow', 
            disable=not is_main_process()
        )
        for batch_idx, batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            # Train one batch and backpropagate the errors
            loss, batch_size = self._train_step(batch)
            loss.backward()

            # Add training loss to MeanMetric (for unified validation loss over all ranks in DDP)
            self.train_loss_meter.update(loss.detach(), weight=batch_size)

            # Step optimizer and update parameters for EMA
            self.optimizer.step()

            if self.ema is not None:
                self.ema.update_params()

            # Update progress bar
            pbar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch,
                    epochs=self.conf.training.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.train_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss.item()
                )
            )
            
            dist.barrier() # DDP barrier to sync after each batch
        
        # ddp barrier
        dist.barrier()

        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.train_loss_meter.compute().item()
        if self.run is not None:
            self.run.log({"epoch_loss/train": avg_loss})

        return avg_loss
        
    @torch.no_grad()
    def _val_step(self, batch) -> Tuple[Any, Any]:
        """Validate over one batch
        
        parameters:
        -----------
            batch : Tuple[torch.Tensor, torch.Tensor]
                A batch of images and targets from the validation DataLoader.
        """

        images, targets, img_ids = batch
        images = images.to(self.conf.device, non_blocking=True)
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v.to(self.conf.device, non_blocking=True)
            elif isinstance(v, list):
                targets[k] = [item.to(self.conf.device, non_blocking=True) for item in v]

        # Get predictions
        preds = self._unwrap_ddp().predict(images)
        # Get validation loss

        self._unwrap_ddp().train_mode()
        loss_dict = self._unwrap_ddp().forward_train(images, targets)
        self._unwrap_ddp().eval_mode()
        # Sum the losses
        loss = sum(loss for loss in loss_dict.values())

        # Format preds and targets for metric logging
        matched_preds, matched_targets, all_preds, all_targets = self._format_preds_and_targets(preds, targets)

        # Update the validation metrics
        self.val_metric_logger.update(
            matched_preds=matched_preds,
            matched_targets=matched_targets,
            all_preds=all_preds,
            all_targets=all_targets,
        )

        return loss, matched_preds, len(images)
    
    @torch.no_grad()
    def _val_epoch(self, epoch) -> Any:
        """Validate over one epoch
        
        parameters:
        -----------
            epoch : int
                The current epoch number.
        """

        self._unwrap_ddp().eval_mode()
        self.val_loss_meter.reset()
        self.val_metric_logger.reset()

        with apply_ema(self.ema):
            # Set progress bar and unpack batches
            p_bar = tqdm(
                enumerate(self.val_loader), 
                total=len(self.val_loader), 
                colour='blue', 
                disable=not is_main_process()
            )

            # out_dir = Path(self.conf.directories.output_dir) / self.conf.model_run / "_".join(["epoch", str(epoch)])
            # if self.conf.local_rank == 0:
            #     os.makedirs(out_dir)

            # Iterate through the batches
            with torch.inference_mode():  
                for batch_idx, batch in p_bar:
                    if batch is None: # Guard against empty batches
                        continue
                    # Validate one batch
                    loss, preds, batch_size = self._val_step(batch)

                    # Add validation loss to MeanMetric (for unified validation loss over all ranks in DDP)
                    self.val_loss_meter.update(loss.detach(), weight=batch_size)

                    # Update the progress bar
                    p_bar.set_description(
                        "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                            epoch=epoch,
                            epochs=self.conf.training.epochs,
                            batch=batch_idx + 1,
                            iter=len(self.val_loader),
                            lr=self.scheduler.get_last_lr()[0],
                            loss=loss.item()
                        )
                    )

                    dist.barrier() # DDP barrier to sync after each batch

        # ddp barrier
        dist.barrier()
        
        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.val_loss_meter.compute().item()

        # Compute validation epoch metrics
        self.val_metric_logger.compute()
        rank_log(self.conf.is_main, self.logger.info, self.val_metric_logger)

        return avg_loss

    def train(self):
        """ Train the model """
        rank_log(self.conf.is_main, self.logger.info, f"Training {self.trainer_id} for {self.conf.training.epochs} epochs.")

        for epoch in range(1, self.conf.training.epochs + 1):
            # Train and validate one epoch
            rank_log(self.conf.is_main, self.logger.info, f"TRAINING EPOCH {epoch}")
            train_loss = self._train_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            rank_log(self.conf.is_main, self.logger.info, f"VALIDATING EPOCH {epoch}")
            val_loss = self._val_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            # Logger Logging
            rank_log(
                self.conf.is_main,
                self.logger.info, 
                f"Epoch {epoch} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
            )

            # Wandb logging
            if self.run is not None:
                wandb_log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                wandb_log_dict |= self.val_metric_logger.results
                self.run.log(wandb_log_dict)

            with apply_ema(self.ema):
                # Create checkpoint logs
                ema_state_dict = self.model.module.state_dict()

            chkpt_logs = {
                "epoch": epoch,
                "train_loss": torch.tensor(train_loss),
                "val_loss": torch.tensor(val_loss),
                "model_state_dict": self.model.module.state_dict(),
                "ema_state_dict": ema_state_dict,
            }

            self.checkpoint_manager(logs=chkpt_logs)

            # Step LR scheduler
            if self.scheduler:
                self.scheduler.step()

        rank_log(self.conf.is_main, self.logger.info, f"Training of {self.trainer_id} completed.")