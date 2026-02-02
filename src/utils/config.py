"""
src.utils.config.py
Configuration File Validation script
BoMeyering 2025
"""

import torch
import omegaconf
import logging
import datetime
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from enum import Enum
from typing import List, Optional, Union, Tuple

@dataclass
class Images:
    resize: List[int]=field(default_factory=lambda: [512, 512])

@dataclass
class Directories:
    train_dir: str='data/processed/train/labeled'
    val_dir: str='data/processed/val'
    test_dir: str='data/processed/test'
    annotations_file: str='data/formatted_bboxes.json'
    output_dir: str='outputs'
    checkpoint_dir: str='model_checkpoints'
    log_dir: str='logs/run_logs'

@dataclass
class FlexmatchDirectories:
    train_labeled_dir: str='data/processed/train/labeled'
    train_unlabeled_dir: str='data/processed/train/unlabeled'
    val_dir: str='data/processed/val'
    test_dir: str='data/processed/test'
    output_dir: str='outputs'
    checkpoint_dir: str='model_checkpoints'
    log_dir: str='logs/run_logs'
    starting_checkpoint_path: Optional[str]=None

@dataclass
class Training:
    epochs: int=30
    sanity_check: bool=True

@dataclass
class FlexMatch:
    tau: float=0.95
    mapping: str="linear"
    warmup: bool=True
    lam: float=1.0
    l_batch_size: int=2
    u_batch_size: int=2

@dataclass
class Model:
    backbone_out_channels: Optional[int]=256
    num_classes: Optional[int]=None
    detections_per_img: int=100


@dataclass
class OptimizerParams:
    lr: float=0.001 # Learning rate
    momentum: float=0.9 # Momentum rate
    nesterov: bool=True # Use Nesterov momentum update
    dampening: float=0 # Dampening parameter for SGD
    alpha: float=0.99 # Alpha parameter for RMSprop
    gamma: float=0.99
    etas: Tuple[float]=field(default_factory=lambda: (0.5, 1.2)) # etas for Rprop
    betas: Tuple[float]=field(default_factory=lambda: (0.9, 0.999)) # betas for Adam
    rho: float=0.9 # Rho parameter for Adadelta
    amsgrad: bool=False
    foreach: Optional[bool]=None # Foreach loop flag


@dataclass
class Optimizer:
    name: str='SGD'
    weight_decay: float=0.0001 # Optimizer weight decay
    original_weight_decay: Optional[float]=None # Used internally if filter_bias_and_bn is True
    filter_bias_and_bn: bool=True
    ema: bool=True
    ema_decay: float=0.9
    optimizer_params: OptimizerParams=field(default_factory=OptimizerParams)

@dataclass
class Scheduler:
    """ Currently implemented for ExponentialLR, LinearLR, CosineAnnealingLR, and CosineAnnealingWarmRestarts """
    name: str='ExponentialLR'
    gamma: float=0.99 # Set for default ExponentialLR
    step_size: float=0.00001
    T_max: int=5 # T_max parameter for CosineAnnealingLR
    eta_min: float=0.0 # Default eta_min for CosineAnnealingLR
    T_0: Optional[int]=None # CosineAnnealingWarmRestarts
    T_mult: int=1 # CosineAnnealingWarmRestarts
    last_epoch: int=-1

@dataclass
class Norm:
    means: List[float]=field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float]=field(default_factory=lambda: [0.229, 0.224, 0.225])

@dataclass
class Metadata:
    norm_path: str='metadata/dataset_norm.json'
    norm: Norm=field(default_factory=Norm)
    target_mapping_path: str='metadata/target_mapping.json'
    target_mapping: Optional[dict]=None

@dataclass
class TrainSupervisedConfig:
    model_run: str='model_run'
    device: str='cpu'
    rank: Optional[int]=None
    local_rank: Optional[int]=None
    world_size: Optional[int]=None
    is_main: Optional[bool]=None
    images: Images=field(default_factory=Images)
    metadata: Metadata=field(default_factory=Metadata)
    logging_level: str='INFO'
    directories: Directories=field(default_factory=Directories)
    training: Training=field(default_factory=Training)
    model: Model=field(default_factory=Model)
    optimizer: Optimizer=field(default_factory=Optimizer)
    scheduler: Scheduler=field(default_factory=Scheduler)
    batch_size: int=2
    num_workers: int=2

# @dataclass
# class TrainFlexmatchConfig:
#     model_run: str='flexmatch_model_run'
#     device: str='cpu'
#     rank: Optional[int]=None
#     local_rank: Optional[int]=None
#     world_size: Optional[int]=None
#     is_main: Optional[bool]=None
#     images: Images=field(default_factory=Images)
#     metadata: Metadata=field(default_factory=Metadata)
#     logging_level: str='INFO'
#     tb_exclude_classes: Optional[List[int]]=None
#     directories: FlexmatchDirectories=field(default_factory=FlexmatchDirectories)
#     training: Training=field(default_factory=Training)
#     model: Model=field(default_factory=Model)
#     optimizer: Optimizer=field(default_factory=Optimizer)
#     scheduler: Scheduler=field(default_factory=Scheduler)
#     flexmatch: FlexMatch=field(default_factory=FlexMatch)
#     num_workers: int=2


def set_run_name(conf: OmegaConf):
    """
    Append timestamp to conf.run_name

    Args:
        conf (OmegaConf): OmegaConf configuration dict
    """
    run_name = "_".join([conf.model_run, datetime.datetime.now().isoformat(timespec='seconds', sep='_').replace(":", ".")])
    conf.model_run = run_name