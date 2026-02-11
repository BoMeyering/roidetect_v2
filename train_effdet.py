"""
train_supervised.py
Main training script for the PGCView V2 semantic segmentation model
BoMeyering 2025
"""

import torch
import os
import logging
import wandb
import omegaconf
import wandb
import torch.distributed as dist
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Local imports
from src.models import EffDetWrapper
from src.datasets import EffdetDataset
from src.trainer import EffdetTrainer
from src.parameters import OptimConfig, EMA
from src.transforms import get_train_transforms, get_val_transforms, set_normalization_values
from src.utils.device import set_torch_device
from src.utils.config import TrainSupervisedConfig, set_run_name
from src.utils.loggers import setup_loggers, rank_log
from src.distributed import set_env_ranks, setup_ddp, shutdown_ddp
from src.callbacks import CheckpointManager

# Create a parser for command line arguments
parser = ArgumentParser(
    prog="train_effdet.py",
    description="Main training script for the PGCView V2 marker detection model."
)
# Add arguments for config file and then parse CLI args
parser.add_argument('-c', '--config', type=str, help="The path to the training config YAML file.", default='configs/train_config.yaml')
parser.add_argument('-b', '--backend', type=str, help="The backend to use for torchrun. Defaults to 'gloo'", default='gloo')
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"The path to the configuration file {args.config} was not found.")

# Set the backend engine for torchrunis_main
backend = args.backend
setup_ddp(backend=backend)

#----------------------------------------#
# Set up configuration objects
#----------------------------------------#
# Read in the configuration file and merge with default dict
yaml_conf = OmegaConf.load(args.config) # Load user supplied config file
default_conf = OmegaConf.structured(TrainSupervisedConfig) # Load the default config structure - to fill in any missing args
conf = OmegaConf.merge(default_conf, yaml_conf) # Any args in yaml_conf will override defaults

# Set the ranks, world size, and is_main
set_env_ranks(conf)

# Append timestamp to run name
set_run_name(conf)

# Set up loggers
setup_loggers(conf)
logger = logging.getLogger()

# Set torch device - will set conf.device as 'TYPE:LOCAL_RANK' e.g. 'cuda:0', 'cpu:2' etc
set_torch_device(conf)

# Set data normalization values
set_normalization_values(conf)

# Initialize Weights and Biases for experiment tracking - only on main process to avoid duplicates
if conf.is_main:
    wandb.init(
        project="roidetect_v2",
        entity="bomeyering-the-land-institute",
        name=conf.model_run,
        config=OmegaConf.to_container(conf, resolve=True),
        sync_tensorboard=True,
    )

def collate_fn(batch):
    """ Collate function to handle batches of data with variable number of boxes per image. """
    images, targets, image_ids = tuple(zip(*batch))

    images = torch.stack(images).float()

    bboxes = [target['bboxes'].float() for target in targets]
    labels = [target['labels'].float() for target in targets]
    img_size = torch.tensor([target['img_size'] for target in targets], dtype=torch.float32)
    img_scale = torch.tensor([target['img_scale'] for target in targets], dtype=torch.float32)

    annotations = {
        'bbox': bboxes,
        'cls': labels,
        'img_size': img_size,
        'img_scale': img_scale,
    }

    return images, annotations, image_ids

#----------------------------------------#
# Main entry point
#----------------------------------------#
def main(conf: omegaconf.OmegaConf=conf):
    """Main function to run the supervised training script

    Run the main training script for supervised training of the PGCView V2 semantic segmentation model.
    Pulls in all of the configurations from the provided config file and sets up the model, datasets, dataloaders,
    optimizer, scheduler, and criterion. Then initializes the SupervisedTrainer class and starts training.

    Parameters:
    -----------
        conf : omegaconf.OmegaConf, optional
            The OmegaConf configuration dictionary, by default conf
    """

    # Log training
    rank_log(conf.is_main, logger.info, "Current Training Configuration\n"+OmegaConf.to_yaml(conf))

    # Create and wrap model for DDP
    model = EffDetWrapper(conf, device=torch.device(conf.device))
    model = DDP(
        model, 
        device_ids=[conf.local_rank] if 'cuda' in conf.device else None, 
        output_device=conf.local_rank if 'cuda' in conf.device else None, 
        find_unused_parameters=True
    )
    rank_log(conf.is_main, logger.info, f"Created model and moved to device {conf.device}")
    if 'cuda' in conf.device:
        rank_log(conf.is_main, logger.info, f"Main process is on {torch.cuda.get_device_name(0)} - {conf.device}")
    else:
        rank_log(conf.is_main, logger.info, f"Main process is on {conf.device}")
    rank_log(conf.is_main, logger.info, f"Total world size: {dist.get_world_size()}")

    # Augmentation Pipelines
    train_transforms = get_train_transforms(
        rgb_means=conf.metadata.norm.means, 
        rgb_stds=conf.metadata.norm.std, 
        resize=conf.images.resize
    )
    val_transforms = get_val_transforms(
        rgb_means=conf.metadata.norm.means, 
        rgb_stds=conf.metadata.norm.std,
        resize=conf.images.resize
    )

    # test_transforms = get_val_transforms(resize=tuple(conf.images.resize))

    # Create Datasets
    train_ds = EffdetDataset(
        conf=conf,
        transforms=train_transforms,
        type='train'
    )

    val_ds = EffdetDataset(
        conf=conf,
        transforms=val_transforms,
        type='val'
    )

    # Create distributed Samplers
    train_sampler = DistributedSampler(
        dataset=train_ds, 
        rank=conf.local_rank, 
        shuffle=True, 
        drop_last=True
    )
    val_sampler = DistributedSampler(
        dataset=val_ds, 
        rank=conf.local_rank, 
        shuffle=False, 
        drop_last=True
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_ds, 
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_ds, 
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
        sampler=val_sampler,
        drop_last=True,
        collate_fn=collate_fn
    )

    # Optimizer
    optim_config = OptimConfig(conf=conf, model=model)
    model, optimizer, scheduler = optim_config.process()

    # Initialize EMA if specified
    if conf.optimizer.ema:
        ema = EMA(model, decay=conf.optimizer.ema_decay, verbose=True)
        rank_log(conf.is_main, logger.info, f"Exponential Moving Average (EMA) enabled with decay rate {conf.optimizer.ema_decay}.")
    else:
        ema = None

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        conf=conf
    )

    # Initialize Trainer
    effdet_trainer = EffdetTrainer(
        name="effdet_trainer",
        conf=conf, 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        ema=ema)

    # Start training
    effdet_trainer.train()

    shutdown_ddp()

if __name__ == '__main__':
    main()