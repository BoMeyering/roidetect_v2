#!/bin/bash

# This script runs a backbone experiment using the specified configuration file.
# It trains the model for a few epochs and saves the checkpoint and configuration.
config_paths=(
    "configs/effdet_d0.yaml"
    "configs/effdet_d1.yaml"
    "configs/effdet_d2.yaml"
    "configs/effdet_d3.yaml"
    "configs/effdet_d4.yaml"
    "configs/effdet_d5.yaml"
)
for config_path in "${config_paths[@]}"; do
    torchrun --standalone --nproc_per_node 2 train_effdet.py --config "$config_path" --backend nccl
done

exit 0;