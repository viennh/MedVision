"""
Device utilities for GPU assignment with Accelerate.

This module provides common utilities for handling device assignment in single and multi-GPU setups
when using HuggingFace Accelerate.
"""

import torch
from accelerate import Accelerator


def setup_device_with_accelerate(accelerator: Accelerator):
    """
    Set up device and device_map using Accelerate's standard practice.
    
    This function handles GPU assignment for both single and multi-GPU inference
    using Accelerate, which automatically manages device placement across processes.
    
    Args:
        accelerator: An instance of Accelerator
        
    Returns:
        tuple: (device, device_map, rank, world_size)
            - device: torch.device for the current process
            - device_map: string device map (e.g., "cuda:0")
            - rank: process index (0 for single GPU, 0-N for multi-GPU)
            - world_size: total number of processes
            
    Example:
        >>> accelerator = Accelerator()
        >>> device, device_map, rank, world_size = setup_device_with_accelerate(accelerator)
        >>> model = Model(...).to(device)
    """
    device = torch.device(f"cuda:{accelerator.local_process_index}")
    device_map = f"cuda:{accelerator.local_process_index}"
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    
    return device, device_map, rank, world_size
