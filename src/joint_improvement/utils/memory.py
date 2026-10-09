"""Utilities for GPU memory management."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer


def free_gpu_memory(
    model: Module | None = None,
    trainer: object | None = None,
    move_model_to_cpu: bool = False,
    clear_optimizer: bool = True,
    clear_cache: bool = True,
) -> None:
    """Free GPU memory by clearing caches and optionally moving model to CPU.
    
    This function helps free up GPU memory after training to make room for
    other GPU-intensive operations (e.g., QuickVina2-GPU docking).
    
    Parameters
    ----------
    model : Module | None, optional
        PyTorch model to potentially move to CPU. If None, no model movement occurs.
    trainer : object | None, optional
        Trainer object that may contain optimizer and other GPU-resident objects.
        If provided and clear_optimizer is True, will attempt to clear optimizer state.
    move_model_to_cpu : bool, default=False
        If True, move the model to CPU. This frees the most memory but requires
        moving it back to GPU if needed for inference.
    clear_optimizer : bool, default=True
        If True, clear optimizer state from trainer. Optimizer state can consume
        significant GPU memory (2x model parameters for Adam).
    clear_cache : bool, default=True
        If True, call torch.cuda.empty_cache() to free cached memory.
    
    Examples
    --------
    >>> from joint_improvement.utils.memory import free_gpu_memory
    >>> # After training, free memory before docking
    >>> free_gpu_memory(model=model, trainer=trainer, move_model_to_cpu=False)
    >>> # Or move model to CPU if not needed for inference
    >>> free_gpu_memory(model=model, trainer=trainer, move_model_to_cpu=True)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU memory cleanup")
        return
    
    # Get initial memory stats
    if torch.cuda.is_available():
        initial_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        initial_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        logger.info(
            f"GPU memory before cleanup: {initial_allocated:.2f} GB allocated, "
            f"{initial_reserved:.2f} GB reserved"
        )
    
    # Clear optimizer state if trainer is provided
    if trainer is not None and clear_optimizer:
        if hasattr(trainer, "optimizer"):
            optimizer: Optimizer | None = trainer.optimizer
            if optimizer is not None:
                # Clear optimizer state dict
                optimizer.state.clear()
                # Delete optimizer reference
                del optimizer
                trainer.optimizer = None  # type: ignore[attr-defined]
                logger.info("Cleared optimizer state")
        
        # Clear any other GPU-resident trainer attributes
        if hasattr(trainer, "scaler") and trainer.scaler is not None:
            # GradScaler doesn't hold much memory, but we can clear it
            del trainer.scaler
            trainer.scaler = None  # type: ignore[attr-defined]
    
    # Move model to CPU if requested
    if model is not None and move_model_to_cpu:
        model_device = next(model.parameters()).device
        if model_device.type == "cuda":
            model = model.cpu()
            logger.info("Moved model to CPU")
    
    # Delete any intermediate tensors and run garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Synchronize to ensure cache is actually cleared
        torch.cuda.synchronize()
        logger.info("Cleared CUDA cache")
    
    # Get final memory stats
    if torch.cuda.is_available():
        final_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        final_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        freed_allocated = initial_allocated - final_allocated
        freed_reserved = initial_reserved - final_reserved
        logger.info(
            f"GPU memory after cleanup: {final_allocated:.2f} GB allocated, "
            f"{final_reserved:.2f} GB reserved"
        )
        logger.info(
            f"Freed: {freed_allocated:.2f} GB allocated, {freed_reserved:.2f} GB reserved"
        )


def move_model_to_device(model: Module, device: torch.device | str) -> Module:
    """Move model to specified device and return it.
    
    Parameters
    ----------
    model : Module
        PyTorch model to move.
    device : torch.device | str
        Target device (e.g., 'cuda', 'cpu', torch.device('cuda:0')).
    
    Returns
    -------
    Module
        Model moved to the specified device.
    """
    device_obj = torch.device(device) if isinstance(device, str) else device
    model = model.to(device_obj)
    logger.info(f"Moved model to {device_obj}")
    return model




