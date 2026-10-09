"""Utilities for ensuring reproducibility in machine learning experiments."""

from __future__ import annotations

import os
import random

import torch
from loguru import logger

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility across all random number generators.

    Sets seeds for Python's random module, NumPy, and PyTorch (CPU and CUDA).
    This ensures reproducible results across runs when the same seed is used.

    Parameters
    ----------
    seed : int
        Random seed value to use for all RNGs.
    deterministic : bool, default=False
        If True, enables deterministic mode for PyTorch CUDA operations.
        Note: This may reduce performance and is not always possible depending
        on the CUDA version and operations used. Some operations may not be
        deterministic even with this flag.

    Notes
    -----
    For full reproducibility, also set the PYTHONHASHSEED environment variable
    before running Python:
        export PYTHONHASHSEED=0  # Unix/MacOS
        set PYTHONHASHSEED=0      # Windows CMD
        $env:PYTHONHASHSEED=0     # Windows PowerShell

    Warning
    -------
    Setting deterministic=True may significantly reduce performance and is not
    always possible. Some CUDA operations are inherently non-deterministic.

    Examples
    --------
    >>> from joint_improvement.utils import set_seed
    >>> set_seed(42)
    >>> set_seed(42, deterministic=True)  # For maximum reproducibility
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy random seed (if available)
    if HAS_NUMPY:
        np.random.seed(seed)

    # PyTorch CPU seed
    torch.manual_seed(seed)

    # PyTorch CUDA seeds (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Set deterministic mode if requested
        if deterministic:
            # Check if deterministic operations are available
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)
            elif hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            else:
                # Fallback for older PyTorch versions
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.warning(
                "Deterministic mode enabled. This may reduce performance and "
                "some operations may still be non-deterministic."
            )
        else:
            # Ensure non-deterministic mode is explicitly set (default)
            if hasattr(torch.backends.cudnn, "deterministic"):
                torch.backends.cudnn.deterministic = False
            if hasattr(torch.backends.cudnn, "benchmark"):
                torch.backends.cudnn.benchmark = True

    # Check PYTHONHASHSEED environment variable
    hashseed = os.environ.get("PYTHONHASHSEED")
    if hashseed and hashseed != "0":
        logger.warning(f"PYTHONHASHSEED is set to {hashseed} (not 0). For full reproducibility, set it to 0.")
    logger.info(f"Random seed set to {seed} for all RNGs")


def get_generator(device: str | torch.device, seed: int | None = None) -> torch.Generator:
    """Create a torch.Generator for reproducible random sampling.

    Parameters
    ----------
    device : str | torch.device
        Device to create the generator on (e.g., "cuda", "cpu").
    seed : int, optional
        Random seed for the generator. If None, uses a random seed.

    Returns
    -------
    torch.Generator
        PyTorch generator instance configured with the specified seed and device.

    Examples
    --------
    >>> from joint_improvement.utils import get_generator
    >>> generator = get_generator(device="cuda", seed=42)
    >>> probs = torch.softmax(logits, dim=-1)
    >>> next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    """
    device = torch.device(device) if isinstance(device, str) else device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator
