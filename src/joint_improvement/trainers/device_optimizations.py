"""Device-specific optimizations for training."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from typing import Any


def _configure_tf32_for_cuda(allow_tf32: bool = True) -> str:
    """Configure TF32 for CUDA."""
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.fp32_precision = "tf32" if allow_tf32 else "ieee"
    torch.backends.cuda.matmul.fp32_precision = "tf32" if allow_tf32 else "ieee"
    torch.backends.cudnn.fp32_precision = "tf32" if allow_tf32 else "ieee"


def setup_device_optimizations(
    config: Any,  # TrainerConfig - using Any to avoid circular import
    device: torch.device | str,
) -> tuple[torch.dtype, str, torch.amp.autocast | nullcontext, torch.amp.GradScaler, str]:
    """Set up device optimizations and mixed precision training.

    Returns
    -------
    tuple[torch.dtype, str, torch.amp.autocast | nullcontext, torch.amp.GradScaler, str]
        ptdtype, device_type, ctx, scaler, compile_mode
    """
    device_obj = torch.device(device) if isinstance(device, str) else device
    device_type = device_obj.type

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map[config.dtype]

    # float16 needs the CUDA GradScaler; bfloat16 autocast also works on MPS
    if config.dtype == "float16" and device_type != "cuda":
        raise RuntimeError(f"{config.dtype} requires CUDA")
    if config.dtype == "bfloat16" and device_type not in ("cuda", "mps"):
        raise RuntimeError(f"{config.dtype} requires CUDA or MPS")

    logger.info(f"Training with dtype: {ptdtype}")

    if device_type == "cuda":
        _configure_tf32_for_cuda(allow_tf32=config.allow_tf32)

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda" and config.dtype == "float16"))

    compile_mode = "default"

    return ptdtype, device_type, ctx, scaler, compile_mode
