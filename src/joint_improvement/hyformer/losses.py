"""Loss functions for task training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shift_labels: bool = True,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute language modeling loss."""
    logits = logits.contiguous()
    labels = labels.contiguous()

    if logits.ndim != 3:
        raise ValueError(f"Expected logits shape [B, T, V], got {logits.shape}")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels shape [B, T], got {labels.shape}")

    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    B, T = labels.shape
    V = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(B * T, V),
        labels.reshape(B * T),
        ignore_index=ignore_index,
        reduction=reduction,
    )


def compute_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute masked language modeling loss with optional label smoothing."""
    logits = logits.contiguous()
    labels = labels.contiguous()

    if logits.ndim != 3:
        raise ValueError(f"Expected logits shape [B, T, V], got {logits.shape}")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels shape [B, T], got {labels.shape}")

    batch_size, seq_len = labels.shape
    vocab_size = logits.size(-1)

    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    labels_flat = labels.reshape(batch_size * seq_len)

    # Create mask for valid (masked) positions
    valid_mask = labels_flat != ignore_index

    if not valid_mask.any():
        # No masked positions -> return a graph-connected zero (avoids NaNs and keeps dtype/device).
        return logits.sum() * 0.0

    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=float(label_smoothing),
    )

    return loss


def compute_binary_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Single-label classification loss with missing-value masking.

    Expects logits and targets of shape [B, 1].
    """
    if logits.ndim != 2 or logits.size(-1) != 1:
        raise ValueError(f"Expected logits shape [B, 1], got {logits.shape}")
    if targets.ndim != 2 or targets.size(-1) != 1:
        raise ValueError(f"Expected targets shape [B, 1], got {targets.shape}")

    logits = logits.contiguous()
    targets = targets.contiguous()

    # Define is_valid mask: finite values and not equal to ignore_index
    if targets.is_floating_point():
        is_valid = torch.isfinite(targets) & (targets != float(ignore_index))
    else:
        is_valid = targets != ignore_index

    # If all valid, compute loss directly
    if is_valid.all():
        B = logits.shape[0]
        return F.binary_cross_entropy_with_logits(
            logits.reshape(B), targets.reshape(B), reduction=reduction
        )

    # Otherwise, mask invalid entries
    targets_filled = torch.where(is_valid, targets, torch.zeros_like(targets))
    per_sample = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1), targets_filled.squeeze(-1), reduction="none"
    )

    if reduction == "none":
        return torch.where(is_valid.squeeze(-1), per_sample, torch.full_like(per_sample, float("nan")))
    if reduction == "sum":
        return torch.where(is_valid.squeeze(-1), per_sample, torch.zeros_like(per_sample)).sum()
    if reduction == "mean":
        if is_valid.any():
            return per_sample[is_valid.squeeze(-1)].mean()
        return logits.sum() * 0.0
    raise ValueError(f"Unsupported reduction: {reduction}. Choose from 'mean', 'sum', 'none'")


def compute_multilabel_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -1,
) -> torch.Tensor:
    """BCE-with-logits over valid label entries; missing values are ignored.

    Missing values are assumed to be either non-finite (NaN/±Inf) or ignore_index.
    """
    logits = logits.contiguous()
    targets = targets.contiguous()
    if not targets.is_floating_point():
        raise ValueError(f"Multilabel BCE expects floating targets. Got targets dtype={targets.dtype}.")

    # Define is_valid mask: finite values and not equal to ignore_index
    is_valid = torch.isfinite(targets) & (targets != ignore_index)

    # If all valid, compute loss directly
    if is_valid.all():
        return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)

    # Otherwise, mask invalid entries
    targets_filled = torch.where(is_valid, targets, torch.zeros_like(targets))
    per_entry = F.binary_cross_entropy_with_logits(logits, targets_filled, reduction="none")

    if reduction == "none":
        return torch.where(is_valid, per_entry, torch.full_like(per_entry, float("nan")))
    if reduction == "sum":
        return torch.where(is_valid, per_entry, torch.zeros_like(per_entry)).sum()
    if reduction == "mean":
        # DINO-style batch-mean, but make it robust to per-sample label sparsity:
        # average per sample over available labels, then average over samples.
        if is_valid.any():
            per_sample_sum = torch.where(is_valid, per_entry, torch.zeros_like(per_entry)).sum(dim=-1)  # [B]
            per_sample_cnt = is_valid.sum(dim=-1)  # [B]
            has_any = per_sample_cnt > 0
            per_sample_mean = per_sample_sum / per_sample_cnt.clamp_min(1).to(dtype=per_entry.dtype)
            return per_sample_mean[has_any].mean()
        return logits.sum() * 0.0
    raise ValueError(f"Unsupported reduction: {reduction}. Choose from 'mean', 'sum', 'none'")


def compute_regression_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -1
) -> torch.Tensor:
    """Masked MSE regression loss.

    Shapes:
    - logits: [B, D]
    - targets: [B, D]
    """
    if logits.ndim != 2 or targets.ndim != 2:
        raise ValueError(f"Expected logits/targets to be 2D [B, D]. Got logits={logits.shape}, targets={targets.shape}")

    logits = logits.contiguous()
    targets = targets.contiguous()
    if not targets.is_floating_point():
        raise ValueError(f"Regression MSE expects floating targets. Got targets dtype={targets.dtype}.")

    # Define is_valid mask: finite values and not equal to ignore_index
    is_valid = torch.isfinite(targets) & (targets != ignore_index)

    # If all valid, compute loss directly
    if is_valid.all():
        return F.mse_loss(logits, targets, reduction=reduction)

    # Otherwise, mask invalid entries
    targets_filled = torch.where(is_valid, targets, torch.zeros_like(targets))
    per_entry = F.mse_loss(logits, targets_filled, reduction="none")

    if reduction == "none":
        return torch.where(is_valid, per_entry, torch.full_like(per_entry, float("nan")))
    if reduction == "sum":
        return torch.where(is_valid, per_entry, torch.zeros_like(per_entry)).sum()
    if reduction == "mean":
        # Robust per-sample averaging (prevents samples with more valid targets dominating).
        if is_valid.any():
            per_sample_sum = torch.where(is_valid, per_entry, torch.zeros_like(per_entry)).sum(dim=-1)  # [B]
            per_sample_cnt = is_valid.sum(dim=-1)  # [B]
            has_any = per_sample_cnt > 0
            per_sample_mean = per_sample_sum / per_sample_cnt.clamp_min(1).to(dtype=per_entry.dtype)
            return per_sample_mean[has_any].mean()
        return logits.sum() * 0.0
    raise ValueError(f"Unsupported reduction: {reduction}. Choose from 'mean', 'sum', 'none'")


def compute_prediction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    prediction_task_type: str,
    ignore_index: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:

    if prediction_task_type == "regression":
        return compute_regression_loss(logits, targets, reduction=reduction, ignore_index=ignore_index)
    if prediction_task_type == "multilabel_classification":
        return compute_multilabel_classification_loss(logits, targets, reduction=reduction, ignore_index=ignore_index)
    if prediction_task_type == "binary_classification":
        return compute_binary_classification_loss(logits, targets, ignore_index=ignore_index, reduction=reduction)
    raise ValueError(
        f"Unknown prediction_task_type={prediction_task_type!r}. Expected one of: "
        f"'binary_classification', 'multilabel_classification', 'regression'."
    )
