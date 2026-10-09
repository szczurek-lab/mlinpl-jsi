"""Collator for prediction tasks (classification and regression)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from joint_improvement.utils.data.dataset import SequenceDataset

from .base import BaseCollatorWithPadding

if TYPE_CHECKING:
    from joint_improvement.tokenizers import SMILESTokenizer


class PredictionCollator(BaseCollatorWithPadding):
    """Collator for prediction tasks (classification or regression).

    Handles both discrete class labels (classification) and continuous values (regression).
    Automatically detects label type or uses specified dtype.
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        task_token: str = "prediction",
        max_length: int = 128,
    ) -> None:
        super().__init__(tokenizer, task_token=task_token, max_length=max_length)

    def _prepare_labels_and_targets(
        self,
        batch: list[dict[str, Any]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare targets for prediction tasks.

        Extracts target values from batch and converts them to tensors.
        Handles both classification (int/bool) and regression (float) targets.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            Original batch containing target values under TARGET_FIELD key.
        input_ids : torch.Tensor
            Input token IDs tensor (unused, kept for interface consistency).
        attention_mask : torch.Tensor
            Attention mask tensor (unused, kept for interface consistency).

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Tuple of (None, targets) where targets is a tensor or None if missing.
        """
        # Targets may be missing for inference/prediction-only scenarios
        targets = [item.get(SequenceDataset.TARGET_FIELD) for item in batch]

        targets_tensor: torch.Tensor | None = None
        if any(t is not None for t in targets):
            # Check if all targets are present
            if not all(t is not None for t in targets):
                raise ValueError(
                    "Mixed batch with some samples having targets and others not. "
                    "All samples in a batch must either have targets or not."
                )
            # Determine label dtype and prepare labels
            targets_dtype = torch.long if isinstance(targets[0], (int, bool)) else torch.float32
            targets_tensor = torch.tensor(np.array(targets), dtype=targets_dtype)
            # Ensure targets are 2D: [batch_size, num_targets]
            if len(targets_tensor.shape) == 1:
                targets_tensor = targets_tensor.unsqueeze(1)

        return None, targets_tensor
