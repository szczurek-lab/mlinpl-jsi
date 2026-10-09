"""Base collator interface for task-specific collators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from joint_improvement.hyformer.inputs import ModelInput
from joint_improvement.utils.data.dataset import SequenceDataset

if TYPE_CHECKING:
    from joint_improvement.tokenizers import SMILESTokenizer


class BaseCollatorWithPadding(ABC):
    """Base collator with common preprocessing (tokenization, padding, truncation).

    Implements the common preprocessing pipeline and delegates label/target
    preparation to subclasses. All collators that need padding and truncation
    should inherit from this class.
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        task_token: str,
        max_length: int = 128,
    ) -> None:
        """Initialize base collator.

        Parameters
        ----------
        tokenizer : SMILESTokenizer
            Tokenizer instance that contains task_tokens and task_token_ids mappings.
        task_token : str
            Task token name (key in tokenizer.task_tokens), required and cannot be empty.
            For example, use "lm" not "<lm>" or the token ID. The tokenizer will convert
            the name to the actual token ID internally.
        max_length : int, default=128
            Maximum sequence length for truncation.

        Raises
        ------
        ValueError
            If task_token is empty, tokenizer doesn't have task_tokens, or task_token
            is not found in tokenizer.task_tokens.
        """
        if not task_token:
            raise ValueError("task_token is required and cannot be empty")

        if not hasattr(tokenizer, "task_tokens") or not tokenizer.task_tokens:
            raise ValueError("Tokenizer must have task_tokens defined")

        if task_token not in tokenizer.task_tokens:
            available_tokens = list(tokenizer.task_tokens.keys())
            raise ValueError(
                f"Task token '{task_token}' not found in tokenizer.task_tokens. "
                f"Available task tokens: {available_tokens}"
            )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_token = task_token

    def _apply_truncation(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Apply truncation to encoded sequences and attention masks.

        Parameters
        ----------
        input_ids : list[list[int]]
            List of encoded sequences.
        attention_mask : list[list[int]]
            List of attention masks.

        Returns
        -------
        tuple[list[list[int]], list[list[int]]]
            Truncated encoded sequences and attention masks.
        """
        input_ids = [seq[: self.max_length] for seq in input_ids]
        attention_mask = [mask[: self.max_length] for mask in attention_mask]
        return input_ids, attention_mask

    def _apply_padding(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Apply padding to encoded sequences and attention masks.

        Pads sequences to a multiple of 32 for optimal device performance (e.g., Tensor Core
        efficiency on GPUs). This ensures better memory alignment and computational efficiency.

        Parameters
        ----------
        encoded : list[list[int]]
            List of encoded sequences.
        attention_mask : list[list[int]]
            List of attention masks.

        Returns
        -------
        tuple[list[list[int]], list[list[int]]]
            Padded encoded sequences and attention masks.
        """
        pad_token_id = self.tokenizer.pad_token_id
        # Fixed padding: pad to max_length (rounded to 32).
        # This keeps tensor shapes/strides stable across batches, which is important for
        # torch.compile / Inductor (avoids recompiles and stride-mismatch assertions).
        pad_to_length = ((self.max_length + 31) // 32) * 32

        input_ids = [seq + [pad_token_id] * (pad_to_length - len(seq)) for seq in input_ids]
        attention_mask = [mask + [0] * (pad_to_length - len(mask)) for mask in attention_mask]
        return input_ids, attention_mask

    def _prepare_inputs(self, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors from batch (common preprocessing pipeline).

        Handles the common preprocessing steps: extract sequences, tokenize,
        prepend task token, truncate, pad, and convert to tensors.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            List of samples, each containing a "sequence" key.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (input_ids, attention_mask) as tensors.
        """
        sequences = [item[SequenceDataset.SEQUENCE_FIELD] for item in batch]

        tokenized = self.tokenizer(sequences)
        encoded = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Add task token first (before truncation and padding)
        encoded, attention_mask = self._prepend_task_token(encoded, attention_mask)

        # Apply truncation
        encoded, attention_mask = self._apply_truncation(encoded, attention_mask)

        # Apply padding
        encoded, attention_mask = self._apply_padding(encoded, attention_mask)

        # Convert to tensors
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        return input_ids, attention_mask_tensor

    def _prepend_task_token(
        self, encoded: list[list[int]], attention_mask: list[list[int]]
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Prepend task token to encoded sequences and update attention_mask.

        Adds the task token at the beginning of each sequence and ensures
        the attention mask includes the task token (with value 1, meaning
        it's not masked out).

        The task token name is converted to a token ID by looking it up in
        tokenizer.task_token_ids. Task token is always required and validated
        during initialization.

        Parameters
        ----------
        encoded : list[list[int]]
            List of encoded sequences (each is a list of token IDs).
        attention_mask : list[list[int]]
            List of attention masks (each is a list of 1s and 0s).

        Returns
        -------
        tuple[list[list[int]], list[list[int]]]
            Tuple of (encoded_with_task_token, attention_mask_with_task_token).
        """
        # Task token is always required and validated during __init__, so we can directly access it
        task_token_id = self.tokenizer.task_token_ids[self.task_token]

        return (
            [[task_token_id] + seq for seq in encoded],
            [[1] + mask for mask in attention_mask],
        )

    def _to_model_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        labels: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> ModelInput:
        """Convert tokenized tensors to ModelInput.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs.
        attention_mask : torch.Tensor
            Attention mask.
        task : str
            Task name.
        labels : torch.Tensor | None
            Labels tensor (for sequence-level tasks like MLM/LM).
        targets : torch.Tensor | None
            Targets tensor (for prediction tasks like regression/classification).

        Returns
        -------
        ModelInput
            Model input container.
        """
        return ModelInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            targets=targets,
            task=task,
        )

    def __call__(self, batch: list[dict[str, Any]]) -> ModelInput:
        """Collate a batch of samples from SequenceDataset.

        Implements the common preprocessing pipeline and delegates label/target
        preparation to subclasses via `_prepare_labels_and_targets`.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            List of samples from dataset.

        Returns
        -------
        ModelInput
            Model input container with input_ids, attention_mask, labels/targets, and task.
        """
        # Prepare input tensors (common preprocessing)
        input_ids, attention_mask = self._prepare_inputs(batch)

        # Prepare labels and targets (task-specific)
        labels, targets = self._prepare_labels_and_targets(batch, input_ids, attention_mask)

        return self._to_model_input(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            targets=targets,
            task=self.task_token,
        )

    @abstractmethod
    def _prepare_labels_and_targets(
        self,
        batch: list[dict[str, Any]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare labels and targets for the task.

        Subclasses implement this method to prepare task-specific labels or targets.
        For sequence-level tasks (LM, MLM), return labels and None for targets.
        For prediction tasks, return None for labels and targets tensor.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            Original batch from dataset (may contain target values).
        input_ids : torch.Tensor
            Prepared input token IDs tensor.
        attention_mask : torch.Tensor
            Prepared attention mask tensor.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Tuple of (labels, targets) where one should be None depending on task type.
        """
        pass
