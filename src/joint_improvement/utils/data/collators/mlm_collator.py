"""Collator for masked language modeling (BERT-style) tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import BaseCollatorWithPadding

if TYPE_CHECKING:
    from joint_improvement.tokenizers import SMILESTokenizer


class MLMCollator(BaseCollatorWithPadding):
    """Collator for masked language modeling (BERT-style pretraining).

    Randomly masks tokens in the input sequence and prepares labels for predicting
    the masked tokens. Uses the standard BERT masking recipe:
    - Each token is independently selected for masking with probability ``mlm_probability``
    - Of masked tokens: 80% replaced with [MASK], 10% with random token, 10% unchanged
    - Labels: -100 for non-masked positions, original token ID for masked positions
    - Task token and padding positions are always -100
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        task_token: str = "mlm",
        max_length: int = 128,
        mlm_probability: float = 0.15,
        mlm_seed: int = 42,
    ) -> None:
        """Initialize MLM collator.

        Parameters
        ----------
        tokenizer : SMILESTokenizer
            Tokenizer instance.
        task_token : str, default="mlm"
            Task token name.
        max_length : int, default=128
            Maximum sequence length.
        mlm_probability : float, default=0.15
            Probability of masking each token (standard BERT setting).
        mlm_seed : int, default=42
            Random seed for masking to make data pipeline deterministic.
        """
        super().__init__(tokenizer, task_token=task_token, max_length=max_length)
        self.mlm_probability = mlm_probability
        self.mlm_seed = mlm_seed
        self._generators: dict[str, torch.Generator] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator:
        """Return a torch.Generator seeded per device for deterministic masking."""
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.mlm_seed)
            self._generators[key] = generator
        return self._generators[key]

    def _mask_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply BERT-style masking to input tokens and prepare labels.

        Modifies input_ids in place and returns labels tensor. Uses the standard BERT
        masking recipe:
        - Masking probability sampled per token from a fixed probability
        - Of masked tokens: 80% replaced with [MASK], 10% with random token, 10% unchanged
        - Labels: -100 for non-masked positions, original token ID for masked positions

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape [B, T] (modified in place).
        attention_mask : torch.Tensor
            Attention mask of shape [B, T] (1 for real tokens, 0 for padding).

        Returns
        -------
        torch.Tensor
            Labels tensor with -100 for non-masked positions and original token ID
            for masked positions. Task token position is set to -100.
        """
        labels = input_ids.clone()
        device = input_ids.device
        shape = input_ids.shape
        generator = self._get_generator(device)

        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError(
                "MLM requires tokenizer.mask_token_id to be set, but it is None. "
                "Please configure a mask token in the tokenizer config."
            )

        # Collect special token IDs that should not be masked
        special_token_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.task_token_ids[self.task_token],
            mask_token_id,
        }

        # Create mask for tokens that can be masked (not padding, not special, real tokens)
        can_mask = attention_mask.bool()
        for special_id in special_token_ids:
            can_mask = can_mask & (input_ids != special_id)

        # Apply fixed masking probability across the batch
        probability_matrix = torch.full(shape, self.mlm_probability, device=device)
        masked_indices = can_mask & (torch.rand(shape, device=device, generator=generator) < probability_matrix)
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # Enforce invariants explicitly: task token + padding are always ignored in labels.
        if labels.numel() > 0:
            labels[:, 0] = -100
        labels = labels.masked_fill(attention_mask == 0, -100)

        # 80% of masked tokens: replace with [MASK]
        replace_with_mask = masked_indices & (torch.rand(shape, device=device, generator=generator) < 0.8)
        input_ids[replace_with_mask] = mask_token_id

        # 10% of masked tokens: replace with random token
        # (50% of remaining masked tokens after [MASK] replacement)
        remaining_masked = masked_indices & ~replace_with_mask
        replace_with_random = remaining_masked & (torch.rand(shape, device=device, generator=generator) < 0.5)

        if replace_with_random.any():
            # Generate random tokens, excluding special tokens
            vocab_size = len(self.tokenizer.vocab)
            random_tokens = torch.randint(vocab_size, shape, dtype=torch.long, device=device, generator=generator)

            # Filter out special tokens from random replacements
            for special_id in special_token_ids:
                replace_with_random = replace_with_random & (random_tokens != special_id)

            input_ids[replace_with_random] = random_tokens[replace_with_random]

        # Remaining 10% of masked tokens keep original token (already handled)
        # Task token is already protected from masking and has -100 label via ~masked_indices

        return labels

    def _prepare_labels_and_targets(
        self,
        batch: list[dict[str, Any]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare labels for masked language modeling.

        Applies BERT-style masking to input_ids (in place) and prepares labels.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            Original batch (unused, kept for interface consistency).
        input_ids : torch.Tensor
            Input token IDs tensor (modified in place by masking).
        attention_mask : torch.Tensor
            Attention mask tensor.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Tuple of (labels, None) where labels have -100 for non-masked positions.
        """
        labels = self._mask_tokens(input_ids, attention_mask)
        return labels, None
