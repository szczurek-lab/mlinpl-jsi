"""Unconditional sequence generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from .generator import GeneratorMixin

if TYPE_CHECKING:
    from joint_improvement.hyformer.model import Hyformer


class UnconditionalGeneratorMixin(GeneratorMixin):
    """Unconditional sequence generation mixin."""

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample next token from logits.

        Parameters
        ----------
        logits : torch.Tensor
            Logits tensor of shape [B, vocab_size].
        temperature : float, default=1.0
            Temperature for unconditional generation. Lower values make the distribution more peaked.
        top_k : int | None, default=None
            If provided, only sample from top-k tokens.
        top_p : float | None, default=None
            If provided, use nucleus (top-p) unconditional generation. Should be in (0, 1].
        generator : torch.Generator | None, default=None
            Random number generator for reproducible unconditional generation. If None, uses default random state.

        Returns
        -------
        torch.Tensor
            Sampled token IDs of shape [B].
        """
        logits = self._transform_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        probs = self._compute_probs(logits)
        next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        return next_tokens

    def generate(
        self,
        prefix_input_ids: torch.LongTensor,
        max_sequence_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.LongTensor:
        """Generate sequences using unconditional generation.

        Parameters
        ----------
        prefix_input_ids : torch.LongTensor
            Input token IDs of shape [B, T] where B is batch size and T is sequence length.
        max_sequence_length : int
            Maximum number of new tokens to generate.
        eos_token_id : int
            End-of-sequence token ID. Generation stops when this token is generated.
        temperature : float, default=1.0
            Unconditional generation temperature. Lower values make the distribution more peaked.
        top_k : int | None, default=None
            If provided, only sample from top-k tokens.
        top_p : float | None, default=None
            If provided, use nucleus (top-p) unconditional generation. Should be in (0, 1].
        generator : torch.Generator | None, default=None
            Random number generator for reproducible generation. If None, uses default random state.

        Returns
        -------
        torch.LongTensor
            Generated sequences of shape [B, T + max_sequence_length].
        """
        # Ensure model is in eval mode for inference
        was_training = self.training
        self.eval()

        try:
            batch_size = prefix_input_ids.shape[0]
            device = prefix_input_ids.device

            # Track generated sequences and whether each is finished
            generated = prefix_input_ids.clone()
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_sequence_length):
                # Get logits for next token
                logits = self._get_model_logits(generated)

                # Sample next token
                next_tokens = self._sample_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                )

                # Update generated sequences
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

                # Check for EOS tokens
                if eos_token_id is not None:
                    finished = finished | (next_tokens == eos_token_id)
                    if finished.all():
                        break

            return generated
        finally:
            # Restore original training mode
            if was_training:
                self.train()

    def generate_batch(
        self,
        input_ids: torch.LongTensor,
        num_samples: int,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> list[torch.LongTensor]:
        """Generate multiple sequences using unconditional generation.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs of shape [B, T] where B is batch size and T is sequence length.
        num_samples : int
            Number of samples to generate for each input.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        eos_token_id : int
            End-of-sequence token ID. Generation stops when this token is generated.
        temperature : float, default=1.0
            Unconditional generation temperature. Lower values make the distribution more peaked.
        top_k : int | None, default=None
            If provided, only sample from top-k tokens.
        top_p : float | None, default=None
            If provided, use nucleus (top-p) unconditional generation. Should be in (0, 1].
        generator : torch.Generator | None, default=None
            Random seed for reproducible generation. If provided, creates generators for each sample.

        Returns
        -------
        list[torch.LongTensor]
            List of generated sequences. Each tensor has shape [T + max_new_tokens] where T is the
            input sequence length. The list contains B * num_samples tensors, where B is the batch
            size. Sequences are ordered such that all samples for the first input come first,
            followed by all samples for the second input, etc.
        """
        batch_size = input_ids.shape[0]

        generated_sequences: list[torch.LongTensor] = []
        for _sample_idx in tqdm(range(num_samples), desc="Generating sequences", miniters=1, mininterval=0):
            generated_batch = self.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )

            for batch_idx in range(batch_size):
                generated_sequences.append(generated_batch[batch_idx])

        return generated_sequences

    @torch.inference_mode()
    def _get_model_logits(self: Hyformer, input_ids: torch.LongTensor) -> tuple[torch.FloatTensor, list | None]:
        """Get model logits for next token prediction.

        Works with Hyformer and other backbone models that follow the same interface.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs of shape [B, T].

        Returns
        -------
        tuple[torch.FloatTensor, list | None]
            Tuple containing:
            - Logits tensor of shape [B, vocab_size] for the next token
            - Updated KV caches (or None if not using cache)
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=None,
            task="lm",
        )
        logits = outputs.logits  # [B, T, vocab_size] or [B, vocab_size]

        # Handle both [B, T, vocab_size] and [B, vocab_size] shapes
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # [B, T, vocab_size] -> [B, vocab_size]

        return logits

    def _get_num_layers(self: Hyformer) -> int:
        """Get number of transformer layers for KV cache initialization.

        Returns
        -------
        int
            Number of layers in the model.
        """
        return len(self.blocks)
