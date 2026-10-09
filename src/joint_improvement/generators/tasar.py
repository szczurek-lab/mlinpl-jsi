"""Gumbeldore sampling implementation for sequence generation.

This implementation is based on the Gumbeldore sampling algorithm, which uses
stochastic beam search with advantage-weighted sampling for conditional generation.

References
----------
  Official TASAR implementation: https://github.com/grimmlab/graphxform
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch

from .generator import GeneratorMixin
from .utils.tasar.incremental_sbs import IncrementalSBS

if TYPE_CHECKING:
    from collections.abc import Callable

    from joint_improvement.hyformer.model import Hyformer

    from .utils.tasar.stochastic_beam_search import BeamLeaf, State

_LOGGER = logging.getLogger(__name__)
StateList: TypeAlias = list[torch.Tensor]
StateTensor: TypeAlias = torch.Tensor


class TasarMixin(GeneratorMixin):
    """TASAR (stochastic beam search) generation mixin."""

    def _cast_to_tensor(self, states: StateList) -> torch.Tensor:
        return torch.stack(states, dim=0)

    def _cast_to_states(self, tensor: torch.Tensor) -> StateList:
        return [tensor[batch_idx].squeeze(0) for batch_idx in range(tensor.shape[0])]

    def _build_fn(self, fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        return functools.partial(fn, **kwargs)

    def _child_log_probability_fn(
        self, states: StateList, temperature: float = 1.0, top_k: int | None = None
    ) -> list[np.ndarray]:
        logits = self._get_model_logits(self._cast_to_tensor(states))
        logits = self._transform_logits(logits, temperature=temperature, top_k=top_k)
        log_probs = self._compute_log_probs(logits)
        return self._cast_to_states(log_probs.detach().cpu().numpy())

    def _child_transition_fn(
        self,
        state_action_pairs: list[tuple[StateTensor, int]],
        max_sequence_length: int,
        eos_token_id: int,
    ) -> list[tuple[StateTensor, bool]]:
        assert eos_token_id is not None, "EOS token ID must be provided."
        assert isinstance(eos_token_id, int), "EOS token ID must be an integer."
        new_states: list[tuple[StateTensor, bool]] = []
        for prefix, action in state_action_pairs:
            new_ids = torch.cat(
                [
                    prefix,
                    torch.tensor([action], dtype=torch.long, device=prefix.device),
                ],
                dim=0,
            )
            is_leaf = (int(action) == int(eos_token_id)) or (new_ids.size(0) >= max_sequence_length)
            new_states.append((new_ids, bool(is_leaf)))
        return new_states

    def _leaf_evaluation_fn(
        self,
        input_ids: torch.LongTensor,
        advantage_fn: Callable[[float], float],
        oracle_fn: Callable[[torch.LongTensor], float] | None = None,
    ) -> float:
        objective_fn_evaluation = (
            self._get_model_predictions(input_ids=input_ids) if oracle_fn is None else oracle_fn(input_ids)  # type: ignore[attr-defined]
        )
        return advantage_fn(float(objective_fn_evaluation))

    def _get_sampler(
        self,
        initial_states: StateList,
        child_log_probability_fn: Callable[[StateList], list[np.ndarray]],
        child_transition_fn: Callable[[list[tuple[StateTensor, int]]], list[tuple[StateTensor, bool]]],
        leaf_evaluation_fn: Callable[[StateTensor], float],
    ) -> IncrementalSBS:
        return IncrementalSBS(
            initial_states=initial_states,
            child_log_probability_fn=child_log_probability_fn,
            child_transition_fn=child_transition_fn,
            leaf_evaluation_fn=leaf_evaluation_fn,
            memory_aggressive=False,
        )

    def generate(
        self,
        prefix_input_ids: torch.LongTensor,
        eos_token_id: int,
        max_sequence_length: int = 128,
        advantage_fn: Callable[[float], float] = lambda x: x,
        oracle_fn: Callable[[StateTensor], float] | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        beam_width: int = 32,
        nucleus_top_p: float = 1.0,
        deterministic: bool = False,
        replan_steps: int = 10,
        rng: np.random.Generator | None = None,
    ) -> torch.FloatTensor:
        """Generate sequences using incremental stochastic beam search.

        This method performs conditional sequence generation using the incremental stochastic beam search (SBS) approach
        and TASAR-style log probability updates.

        Parameters
        ----------
        prefix_input_ids : torch.LongTensor
            Input token IDs tensor representing the initial sequence/prompt of shape
            (sequence_length, ) for a single sequence.
        max_sequence_length : int
            Maximum sequence length of the generated sequence.
        advantage_fn : Callable[[float], float]
            Function that calculates the advantage of a given prediction to be maximized.
        eos_token_id : int
            EOS token ID.
        oracle_fn : Optional[Callable[[torch.LongTensor], float]]
            Function that evaluates the leaf states (complete sequences) and returns a float
            score. If None, the model predictions are used to calculate the advantage.
        temperature : float
            Temperature for the logits.
        top_k : Optional[int]
            Top-k value for the logits.
        beam_width : int
            Beam width for one round of SBS.
        nucleus_top_p : float
            Minimum nucleus (top-p) sampling parameter.
        deterministic : bool
            If True, perform deterministic beam search (no Gumbel noise).
        replan_steps : int
            Number of SBS rounds, where we update the log-probs after each round.
        rng : np.random.Generator | None, default=None
            NumPy random number generator for reproducible sampling.

        Returns
        -------
        list[torch.LongTensor]
            List of generated sequences of shape (sequence_length,) for each round.

        Examples
        --------
        >>> rng = np.random.default_rng(42)
        >>> samples = model.generate(..., rng=rng)
        """
        was_training = self.training
        self.eval()

        try:
            child_log_probability_fn = self._build_fn(
                self._child_log_probability_fn,
                temperature=temperature,
                top_k=top_k,
            )

            child_transition_fn = self._build_fn(
                self._child_transition_fn,
                max_sequence_length=max_sequence_length,
                eos_token_id=eos_token_id,
            )

            leaf_evaluation_fn = self._build_fn(
                self._leaf_evaluation_fn,
                advantage_fn=advantage_fn,
                oracle_fn=oracle_fn,
            )

            _sampler = self._get_sampler(
                initial_states=self._cast_to_states(prefix_input_ids.unsqueeze(0)),
                child_log_probability_fn=child_log_probability_fn,
                child_transition_fn=child_transition_fn,
                leaf_evaluation_fn=leaf_evaluation_fn,
            )

            result = _sampler.perform_tasar(
                beam_width=beam_width,
                nucleus_top_p=nucleus_top_p,
                replan_steps=replan_steps,
                deterministic=deterministic,
                rng=rng,
            )

            return self._cast_tasar_result(result)
        finally:
            if was_training:
                self.train()

    def _cast_tasar_result(self, result: list[list[BeamLeaf]]) -> list[list[State]]:
        output_result = result[0]
        return [leaf.state for leaf in output_result]

    @torch.inference_mode()
    def _get_model_logits(self: Hyformer, input_ids: torch.LongTensor) -> torch.FloatTensor:  # type: ignore[misc]
        logits = self.forward(input_ids=input_ids, task="lm")["logits"]  # type: ignore[attr-defined]
        return logits[:, [-1]]  # next token logits

    @torch.inference_mode()
    def _get_model_predictions(self, input_ids: torch.LongTensor) -> float:
        raise NotImplementedError("Subclass must implement this method.")


class TasarMixinLegacy(TasarMixin):  # type: ignore[misc]
    @torch.inference_mode()
    def _get_model_logits(self: Hyformer, input_ids: torch.LongTensor) -> torch.FloatTensor:  # type: ignore[misc]
        logits = self.forward(input_ids=input_ids, attention_mask=None, task="lm", use_cache=False)["logits"]
        return logits

    @torch.inference_mode()
    def _get_model_predictions(  # type: ignore[misc]
        self: Hyformer,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _logits_prediction = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task="prediction",
            **kwargs,
        )["logits"]
        if self.prediction_task_type == "classification":  # type: ignore[attr-defined]
            return torch.sigmoid(_logits_prediction)
        elif self.prediction_task_type == "regression":  # type: ignore[attr-defined]
            return _logits_prediction
        else:
            raise ValueError("Variable `prediction_task_type` must be either `classification` or `regression`.")
