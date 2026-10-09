"""Gumbeldore sampling for sequence generation.

This implementation is based on the Gumbeldore sampling, which uses
stochastic beam search with advantage-weighted sampling for sequence generation.

References
----------
  Official Gumbeldore implementation: https://github.com/grimmlab/gumbeldore
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch

from .generator import GeneratorMixin
from .utils.gumbeldore.incremental_sbs import IncrementalSBS
from .utils.linear_scheduler import LinearScheduler

if TYPE_CHECKING:
    from collections.abc import Callable

    from joint_improvement.hyformer.model import Hyformer


_LOGGER = logging.getLogger(__name__)
StateList: TypeAlias = list[torch.Tensor]
StateTensor: TypeAlias = torch.Tensor


class GumbeldoreMixin(GeneratorMixin):
    """Gumbeldore sampling-based generation."""

    def generate(
        self,
        prefix_input_ids: torch.LongTensor,
        max_sequence_length: int,
        advantage_fn: Callable[[float], float],
        eos_token_id: int,
        oracle_fn: Callable[[StateTensor], float] | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        beam_width: int = 32,
        num_rounds: int = 4,
        log_prob_update_type: str = "gumbeldore",
        advantage_constant: float | LinearScheduler = 1.0,
        normalize_advantage_value: bool = True,
        min_nucleus_top_p: float = 1.0,
        rng: np.random.Generator | None = None,
        return_full_info: bool = False,
    ) -> Any:
        """Generate sequences using incremental stochastic beam search.

        This method performs conditional sequence generation using the incremental stochastic beam search (SBS) approach
        and Gumbeldore-style log probability updates.

        Parameters
        ----------
        prefix_input_ids : torch.LongTensor
            Input token IDs tensor representing the initial sequence/prompt of shape
            (sequence_length, ) for a single sequence.
        max_sequence_length : int
            Maximum sequence length of the generated sequence.
        advantage_fn : Callable[[float], float]
            Function that calculates the advantage of a given prediction to be maximized.
            The function should accept a float value and return a float value.
            Example: lambda x: (x - target_value) ** 2.
        eos_token_id : int
            EOS token ID.
        oracle_fn : Optional[Callable[[torch.LongTensor], float]]
            Function that evaluates the leaf states (complete sequences) and returns a float
            score. The function should accept a torch.LongTensor object and return a float value.
            If None, the model predictions are used to calculate the advantage.
        temperature : float
            Temperature for the logits.
        top_k : Optional[int]
            Top-k value for the logits.
        beam_width : int
            Beam width for one round of SBS.
        num_rounds : int
            Number of SBS rounds, where we update the log-probs after each round.
        log_prob_update_type : str, default="gumbeldore"
            Log-probability update rule used inside incremental SBS. This is passed through to
            `IncrementalSBS.perform_incremental_sbs(...)` (e.g. "wor", "gumbeldore", "theory_gumbeldore").
        advantage_constant : float | LinearScheduler, default=1.0
            σ in the paper. Pass a float for a fixed value each round, or :class:`LinearScheduler`
            with ``min_value``, ``max_value``, and ``max_rounds``; round ``r`` uses ``get_value(r)``
            (set ``max_rounds`` equal to ``num_rounds``).
        normalize_advantage_value : bool
            Min-max normalize advantage values, to set the highest/lowest advantage to 1/-1.
        min_nucleus_top_p : float
            Minimum nucleus (top-p) sampling parameter.
        rng : np.random.Generator | None, default=None
            NumPy random number generator for reproducible sampling. If None, uses the global
            NumPy random state (default).
        return_full_info : bool, default=False
            If True, additionally return per-leaf metadata (log-probabilities, round indices,
            ranks within each round).

        Returns
        -------
        list[torch.Tensor] | tuple[list[torch.Tensor], dict[str, list]]
            - If `return_full_info=False`: returns `states` as a list of token-id tensors (leaf states).
            - If `return_full_info=True`: returns `(states, info)` where `info` is a flattened dict with:
              - `log_prob`: list of floats (SBS leaf log-probabilities)
              - `round_idx`: list of ints
              - `rank_in_round`: list of ints

        Raises
        ------
        AttributeError
            If required instance attributes are not set.
        RuntimeError
            If model logits cannot be computed or sampling fails.

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

            result = _sampler.perform_incremental_sbs(
                beam_width=beam_width,
                num_rounds=num_rounds,
                log_prob_update_type=log_prob_update_type,
                advantage_constant=advantage_constant,
                min_max_normalize_advantage=normalize_advantage_value,
                expected_value_use_simple_mean=False,
                use_pure_outcomes=False,
                normalize_advantage_by_visit_count=False,
                perform_first_round_deterministic=False,
                min_nucleus_top_p=min_nucleus_top_p,
                return_round_info=return_full_info,
                rng=rng,
            )

            return self._cast_incremental_sbs_result(
                result=result,
                include_info=return_full_info,
                leaf_evaluation_fn=leaf_evaluation_fn if return_full_info else None,
            )
        finally:
            if was_training:
                self.train()

    @torch.inference_mode()
    def _get_model_logits(self: Hyformer, input_ids: torch.LongTensor) -> torch.FloatTensor:  # type: ignore[misc]
        logits = self.forward(input_ids=input_ids, task="lm")["logits"]  # type: ignore[attr-defined]
        return logits[:, [-1]]

    def _cast_incremental_sbs_result(
        self,
        *,
        result: Any,
        include_info: bool,
        leaf_evaluation_fn: Callable[[StateTensor], float] | None,
    ) -> Any:
        """Cast `perform_incremental_sbs` output.

        - If `include_info=False`: returns `list[torch.Tensor]` (states).
        - If `include_info=True`: returns `(states, info)` where `info` is a flattened dict with
          parallel metadata lists.
        """
        if include_info:
            leaves_batch, round_info_batch = result
            leaves = leaves_batch[0]
            round_info = round_info_batch[0]

            states: list[torch.Tensor] = []
            log_prob: list[float] = []
            round_idx: list[int] = []
            rank_in_round: list[int] = []

            for leaf, (r, k) in zip(leaves, round_info, strict=True):
                states.append(leaf.state)
                log_prob.append(float(leaf.log_probability))
                round_idx.append(int(r))
                rank_in_round.append(int(k))

            return states, {
                "log_prob": log_prob,
                "round_idx": round_idx,
                "rank_in_round": rank_in_round,
            }

        leaves_batch = result
        leaves = leaves_batch[0]
        return [leaf.state for leaf in leaves]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
