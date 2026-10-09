"""Adapted from https://github.com/MolecularTeam/MolStitch/tree/main."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

# maximum tensor size for simple pareto computation
MAX_BYTES = 5e6


def is_non_dominated(
    Y: Tensor,
    maximize: bool = True,
    deduplicate: bool = True,
) -> Tensor:
    r"""Computes the non-dominated front.

    Note: this assumes maximization.

    For small `n`, this method uses a highly parallel methodology
    that compares all pairs of points in Y. However, this is memory
    intensive and slow for large `n`. For large `n` (or if Y is larger
    than 5MB), this method will dispatch to a loop-based approach
    that is faster and has a lower memory footprint.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
            If any element of `Y` is NaN, the corresponding point
            will be treated as a dominated point (returning False).
        maximize: If True, assume maximization (default).
        deduplicate: A boolean indicating whether to only return
            unique points on the pareto frontier.

    Returns
    -------
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
    """
    n = Y.shape[-2]
    if n == 0:
        return torch.zeros(Y.shape[:-1], dtype=torch.bool, device=Y.device)
    el_size = 64 if Y.dtype == torch.double else 32
    if n > 1000 or n**2 * Y.shape[:-2].numel() * el_size / 8 > MAX_BYTES:
        return _is_non_dominated_loop(Y, maximize=maximize, deduplicate=deduplicate)

    is_all_nan = Y.isnan().all(dim=-1)  # edge case: all elements are NaN
    Y1 = Y.unsqueeze(-3)
    Y2 = Y.unsqueeze(-2)
    if maximize:
        dominates = (Y1 >= Y2).all(dim=-1) & (Y1 > Y2).any(dim=-1)
    else:
        dominates = (Y1 <= Y2).all(dim=-1) & (Y1 < Y2).any(dim=-1)
    nd_mask = ~(dominates.any(dim=-1)) & ~is_all_nan
    if deduplicate:
        # remove duplicates
        # find index of first occurrence  of each unique element
        indices = (Y1 == Y2).all(dim=-1).long().argmax(dim=-1)
        keep = torch.zeros_like(nd_mask)
        keep.scatter_(dim=-1, index=indices, value=1.0)
        return nd_mask & keep
    return nd_mask


def _is_non_dominated_loop(
    Y: Tensor,
    maximize: bool = True,
    deduplicate: bool = True,
) -> Tensor:
    r"""Determine which points are non-dominated.

    Compared to `is_non_dominated`, this method is significantly
    faster for large `n` on a CPU and will significant reduce memory
    overhead. However, `is_non_dominated` is faster for smaller problems.

    Args:
        Y: A `(batch_shape) x n x m` Tensor of outcomes.
        maximize: If True, assume maximization (default).
        deduplicate: A boolean indicating whether to only return unique points on
            the pareto frontier.

    Returns
    -------
        A `(batch_shape) x n`-dim Tensor of booleans indicating whether each point is
            non-dominated.
    """
    is_efficient = torch.ones(*Y.shape[:-1], dtype=bool, device=Y.device)
    for i in range(Y.shape[-2]):
        i_is_efficient = is_efficient[..., i]
        if i_is_efficient.any():
            vals = Y[..., i : i + 1, :]
            if maximize:
                update = (Y > vals).any(dim=-1)
            else:
                update = (Y < vals).any(dim=-1)
            # If an element in Y[..., i, :] is efficient, mark it as efficient
            update[..., i] = i_is_efficient.clone()
            # Only include batches where  Y[..., i, :] is efficient
            # Create a copy
            is_efficient2 = is_efficient.clone()
            if Y.ndim > 2:
                # Set all elements in all batches where Y[..., i, :] is not
                # efficient to False
                is_efficient2[~i_is_efficient] = False
            # Only include elements from is_efficient from the batches
            # where Y[..., i, :] is efficient
            is_efficient[is_efficient2] = update[is_efficient2]

    if not deduplicate:
        # Doing another pass over the data to remove duplicates. There may be a
        # more efficient way to do this. One could broadcast this as in
        # `is_non_dominated`, but we loop here to avoid high memory usage.
        is_efficient_dedup = is_efficient.clone()
        for i in range(Y.shape[-2]):
            i_is_efficient = is_efficient[..., i]
            if i_is_efficient.any():
                vals = Y[..., i : i + 1, :]
                duplicate = (vals == Y).all(dim=-1) & i_is_efficient.unsqueeze(-1)
                if duplicate.any():
                    is_efficient_dedup[duplicate] = True
        return is_efficient_dedup

    return is_efficient


def get_all_metrics(
    solutions: np.ndarray,
    eval_metrics: list[str],
    **kwargs: Any,
) -> dict[str, float]:
    """Compute metrics for solutions already filtered to the Pareto front."""
    metrics: dict[str, float] = {}
    if "hypervolume" in eval_metrics and "hv_ref" in kwargs:
        from pymoo.indicators.hv import HV

        hv_indicator = HV(ref_point=kwargs["hv_ref"])
        # `-` cause pymoo assumes minimization
        metrics["hypervolume"] = hv_indicator.do(-solutions)

    if "r2" in eval_metrics and "r2_prefs" in kwargs and "num_obj" in kwargs:
        metrics["r2"] = r2_indicator_set(kwargs["r2_prefs"], solutions, np.ones(kwargs["num_obj"]))

    # if "hsri" in eval_metrics and "num_obj" in kwargs.keys():
    #     # class assumes minimization so transformer to negative problem
    #     hsr_class = HSR_Calculator(lower_bound=-np.ones(kwargs["num_obj"]) - 0.1,
    #                                upper_bound=np.zeros(kwargs["num_obj"]) + 0.1)
    #     # try except cause hsri can run into divide by zero errors
    #     try:
    #         metrics["hsri"], x = hsr_class.calculate_hsr(-solutions)
    #     except:
    #         metrics["hsri"] = 0.
    #     try:
    #         metrics["hsri"] = metrics["hsri"] if type(metrics["hsri"]) is float else metrics["hsri"][0]
    #     except:
    #         metrics["hsri"] = 0.
    return metrics


def r2_indicator_set(
    reference_points: np.ndarray,
    solutions: np.ndarray,
    utopian_point: np.ndarray,
) -> float:
    """Compute R2 indicator value for solutions relative to reference points."""
    min_list: list[float] = []
    for v in reference_points:
        max_list: list[float] = []
        for a in solutions:
            max_list.append(np.max(v * np.abs(utopian_point - a)))
        min_list.append(np.min(max_list))

    v_norm = np.linalg.norm(reference_points)
    r2 = np.sum(min_list) / v_norm

    return float(r2)


def pareto_frontier(
    solutions: np.ndarray | None,
    rewards: np.ndarray,
    maximize: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    pareto_mask = is_non_dominated(torch.tensor(rewards).cpu() if maximize else -torch.tensor(rewards).cpu())
    if solutions is not None:
        if solutions.shape[0] == 1:
            pareto_front = solutions
        else:
            pareto_front = solutions[pareto_mask]
    else:
        pareto_front = None
    if rewards.shape[0] == 1:
        pareto_rewards = rewards
    else:
        pareto_rewards = rewards[pareto_mask]
    return pareto_front, pareto_rewards


def generate_simplex(dims: int, n_per_dim: int) -> np.ndarray:
    spaces = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(dims)]
    return np.array([comb for comb in itertools.product(*spaces) if np.allclose(sum(comb), 1.0)])


def get_pareto_fronts(
    states: np.ndarray | Sequence[str] | None,
    rewards: np.ndarray | Sequence[float],
) -> tuple[np.ndarray | None, np.ndarray]:
    states_array: np.ndarray | None = None
    if states is not None:
        states_array = np.array(states)
    rewards_array = cast("np.ndarray", np.array(rewards))
    if rewards_array.ndim == 1:
        rewards_array = np.expand_dims(rewards_array, 0)

    pareto_candidates, pareto_rewards = pareto_frontier(states_array, rewards_array, maximize=True)
    return pareto_candidates, pareto_rewards


def get_hypervolume(
    states: np.ndarray | Sequence[str] | None,
    rewards: np.ndarray | Sequence[float],
    num_objectives: int,
) -> tuple[float, float]:
    pareto_candidates, pareto_rewards = get_pareto_fronts(states, rewards)
    simplex = generate_simplex(num_objectives, n_per_dim=10)
    mo_metrics = get_all_metrics(
        pareto_rewards, ["hypervolume", "r2"], hv_ref=[0] * num_objectives, r2_prefs=simplex, num_obj=num_objectives
    )
    return mo_metrics["hypervolume"], mo_metrics["r2"]


def calculate_hypervolume(
    normalized_qed_scores: np.ndarray | Sequence[float],
    normalized_sa_scores: np.ndarray | Sequence[float],
    normalized_docking_scores: np.ndarray | Sequence[float],
    sampled_sequences: np.ndarray | Sequence[str] | None = None,
) -> tuple[float, float]:
    all_score = np.stack([normalized_qed_scores, normalized_sa_scores, normalized_docking_scores], axis=1)  # (N, 3)

    # 3) (Optional) keep track of SMILES, as in Oracle.input_offline_data
    smiles = np.asarray(sampled_sequences) if sampled_sequences is not None else None  # shape [N]

    # 4) Get Pareto front and hypervolume
    candidates, pareto_rewards = get_pareto_fronts(smiles, all_score)  # pareto_rewards: [P, 3]
    HV, R2 = get_hypervolume(None, pareto_rewards, 3)
    return HV, float(R2)
