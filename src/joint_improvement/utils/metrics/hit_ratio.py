"""Hit Ratio metric for evaluating molecular optimization performance.

The Hit Ratio measures the proportion of generated molecules that meet a
specified threshold criterion. This metric is commonly used in molecular
optimization to assess the effectiveness of generative models in producing
molecules with desired properties.

This implementation follows SATURN's Hit Ratio calculation methodology.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculate_hit_ratio(
    objective_scores: np.ndarray | Sequence[float],
    objective_threshold: float,
    qed_scores: np.ndarray | Sequence[float] | None = None,
    sa_scores: np.ndarray | Sequence[float] | None = None,
    qed_threshold: float | None = 0.5,
    sa_threshold: float | None = 5.0,
) -> float:
    """Calculate hit ratio.

    Parameters
    ----------
    objective_scores : np.ndarray | Sequence[float]
        Objective scores (1D).
    objective_threshold : float
        Threshold for objective scores (strictly greater than).
    qed_scores : np.ndarray | Sequence[float] | None
        QED scores (1D).
    sa_scores : np.ndarray | Sequence[float] | None
        SA scores (1D).
    qed_threshold : float | None
        Threshold for QED scores (greater than).
    sa_threshold : float | None
        Threshold for SA scores (less than).

    Returns
    -------
    float
        Fraction of molecules satisfying the objective threshold.
    """
    objective_scores = cast("np.ndarray", np.asarray(objective_scores, dtype=float))
    if objective_scores.ndim != 1:
        raise ValueError("Objective scores must be 1D.")

    hits = (objective_scores < objective_threshold)

    if qed_scores is not None and qed_threshold is not None:
        qed_scores = cast("np.ndarray", np.asarray(qed_scores, dtype=float))
        if qed_scores.ndim != 1:
            raise ValueError("QED scores must be 1D.")
        hits &= (qed_scores > qed_threshold)

    if sa_scores is not None and sa_threshold is not None:
        sa_scores = cast("np.ndarray", np.asarray(sa_scores, dtype=float))
        if sa_scores.ndim != 1:
            raise ValueError("SA scores must be 1D.")
        hits &= (sa_scores < sa_threshold)

    return float(np.sum(hits) / len(objective_scores))
