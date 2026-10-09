"""Target transformation utilities for sequence datasets.

This module provides target transformation functions and classes for
normalizing and scaling target values in chemistry ML datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class ZScoreScaler:
    """Z-score normalization (standardization) scaler for target values.

    Transforms targets using z-score normalization: (x - mean) / std.
    Supports both scalar values and numpy arrays.
    """

    def __init__(self, mean: float | np.ndarray, std: float | np.ndarray) -> None:
        """Initialize ZScoreScaler with precomputed statistics.

        Parameters
        ----------
        mean : float | np.ndarray
            Mean of the target values. Can be a scalar or array for element-wise normalization.
        std : float | np.ndarray
            Standard deviation of the target values. Can be a scalar or array for element-wise normalization.
        """
        self.mean = np.asarray(mean) if isinstance(mean, (list, tuple, np.ndarray)) else mean
        self.std = np.asarray(std) if isinstance(std, (list, tuple, np.ndarray)) else std

    def __call__(self, value: float | int | np.ndarray) -> float | np.ndarray:
        """Transform target value(s) using z-score normalization.

        Parameters
        ----------
        value : float | int | np.ndarray
            Target value(s) to transform. Can be a scalar or numpy array.

        Returns
        -------
        float | np.ndarray
            Z-score normalized target value(s): (value - mean) / std.
            Returns float for scalar input, np.ndarray for array input.
        """
        # Check if we're dealing with arrays (either value is array-like or mean/std are arrays)
        is_array_case = (
            isinstance(value, (np.ndarray, list, tuple)) or
            isinstance(self.mean, np.ndarray) or
            isinstance(self.std, np.ndarray)
        )
        
        if is_array_case:
            value = np.asarray(value, dtype=np.float32)
            mean = np.asarray(self.mean, dtype=np.float32)
            std = np.asarray(self.std, dtype=np.float32)
            # Handle broadcasting: if mean/std are scalars, they broadcast; if arrays, element-wise
            result = (value - mean) / std
            # If result is a scalar (0D array), return as float; otherwise return array
            if result.ndim == 0:
                return float(result)
            return result.astype(np.float32)
        else:
            # Scalar case - both value and mean/std are scalars
            return float((value - self.mean) / self.std)

    def inverse_transform(self, value: float | np.ndarray) -> float | np.ndarray:
        """Inverse transform z-score normalized value(s) back to original scale.

        Parameters
        ----------
        value : float | np.ndarray
            Z-score normalized value(s) to transform back. Can be a scalar or numpy array.

        Returns
        -------
        float | np.ndarray
            Value(s) in original scale: value * std + mean.
            Returns float for scalar input, np.ndarray for array input.
        """
        # Check if we're dealing with arrays (either value is array-like or mean/std are arrays)
        is_array_case = (
            isinstance(value, (np.ndarray, list, tuple)) or
            isinstance(self.mean, np.ndarray) or
            isinstance(self.std, np.ndarray)
        )
        
        if is_array_case:
            value = np.asarray(value, dtype=np.float32)
            mean = np.asarray(self.mean, dtype=np.float32)
            std = np.asarray(self.std, dtype=np.float32)
            # Handle broadcasting: if mean/std are scalars, they broadcast; if arrays, element-wise
            result = value * std + mean
            # If result is a scalar (0D array), return as float; otherwise return array
            if result.ndim == 0:
                return float(result)
            return result.astype(np.float32)
        else:
            # Scalar case - both value and mean/std are scalars
            return float(value * self.std + self.mean)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ZScoreScaler:
        """Create scaler from configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary with "type": "zscore", "mean", and "std" keys.
            Mean and std can be scalars or arrays (lists/tuples/numpy arrays).

        Returns
        -------
        ZScoreScaler
            Scaler instance created from config.
        """
        return cls(mean=config["mean"], std=config["std"])


def create_target_transform(config: dict[str, Any]) -> Callable[[float | int | np.ndarray], float | np.ndarray]:
    """Create target transform from configuration dictionary.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with "type": "zscore", "mean", and "std" keys.
        Mean and std can be scalars or arrays for element-wise normalization.

    Returns
    -------
    Callable[[float | int | np.ndarray], float | np.ndarray]
        Target transform function that accepts scalars or numpy arrays.

    Raises
    ------
    ValueError
        If transform type is not recognized or mean/std are missing.
    """
    transform_type = config.get("type")
    if transform_type not in ("zscore"):
        raise ValueError(f"Unknown transform type '{transform_type}'. Supported types: 'zscore'")

    if "mean" not in config or "std" not in config:
        raise ValueError(f"For '{transform_type}' transform, 'mean' and 'std' must be provided in config")

    return ZScoreScaler.from_config(config)
