"""Data I/O utilities for loading and saving NPZ archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


def load_npz_data(
    path: Path | str,
    sequence_key: str | None = None,
    target_key: str | None = None,
    verbose: bool = False,
    **additional_keys: Any,
) -> dict[str, Any]:
    """Load sequences, optionally targets, and additional keys from an NPZ archive.

    Parameters
    ----------
    path : Path | str
        Path to the NPZ archive file. Can be a Path object or string.
    sequence_key : str, optional
        Key name for the sequence data in the NPZ archive.
    target_key : str, optional
        Key name for the target data in the NPZ archive.
    verbose : bool, default=False
        If True, log summary information (length, dtype, shape) for each loaded array.
    **additional_keys : Any, optional
        Additional keys to load from the NPZ archive. Each keyword argument
        name should be the key name in the NPZ file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing all loaded data.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    KeyError
        If any specified key (sequence_key, target_key, or additional_keys) is not found in the archive.

    Examples
    --------
    >>> data = load_npz_data(
    ...     "data.npz",
    ...     sequence_key="sequences",
    ...     target_key="targets",
    ... )
    >>> sequences = data["sequences"]
    >>> targets = data["targets"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path, allow_pickle=False)
    result: dict[str, Any] = {}

    # Load sequences, if requested
    if sequence_key is not None:
        if sequence_key not in data:
            raise KeyError(f"Sequence key '{sequence_key}' not found in {path}. Available keys: {list(data.keys())}")
        arr = data[sequence_key]
        if verbose:
            logger.info(f"Loaded '{sequence_key}': length={len(arr)}, dtype={arr.dtype}, shape={arr.shape}")
        result[sequence_key] = arr.tolist()

    # Load targets, if requested
    if target_key is not None:
        if target_key not in data:
            raise KeyError(f"Target key '{target_key}' not found in {path}. Available keys: {list(data.keys())}")
        arr = data[target_key]
        if verbose:
            logger.info(f"Loaded '{target_key}': length={len(arr)}, dtype={arr.dtype}, shape={arr.shape}")
        # Keep as numpy array (not list) so forked DataLoader workers read from a C buffer
        # rather than Python objects, avoiding copy-on-write page faults under high worker counts.
        result[target_key] = arr

    # Load additional keys
    for key_name in additional_keys.keys():
        if key_name not in data:
            raise KeyError(f"Additional key '{key_name}' not found in {path}. Available keys: {list(data.keys())}")
        arr = data[key_name]
        if verbose:
            logger.info(f"Loaded '{key_name}': length={len(arr)}, dtype={arr.dtype}, shape={arr.shape}")
        result[key_name] = arr

    return result


def save_npz_data(path: Path | str, data: dict[str, Any], verbose: bool = False) -> None:
    """Save data to an NPZ archive.

    Parameters
    ----------
    path : Path | str
        Path where the NPZ archive will be saved. Can be a Path object or string.
    data : dict[str, Any]
        Dictionary of data to save. Keys will be used as array names in the archive.
        Values should be numpy arrays or array-like objects.
    verbose : bool, default=False
        If True, log summary information (length, dtype, shape) for each array being saved.

    Examples
    --------
    >>> import numpy as np
    >>> from pathlib import Path
    >>> save_npz_data(
    ...     Path("data.npz"), {"sequences": np.array(["ACDEFGHIK", "LMNPQRSTV"]), "targets": np.array([0.5, 0.8])}
    ... )
    >>> save_npz_data("output.npz", {"sequences": np.array(["ACDEFGHIK"]), "targets": np.array([0.5])})
    """
    path = Path(path)

    # Log summary of data being saved
    if verbose:
        for key, value in data.items():
            arr = np.asarray(value)
            logger.info(f"Saving '{key}': length={len(arr)}, dtype={arr.dtype}, shape={arr.shape}")

    np.savez(path, **data)
