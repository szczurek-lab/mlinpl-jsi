from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from torch.utils.data import Dataset

from joint_improvement.utils.config import BaseConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class SequenceDatasetConfig(BaseConfig):
    """SequenceDataset configuration.

    Parameters
    ----------
    data_path : Path | str
        Path to the NPZ archive file.
    sequence_key : str
        Key name for the sequence data in the NPZ archive.
    target_key : str | None, optional
        Key name for the target data in the NPZ archive.
    transforms : Sequence[dict[str, Any]], optional
        Sequence of transform configuration dictionaries.
    target_transforms : Sequence[dict[str, Any]], optional
        Sequence of target transform configuration dictionaries.

    """

    data_path: Path | str
    sequence_key: str
    target_key: str | None = None
    transforms: Sequence[dict[str, Any]] | None = None
    target_transforms: Sequence[dict[str, Any]] | None = None


class SequenceDataset(Dataset):
    """Minimal PyTorch Dataset.

    Follows PyTorch conventions by exposing `transforms` and `target_transforms`
    applied on-the-fly during `__getitem__`.
    """

    SEQUENCE_FIELD: str = "sequence"
    TARGET_FIELD: str = "targets"

    def __init__(
        self,
        sequences: Sequence[str],
        targets: Sequence[float | int] | None = None,
        transforms: Sequence[Callable[[str], str]] | None = None,
        target_transforms: Sequence[Callable[[float | int], float | int]] | None = None,
    ) -> None:
        """Initialize SequenceDataset.

        Parameters
        ----------
        sequences : Sequence[str]
            Input sequences. Must have the same length as `targets`.
        targets : Sequence[float | int], optional
            Target values (labels for classification, continuous values for regression).
        transforms : Sequence[Callable[[str], str]], optional
            Optional sequence of transforms applied to each sequence during `__getitem__`.
            Transforms are applied in order and must return strings (e.g., for data augmentation).
            Tokenization is handled separately in collators, not in transforms.
        target_transforms : Sequence[Callable[[float | int], float | int]], optional
            Optional sequence of transforms applied to each target during `__getitem__`.
            Transforms are applied in order.

        Raises
        ------
        ValueError
            If `sequences` and `targets` have different lengths.

        """
        if targets is not None and len(sequences) != len(targets):
            raise ValueError("`sequences` and `targets` must have the same length.")

        self._sequences = list(sequences)
        self._targets = np.asarray(targets) if targets is not None else None
        self.transforms = transforms
        self.target_transforms = target_transforms

    @property
    def sequences(self) -> list[str]:
        """Return the sequences.

        Returns
        -------
        list[str]
            Sequences.
        """
        return self._sequences

    @property
    def targets(self) -> np.ndarray | None:
        """Return the targets.

        Returns
        -------
        np.ndarray | None
            Targets, or None if no targets are loaded.
        """
        return self._targets

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self._sequences)

    def __getitem__(self, index: int) -> dict[str, float | int | str]:
        """Get a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve (0-based).

        Returns
        -------
        dict[str, float | int | str]
            Dictionary containing:
            - `SEQUENCE_FIELD`: The sequence (possibly transformed)
            - `TARGET_FIELD`: The target (possibly transformed)

        """
        sequence = self._sequences[index]
        target: float | int | None = self._targets[index] if self._targets is not None else None

        if self.transforms is not None:
            for transform in self.transforms:
                sequence = transform(sequence)

        if target is not None and self.target_transforms is not None:
            for target_transform in self.target_transforms:
                target = target_transform(target)

        if target is None:
            return {
                self.SEQUENCE_FIELD: sequence,
            }
        else:
            return {
                self.SEQUENCE_FIELD: sequence,
                self.TARGET_FIELD: target,
            }

    def append(
        self,
        sequences: Sequence[str],
        targets: Sequence[float | int | Sequence[float | int]] | None = None,
    ) -> None:
        """Safely append sequences and targets to the dataset with shape and dtype validation.

        Parameters
        ----------
        sequences : Sequence[str]
            New sequences to append. Must be non-empty and contain only strings.
        targets : Sequence[float | int | Sequence[float | int]], optional
            New targets to append. Can be scalars or arrays (for multi-target regression).

        Examples
        --------
        >>> dataset = SequenceDataset(sequences=["ABC", "DEF"], targets=[1.0, 2.0])
        >>> dataset.append(sequences=["GHI"], targets=[3.0])
        >>> len(dataset)
        3

        >>> # Multi-target regression
        >>> dataset = SequenceDataset(sequences=["ABC"], targets=[[1.0, 2.0, 3.0]])
        >>> dataset.append(sequences=["DEF"], targets=[[4.0, 5.0, 6.0]])
        """
        new_sequences = sequences.tolist() if isinstance(sequences, np.ndarray) else list(sequences)
        self._sequences.extend(new_sequences)
        if self._targets is not None and targets is not None:
            new_targets_arr = np.asarray(targets)
            assert len(new_sequences) == len(new_targets_arr), "`sequences` and `targets` must have the same length."
            if new_targets_arr.ndim > 1:
                assert self._targets.ndim > 1, "Cannot mix scalar and array targets."
                assert new_targets_arr.shape[1:] == self._targets.shape[1:], (
                    "`targets` must have the same shape as the existing targets."
                )
            self._targets = np.concatenate([self._targets, new_targets_arr])

    @classmethod
    def from_config(
        cls,
        config: SequenceDatasetConfig,
    ) -> SequenceDataset:
        """Create a SequenceDataset from a configuration.

        Parameters
        ----------
        config : SequenceDatasetConfig
            Configuration for the SequenceDataset.

        Returns
        -------
        SequenceDataset

        """
        from .data_io import load_npz_data
        from .target_transforms import create_target_transform

        data_path = Path(config.data_path)

        data = load_npz_data(
            data_path,
            sequence_key=config.sequence_key,
            target_key=config.target_key,
        )

        sequences = data.get(config.sequence_key)
        targets = data.get(config.target_key) if config.target_key else None

        transforms: list[Callable[[str], str]] | None = None
        if config.transforms:
            # TODO: Implement create_transform when transforms module is populated
            transforms = []

        target_transforms = None
        if config.target_transforms:
            target_transforms = [
                create_target_transform(transform_config) for transform_config in config.target_transforms
            ]

        return cls(
            sequences=sequences,
            targets=targets,
            transforms=transforms,
            target_transforms=target_transforms,
        )
