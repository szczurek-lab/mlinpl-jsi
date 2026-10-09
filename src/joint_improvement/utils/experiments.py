"""Utilities for experiment scripts (training, evaluation, etc.)."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from pathlib import Path

    from joint_improvement.utils import SequenceDataLoader, SequenceDataset


def load_datasets(
    train_config_path: Path,
    val_config_path: Path | None = None,
) -> tuple[SequenceDataset, SequenceDataset | None]:
    """Load training and validation datasets.

    Parameters
    ----------
    train_config_path : Path
        Path to training dataset config.
    val_config_path : Path, optional
        Path to validation dataset config.

    Returns
    -------
    tuple[SequenceDataset, Optional[SequenceDataset]]
        Training and validation datasets.

    Examples
    --------
    >>> from pathlib import Path
    >>> train_dataset, val_dataset = load_datasets(
    ...     train_config_path=Path("configs/datasets/train.json"),
    ...     val_config_path=Path("configs/datasets/val.json"),
    ... )
    """
    from joint_improvement.utils import SequenceDataset, SequenceDatasetConfig

    logger.info(f"Loading training dataset from: {train_config_path}")
    train_config = SequenceDatasetConfig.from_pretrained(train_config_path)
    train_dataset = SequenceDataset.from_config(train_config)
    logger.info(f"Training dataset size: {len(train_dataset):,}")

    val_dataset = None
    if val_config_path is not None:
        logger.info(f"Loading validation dataset from: {val_config_path}")
        val_config = SequenceDatasetConfig.from_pretrained(val_config_path)
        val_dataset = SequenceDataset.from_config(val_config)
        logger.info(f"Validation dataset size: {len(val_dataset):,}")

    return train_dataset, val_dataset


def create_task_dataloaders(
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset | None,
    tasks: dict[str, float],
    collator_factory: Callable[[str], Any],
    batch_size: int,
    num_workers: int,
    device: str | torch.device,
) -> tuple[dict[str, SequenceDataLoader], dict[str, SequenceDataLoader] | None]:
    """Create training and validation dataloaders for each task.

    Parameters
    ----------
    train_dataset : SequenceDataset
        Training dataset.
    val_dataset : SequenceDataset, optional
        Validation dataset.
    tasks : dict[str, float]
        Dictionary mapping task names to their weights.
    collator_factory : Callable[[str], Any]
        Function that takes a task name and returns a collator instance.
    batch_size : int
        Batch size.
    num_workers : int
        Number of data loading workers.
    device : str | torch.device
        Device to train on (for pin_memory).

    Returns
    -------
    tuple[dict[str, SequenceDataLoader], dict[str, SequenceDataLoader] | None]
        Training loaders keyed by task name, and optionally validation loaders.

    Examples
    --------
    >>> from joint_improvement.utils import SequenceDataset
    >>> def make_collator(task: str):
    ...     if task == "lm":
    ...         return LMCollator(tokenizer, max_length=512)
    ...     return MLMCollator(tokenizer, max_length=512)
    >>> train_loaders, val_loaders = create_task_dataloaders(
    ...     train_dataset=train_ds,
    ...     val_dataset=val_ds,
    ...     tasks={"lm": 1.0, "mlm": 0.5},
    ...     collator_factory=make_collator,
    ...     batch_size=32,
    ...     num_workers=4,
    ...     device="cuda",
    ... )
    """
    from joint_improvement.utils import SequenceDataLoader

    device_obj = torch.device(device) if isinstance(device, str) else device
    pin_memory = device_obj.type == "cuda"

    train_loaders: dict[str, SequenceDataLoader] = {}
    val_loaders: dict[str, SequenceDataLoader] | None = None

    for task_name in tasks.keys():
        # Create collator for this task
        collator = collator_factory(task_name)

        # Create training loader for this task
        train_loader = SequenceDataLoader(
            dataset=train_dataset,
            collator=collator,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        train_loaders[task_name] = train_loader

        # Create validation loader for this task if validation dataset exists
        if val_dataset is not None:
            if val_loaders is None:
                val_loaders = {}
            val_loader = SequenceDataLoader(
                dataset=val_dataset,
                collator=collator,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loaders[task_name] = val_loader

    return train_loaders, val_loaders


def save_configs(
    output_dir: Path,
    configs: dict[str, Any],
    tokenizer_config_path: Path | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    """Save all configurations to output directory for reproducibility.

    Parameters
    ----------
    output_dir : Path
        Output directory to save configs to.
    configs : dict[str, Any]
        Dictionary mapping config names to config objects. Config objects should
        have a `to_json()` method or be dataclasses that can be converted with `asdict()`.
    tokenizer_config_path : Path, optional
        Path to tokenizer config directory (will be copied).
    args : argparse.Namespace, optional
        Command-line arguments to save.

    Examples
    --------
    >>> from pathlib import Path
    >>> save_configs(
    ...     output_dir=Path("results/experiment"),
    ...     configs={
    ...         "model_config": model_config,
    ...         "trainer_config": trainer_config,
    ...         "train_dataset_config": train_dataset_config,
    ...     },
    ...     tokenizer_config_path=Path("configs/tokenizers/smiles"),
    ...     args=args,
    ... )
    """
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving all configurations to: {configs_dir}")

    # Save each config
    for config_name, config_obj in configs.items():
        config_path = configs_dir / f"{config_name}.json"

        # Try to_json() method first (for config classes)
        if hasattr(config_obj, "to_json"):
            config_obj.to_json(config_path)
            logger.info(f"  ✓ Saved {config_name}: {config_path}")
        # Fall back to dataclass asdict
        elif hasattr(config_obj, "__dataclass_fields__"):
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(config_obj), f, indent=2, default=str)
            logger.info(f"  ✓ Saved {config_name}: {config_path}")
        else:
            logger.warning(f"  ⚠ Skipped {config_name}: no to_json() method or dataclass fields")

    # Copy tokenizer config if provided
    if tokenizer_config_path is not None:
        tokenizer_configs_dir = configs_dir / "tokenizer"
        tokenizer_configs_dir.mkdir(parents=True, exist_ok=True)

        # Copy tokenizer_config.json
        tokenizer_config_file = tokenizer_config_path / "tokenizer_config.json"
        if not tokenizer_config_file.exists() and tokenizer_config_path.is_file():
            tokenizer_config_file = tokenizer_config_path
            tokenizer_configs_dir = configs_dir

        if tokenizer_config_file.exists():
            shutil.copy2(tokenizer_config_file, tokenizer_configs_dir / "tokenizer_config.json")
            logger.info(f"  ✓ Copied tokenizer config: {tokenizer_configs_dir / 'tokenizer_config.json'}")

        # Copy vocab files if they exist in tokenizer directory
        if tokenizer_config_path.is_dir():
            vocab_files = list(tokenizer_config_path.glob("vocab*.txt")) + list(
                tokenizer_config_path.glob("vocab*.json")
            )
            for vocab_file in vocab_files:
                shutil.copy2(vocab_file, tokenizer_configs_dir / vocab_file.name)
                logger.info(f"  ✓ Copied vocab file: {vocab_file.name}")

    # Save command-line arguments if provided
    if args is not None:
        args_dict = vars(args)
        args_path = configs_dir / "command_line_args.json"
        with args_path.open("w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2, default=str)
        logger.info(f"  ✓ Saved command-line arguments: {args_path}")

    logger.info("All configurations saved successfully.")


def validate_paths(
    required_paths: dict[str, Path],
    optional_paths: dict[str, Path] | None = None,
) -> None:
    """Validate that all required paths exist.

    Parameters
    ----------
    required_paths : dict[str, Path]
        Dictionary mapping path names to Path objects that must exist.
    optional_paths : dict[str, Path], optional
        Dictionary mapping path names to Path objects that should exist if provided.

    Raises
    ------
    FileNotFoundError
        If any required path does not exist.

    Examples
    --------
    >>> from pathlib import Path
    >>> validate_paths(
    ...     required_paths={
    ...         "model_config": Path("configs/model.json"),
    ...         "train_dataset": Path("data/train.json"),
    ...     },
    ...     optional_paths={
    ...         "val_dataset": Path("data/val.json"),
    ...     },
    ... )
    """
    all_paths = required_paths.copy()
    if optional_paths:
        # Only validate optional paths that are not None
        all_paths.update({k: v for k, v in optional_paths.items() if v is not None})

    for name, path in all_paths.items():
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
