"""Trainer checkpoint dataclass and mixin."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger


@dataclass
class TrainerCheckpoint:
    """Trainer checkpoint containing optimizer and trainer state."""

    optimizer_state: dict[str, Any]
    trainer_state: dict[str, Any]
    model_state_dict: dict[str, Any] | None = None


class TrainerCheckpointMixin:
    """Mixin providing core checkpoint save/load functionality for trainers.

    Assumes the trainer has:
    - self.model: Hyformer (inherits from PretrainedMixin)
    - self.optimizer: torch.optim.Optimizer
    - self.state: TrainerState (or compatible dataclass)
    - self.device: torch.device
    """

    BEST_CHECKPOINT_FILE_NAME = "checkpoint.pt"
    BEST_MODEL_CHECKPOINT_FILE_NAME = "model.pt"
    CHECKPOINT_EPOCH_FILE_PATTERN = "checkpoint_epoch_{epoch}.pt"
    MODEL_EPOCH_FILE_PATTERN = "model_epoch_{epoch}.pt"

    def _build_checkpoint(self, include_model: bool = False) -> TrainerCheckpoint:
        """Build checkpoint object with optimizer and trainer state.

        Parameters
        ----------
        include_model : bool, default=False
            If True, includes model state dict in checkpoint.

        Returns
        -------
        TrainerCheckpoint
            Checkpoint object.
        """
        return TrainerCheckpoint(
            optimizer_state=self.optimizer.state_dict(),  # type: ignore[attr-defined]
            trainer_state=asdict(self.state),  # type: ignore[attr-defined]
            model_state_dict=self.model.get_model_state_dict() if include_model else None,  # type: ignore[attr-defined]
        )

    def _verify_checkpoint(self, path: Path) -> None:
        """Verify checkpoint can be loaded and has required fields.

        Parameters
        ----------
        path : Path
            Path to checkpoint file to verify.

        Raises
        ------
        RuntimeError
            If checkpoint verification fails.
        """
        try:
            data = torch.load(path, map_location="cpu")
            required_fields = ["optimizer_state", "trainer_state"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise RuntimeError(f"Checkpoint missing required fields: {missing_fields}")
        except Exception as e:
            raise RuntimeError(f"Checkpoint verification failed: {e}") from e

    def save_trainer_checkpoint(
        self,
        path: str | Path,
        include_model: bool = False,
        verify: bool = True,
    ) -> None:
        """Save trainer checkpoint with atomic writes and optional verification.

        Parameters
        ----------
        path : str | Path
            Path to save checkpoint to.
        include_model : bool, default=False
            If True, includes model state dict in checkpoint. If False, only saves
            optimizer and trainer state.
        verify : bool, default=True
            If True, verifies checkpoint can be loaded after saving.
        """
        path = Path(path)
        temp_path = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            checkpoint = self._build_checkpoint(include_model=include_model)
            checkpoint_dict = asdict(checkpoint)

            # Save to temporary file first
            torch.save(checkpoint_dict, temp_path, _use_new_zipfile_serialization=True)

            # Verify checkpoint if requested
            if verify:
                self._verify_checkpoint(temp_path)

            # Atomic rename
            temp_path.replace(path)
            logger.info(f"Checkpoint saved: {path}")
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def save_model_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint only.

        Uses PretrainedMixin.save_pretrained() for consistency.

        Parameters
        ----------
        path : str | Path
            Path to save model checkpoint to.
        """
        self.model.save_pretrained(path)  # type: ignore[attr-defined]

    def _load_checkpoint(
        self,
        path: str | Path,
        resume_training: bool = True,
        strict: bool = False,
        map_location: str | torch.device | None = None,
    ) -> None:
        """Load trainer checkpoint and optionally resume training.

        Parameters
        ----------
        path : str | Path
            Path to checkpoint file.
        resume_training : bool, default=True
            If True, loads optimizer state and trainer state. If False, only loads model weights.
        strict : bool, default=False
            If True, raise error on missing/unexpected keys in optimizer/model state.
            If False, log warnings.
        map_location : str | torch.device | None, default=None
            Device to load checkpoint on. If None, uses self.device.

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.
        RuntimeError
            If checkpoint loading fails.
        """
        from .multitask import TrainerState

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = map_location or self.device  # type: ignore[attr-defined]

        try:
            data = torch.load(path, map_location=map_location)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file {path}: {e}") from e

        # Build checkpoint object (handles missing fields gracefully for backward compatibility)
        checkpoint = TrainerCheckpoint(
            optimizer_state=data.get("optimizer_state", {}),
            trainer_state=data.get("trainer_state", {}),
            model_state_dict=data.get("model_state_dict"),
        )

        # Load model state dict if present in checkpoint
        # Use PretrainedMixin method for consistency
        if checkpoint.model_state_dict is not None:
            self.model.load_state_dict_from_dict(checkpoint.model_state_dict, strict=strict)  # type: ignore[attr-defined]

        # Load optimizer and trainer state if resuming training
        if resume_training:
            try:
                self.optimizer.load_state_dict(checkpoint.optimizer_state)  # type: ignore[attr-defined]
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to load optimizer state: {e}") from e
                logger.warning(f"Failed to load optimizer state: {e}. Continuing without optimizer state.")

            try:
                self.state = TrainerState(**checkpoint.trainer_state)
            except Exception as e:
                raise RuntimeError(f"Failed to load trainer state: {e}") from e

        logger.info(f"Loaded checkpoint: {path}")
