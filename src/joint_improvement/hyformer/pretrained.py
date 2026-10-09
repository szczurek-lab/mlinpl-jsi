"""Pretrained model loading and saving for backbones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from .config import HyformerConfig

try:
    from huggingface_hub import hf_hub_download

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    hf_hub_download = None


class PretrainedMixin:
    """Mixin for pretrained model loading and saving."""

    MODEL_CONFIG_FILE_NAME = "model_config.json"
    CHECKPOINT_FILE_NAME = "model.pt"

    @staticmethod
    def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remove compile prefixes."""
        return {k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}

    @staticmethod
    def _get_underlying_model(model: torch.nn.Module) -> torch.nn.Module:
        """Get underlying model, handling wrapped models (e.g., compiled)."""
        underlying_model = model
        if hasattr(model, "module"):
            underlying_model = model.module
        if hasattr(underlying_model, "_orig_mod"):
            underlying_model = underlying_model._orig_mod
        return underlying_model

    def get_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Get cleaned model state dict, handling compiled models."""
        underlying_model = self._get_underlying_model(self)
        state_dict = underlying_model.state_dict()
        return self._clean_state_dict(state_dict)

    def save_pretrained(self, path: str | Path) -> None:
        """Save model weights.

        Parameters
        ----------
        path : str | Path
            Path to save weights to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.get_model_state_dict()
        torch.save(state_dict, path)
        logger.info("Model weights saved to {}", path)

    def load_state_dict_from_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> None:
        """Load model weights from a state dict.

        Parameters
        ----------
        state_dict : dict[str, torch.Tensor]
            State dict to load weights from.
        strict : bool, default=False
            If True, raise error on missing/unexpected keys. If False, log warnings.
        """
        cleaned_state_dict = self._clean_state_dict(state_dict)
        underlying_model = self._get_underlying_model(self)
        missing_keys, unexpected_keys = underlying_model.load_state_dict(cleaned_state_dict, strict=False)

        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys in state_dict: {unexpected_keys}")
        else:
            if missing_keys:
                logger.warning("Missing keys (not loaded): {}", missing_keys)
            if unexpected_keys:
                logger.warning("Unexpected keys (ignored): {}", unexpected_keys)

    def load_pretrained(self, path: str | Path, device: torch.device | str | None = None, strict: bool = False) -> None:
        """Load model weights.

        Parameters
        ----------
        path : str | Path
            Path to load weights from.
        device : torch.device | str | None, default=None
            Device to load on.
        strict : bool, default=False
            If True, raise error on missing/unexpected keys. If False, log warnings.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state_dict = torch.load(path, map_location=device)
        self.load_state_dict_from_dict(state_dict, strict=strict)

        if device:
            # Type check: classes using this mixin should inherit from nn.Module
            if hasattr(self, "to"):
                self.to(device)

        logger.info("Loaded weights from {}", path)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path,
        config: HyformerConfig | None = None,
        device: torch.device | str | None = None,
        strict: bool = False,
    ) -> torch.nn.Module:
        """Load model from local path or HuggingFace Hub.

        Parameters
        ----------
        model_id : str | Path
            Local path to model directory/file or HuggingFace model identifier.
        config : HyformerConfig | None, default=None
            Model configuration. If None, loads from config.json.
        device : torch.device | str | None, default=None
            Device to load on.
        strict : bool, default=False
            If True, raise error on missing/unexpected keys. If False, log warnings.

        Returns
        -------
        torch.nn.Module
            Loaded model instance.
        """
        model_id_str = str(model_id)
        path = Path(model_id_str)

        # Determine if local path or HuggingFace Hub
        # Check if path exists and is a local file/directory
        is_local = path.exists()

        if is_local:
            # Local path loading
            if path.is_dir():
                config_path = path / cls.MODEL_CONFIG_FILE_NAME
                checkpoint_path = path / cls.CHECKPOINT_FILE_NAME
            else:
                # Single file - assume it's a checkpoint, look for config in parent
                checkpoint_path = path
                config_path = path.parent / cls.MODEL_CONFIG_FILE_NAME
        else:
            # HuggingFace Hub loading
            if not _HF_AVAILABLE:
                raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")

            # Try to download config
            config_path = None
            try:
                config_path_str = hf_hub_download(model_id_str, cls.MODEL_CONFIG_FILE_NAME, cache_dir=None)
                config_path = Path(config_path_str)
            except Exception as e:
                logger.debug("Could not download {}: {}", cls.MODEL_CONFIG_FILE_NAME, e)
                config_path = None

            # Try to download checkpoint
            try:
                checkpoint_path_str = hf_hub_download(model_id_str, cls.CHECKPOINT_FILE_NAME, cache_dir=None)
                checkpoint_path = Path(checkpoint_path_str)
            except Exception as e:
                raise FileNotFoundError(
                    f"No checkpoint found in {model_id_str}. Expected {cls.CHECKPOINT_FILE_NAME}: {e}"
                ) from e

        # Load config
        if config is None:
            if config_path is None or not config_path.exists():
                raise ValueError(f"config required. Could not load {cls.MODEL_CONFIG_FILE_NAME}.")
            with open(config_path) as f:
                from .config import HyformerConfig

                config = HyformerConfig(**json.load(f))

        # Load model
        model = cls.from_config(config) if hasattr(cls, "from_config") else cls(**config.to_dict())
        model.load_pretrained(checkpoint_path, device=device, strict=strict)
        return model
