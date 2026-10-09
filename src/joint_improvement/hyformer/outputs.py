"""Output containers for backbone models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass
class ModelOutput:
    """Base output for backbone models."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    extras: dict[str, Any] | None = None

    def __getitem__(self, key: str) -> Any:
        """Provide dict-style access for backward compatibility."""
        if key == "logits":
            return self.logits
        if key == "loss":
            return self.loss
        if key == "extras":
            return self.extras
        raise KeyError(f"ModelOutput has no key '{key}'")


__all__ = ["ModelOutput"]
