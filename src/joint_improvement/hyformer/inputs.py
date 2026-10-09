"""Input container for backbone models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
else:
    import torch


@dataclass
class ModelInput:
    """Input container for backbone models.

    Supports dictionary unpacking for compatibility with model forward calls:
        model(**model_input)  # Works!
    """

    input_ids: torch.Tensor
    task: str
    attention_mask: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    targets: torch.Tensor | None = None

    def keys(self) -> list[str]:
        """Return keys for dictionary unpacking compatibility."""
        keys = ["input_ids", "task"]
        if self.attention_mask is not None:
            keys.append("attention_mask")
        if self.labels is not None:
            keys.append("labels")
        if self.targets is not None:
            keys.append("targets")
        return keys

    def __getitem__(self, key: str) -> torch.Tensor | str | None:
        """Support dictionary-style access for unpacking."""
        if key == "input_ids":
            return self.input_ids
        elif key == "task":
            return self.task
        elif key == "attention_mask":
            return self.attention_mask
        elif key == "labels":
            return self.labels
        elif key == "targets":
            return self.targets
        else:
            raise KeyError(f"ModelInput has no key '{key}'")

    def __iter__(self):
        """Support iteration for dictionary unpacking."""
        return iter(self.keys())

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dictionary compatibility."""
        return key in ["input_ids", "task", "attention_mask", "labels", "targets"]

    def to_dict(self) -> dict[str, torch.Tensor | str | None]:
        """Convert to dictionary for explicit conversion if needed."""
        result: dict[str, torch.Tensor | str | None] = {
            "input_ids": self.input_ids,
            "task": self.task,
        }
        if self.attention_mask is not None:
            result["attention_mask"] = self.attention_mask
        if self.labels is not None:
            result["labels"] = self.labels
        if self.targets is not None:
            result["targets"] = self.targets
        return result

    def to(self, device: torch.device | str) -> ModelInput:
        """Move all tensors to the specified device.

        Parameters
        ----------
        device : torch.device | str
            Target device (e.g., "cuda", "cpu", torch.device("cuda:0")).

        Returns
        -------
        ModelInput
            New ModelInput instance with all tensors moved to the device.
            Non-tensor fields (like `task`) remain unchanged.

        Examples
        --------
        >>> model_input = ModelInput(input_ids=..., task="prediction", attention_mask=...)
        >>> model_input_gpu = model_input.to("cuda")
        >>> model_input_gpu = model_input.to(torch.device("cuda:0"))
        """
        # Keep this method minimal: device transfer only.
        # Contiguity/layout constraints (e.g., for torch.compile) should be handled
        # closer to where tensors are consumed (loss/model), not at transport time.
        return ModelInput(
            input_ids=self.input_ids.to(device),
            task=self.task,
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            labels=self.labels.to(device) if self.labels is not None else None,
            targets=self.targets.to(device) if self.targets is not None else None,
        )

    def __reduce_ex__(self, protocol):
        """Custom reducer for pickling that works with module reloading."""
        return (
            _reconstruct_modelinput,
            (
                self.input_ids,
                self.task,
                self.attention_mask,
                self.labels,
                self.targets,
            ),
        )


def _reconstruct_modelinput(input_ids, task, attention_mask, labels, targets):
    """Reconstruct ModelInput from pickled state.

    This function is used instead of the class directly to avoid issues
    with module reloading (e.g., autoreload in Jupyter notebooks).
    """
    # Import here to get the current version of the class
    from joint_improvement.hyformer.inputs import ModelInput

    return ModelInput(
        input_ids=input_ids,
        task=task,
        attention_mask=attention_mask,
        labels=labels,
        targets=targets,
    )


__all__ = ["ModelInput"]
