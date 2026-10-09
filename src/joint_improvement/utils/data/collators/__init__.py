"""Task-specific collators for sequence datasets."""

from __future__ import annotations

from .base import BaseCollatorWithPadding
from .lm_collator import LMCollator
from .mlm_collator import MLMCollator
from .prediction_collator import PredictionCollator

# Backward compatibility alias
BaseCollator = BaseCollatorWithPadding

__all__ = [
    "BaseCollator",
    "BaseCollatorWithPadding",
    "LMCollator",
    "MLMCollator",
    "PredictionCollator",
]
