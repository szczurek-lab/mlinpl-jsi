"""Configuration for Hyformer backbone model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from joint_improvement.utils.config import BaseConfig


@dataclass
class HyformerConfig(BaseConfig):
    """Configuration for Hyformer transformer model.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (number of tokens).
    d_model : int, default=512
        Model dimension (hidden size). Must be divisible by n_heads.
    n_heads : int, default=8
        Number of attention heads.
    n_layers : int, default=6
        Number of transformer layers.
    max_seq_len : int, default=128
        Maximum sequence length.
    attn_dropout_p : float, default=0.0
        Attention dropout probability.
    rms_norm_eps : float, default=1e-5
        RMSNorm epsilon value.
    predictor_dropout_p : float, default=0.0
        Dropout probability used inside the prediction head (when enabled).
    predictor_head_depth : int, default=1
        Depth of the prediction head after normalization. If 1, uses a single linear layer.
        If >1, uses an MLP stack before the final classifier/regressor.
    predictor_head_act_fn : str, default="gelu"
        Activation function used for the MLP when predictor_head_depth > 1.
        Supported: "gelu", "relu", "silu".
    num_prediction_tasks : int | None, default=None
        Number of prediction task outputs. If None, prediction head is disabled.
    prediction_task_type : str | None, default=None
        Explicit type of the "prediction" task to select the correct loss.
        If None, the loss is inferred from targets dtype/shape.
        Supported values:
        - "regression"
        - "binary_classification"
        - "multilabel_classification"
    """

    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 128
    attn_dropout_p: float = 0.0
    rms_norm_eps: float = 1e-5
    predictor_dropout_p: float = 0.0
    predictor_head_depth: int = 1
    predictor_head_act_fn: str = "gelu"
    num_prediction_tasks: int | None = None
    prediction_task_type: str | None = None
    generator_type: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> HyformerConfig:
        """Load configuration from JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> HyformerConfig:
        """Alias for from_json (keeps interface consistent)."""
        return cls.from_json(path)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if not (0.0 <= self.attn_dropout_p <= 1.0):
            raise ValueError(f"attn_dropout_p must be in [0, 1], got {self.attn_dropout_p}")
        if not (0.0 <= self.rms_norm_eps <= 1.0):
            raise ValueError(f"rms_norm_eps must be in [0, 1], got {self.rms_norm_eps}")
        if not (0.0 <= self.predictor_dropout_p <= 1.0):
            raise ValueError(f"predictor_dropout_p must be in [0, 1], got {self.predictor_dropout_p}")
        if self.predictor_head_depth < 1:
            raise ValueError(f"predictor_head_depth must be >= 1, got {self.predictor_head_depth}")
        allowed_act_fns = {"gelu", "relu", "silu"}
        if self.predictor_head_act_fn.lower().strip() not in allowed_act_fns:
            raise ValueError(
                f"predictor_head_act_fn must be one of {sorted(allowed_act_fns)}, got {self.predictor_head_act_fn!r}"
            )
        if self.num_prediction_tasks is not None and self.num_prediction_tasks <= 0:
            raise ValueError(f"num_prediction_tasks must be positive when set, got {self.num_prediction_tasks}")
        if self.prediction_task_type is not None:
            allowed = {
                "regression",
                "binary_classification",
                "multilabel_classification",
            }
            if self.prediction_task_type not in allowed:
                raise ValueError(
                    f"prediction_task_type must be one of {sorted(allowed)}, got {self.prediction_task_type!r}"
                )
        if self.generator_type is not None and self.generator_type not in (
            "unconditional",
            "gumbeldore",
            "tasar",
        ):
            raise ValueError(
                f"generator_type must be one of 'unconditional', 'gumbeldore', 'tasar', got {self.generator_type!r}"
            )
