"""Prediction head module for Hyformer model (regression and classification tasks)."""

from __future__ import annotations

import torch
from torch import nn


class PredictionHeadModule(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_targets: int,
        predictor_dropout_p: float,
        predictor_head_depth: int,
        predictor_head_act_fn: str,
    ) -> None:
        super().__init__()
        if predictor_head_depth < 1:
            raise ValueError(
                f"predictor_head_depth must be >= 1, got {predictor_head_depth}"
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(predictor_dropout_p)
        self.predictor_head_depth = predictor_head_depth
        self.predictor_head_act_fn = predictor_head_act_fn

        self.mlp: nn.Sequential | None = None
        if predictor_head_depth > 1:
            activation = self._build_activation(predictor_head_act_fn)
            layers: list[nn.Module] = []
            for _ in range(predictor_head_depth - 1):
                layers.append(nn.Linear(d_model, d_model))
                layers.append(activation)
                layers.append(nn.Dropout(predictor_dropout_p))
            self.mlp = nn.Sequential(*layers)

        self.classifier = nn.Linear(d_model, num_targets)

        self._init_weights()

    @staticmethod
    def _build_activation(predictor_head_act_fn: str) -> nn.Module:
        key = predictor_head_act_fn.lower().strip()
        if key == "gelu":
            return nn.GELU()
        if key == "relu":
            return nn.ReLU()
        if key in {"silu", "swish"}:
            return nn.SiLU()
        raise ValueError(
            f"Unsupported predictor_head_act_fn {predictor_head_act_fn!r}. Supported: 'gelu', 'relu', 'silu'."
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        if self.mlp is not None:
            pooled = self.mlp(pooled)
        logits = self.classifier(pooled)
        return logits
