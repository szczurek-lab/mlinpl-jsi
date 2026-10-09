"""LLaMA-style SwiGLU FFN."""

import torch
import torch.nn.functional as F
from torch import nn

MLP_RATIO = 4
MULTIPLE_OF = 256


class SwiGLUFFN(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden_dim = int((2 / 3) * MLP_RATIO * d_model)
        hidden_dim = (hidden_dim + MULTIPLE_OF - 1) // MULTIPLE_OF * MULTIPLE_OF

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
