"""Transformer layer."""

from __future__ import annotations

import torch
from torch import nn

from .attention import HybridSelfAttention
from .ffn import SwiGLUFFN


class TransformerLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        attn_dropout_p: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()

        self.attn_norm = nn.RMSNorm(d_model, eps=rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(d_model, eps=rms_norm_eps)

        self.attn = HybridSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout_p=attn_dropout_p,
            max_seq_len=max_seq_len,
        )
        self.ffn = SwiGLUFFN(d_model=d_model)

    def forward(
        self, x: torch.Tensor, is_causal: bool, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        h = x + self.attn(self.attn_norm(x), attn_mask=attn_mask, is_causal=is_causal)
        output = h + self.ffn(self.ffn_norm(h))
        return output
