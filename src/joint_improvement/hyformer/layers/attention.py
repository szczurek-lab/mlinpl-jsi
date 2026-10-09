"""Self-attention module with rotary positional embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .rotary import RotaryPositionalEmbedding


class HybridSelfAttention(nn.Module):

    def __init__(
        self, d_model: int, n_heads: int, attn_dropout_p: float, max_seq_len: int
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout_p = attn_dropout_p
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.relative_embedding = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:

        B, T, D = x.shape

        q, k, v = self.qkv(x).split(D, dim=2)  # 3 * (B, T, D)

        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        q = self.relative_embedding(q)
        k = self.relative_embedding(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if is_causal:
            attn_mask = None
        else:
            attn_mask = attn_mask.bool().unsqueeze(1).unsqueeze(1).expand(B, self.n_heads, T, T)  # (B, n_heads, T, T)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )  # (B, n_heads, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        y = self.out(y)

        return y
