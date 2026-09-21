"""Transformer primitives: RMSNorm, rotary attention, SwiGLU, pre-norm block.

Standard frontier building blocks (RMSNorm + RoPE + SwiGLU, pre-norm residual).
Attention combines a causal mask with a key-padding mask; because padding is
always a suffix (it follows EOS), position 0 (BOS) is never padded, so no query
row is ever fully masked.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def _rope_cache(seq_len: int, head_dim: int, base: float, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh); cos/sin: (T, Dh)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must divide n_heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.rope_base = cfg.rope_base

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Dh)

        cos, sin = _rope_cache(t, self.head_dim, self.rope_base, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Boolean mask, True = attend. Combine causal (j <= i) with key padding.
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        allowed = causal[None, None, :, :]
        if pad_mask is not None:
            keep_key = (~pad_mask)[:, None, None, :]  # (B, 1, 1, T)
            allowed = allowed & keep_key

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.wout = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wout(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    """Pre-norm transformer block: attention then SwiGLU MLP, both residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = SelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), pad_mask))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x
