"""Number-native input embedding.

The input representation sums three channels:

* **Token embedding** -- over the small digit/operator vocabulary.
* **Abacus embedding** -- the digit's position *within its number* (index 0 is
  the shared "not a digit" row); this is the ingredient that lets arithmetic
  generalise to unseen operand lengths.
* **Value channel** -- multi-scale sinusoidal features of the operand's
  magnitude plus a signed-log term, so a digit token also "knows" the value of
  the number it belongs to. Gated by ``value_mask`` and, by construction of the
  tokenizer, present only on problem operands -- never on the answer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .transformer import RMSNorm


class NumberAwareEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.abacus_emb = nn.Embedding(cfg.max_number_len + 1, cfg.d_model)
        self.value_freqs = cfg.value_freqs

        # Log-spaced angular frequencies spanning unit to ~1e-4, so sin/cos of
        # value resolve magnitude across the operand range used in training.
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, cfg.value_freqs).float() / cfg.value_freqs))
        self.register_buffer("value_inv_freq", inv_freq, persistent=False)

        self.value_proj = nn.Linear(1 + 2 * cfg.value_freqs, cfg.d_model)
        self.norm = RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def _value_features(self, value: torch.Tensor) -> torch.Tensor:
        # value: (B, T) float -> (B, T, 1 + 2F)
        slog = torch.sign(value) * torch.log1p(value.abs())
        angles = value.unsqueeze(-1) * self.value_inv_freq  # (B, T, F)
        return torch.cat([slog.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        abacus_ids: torch.Tensor,
        value: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> torch.Tensor:
        tok = self.token_emb(input_ids)
        ab = self.abacus_emb(abacus_ids)
        val = self.value_proj(self._value_features(value)) * value_mask.unsqueeze(-1)
        return self.dropout(self.norm(tok + ab + val))
