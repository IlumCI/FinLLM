"""Depth-recurrent latent reasoning core -- where LAMb "thinks".

Structure (following the recurrent-depth / Universal-Transformer line, and the
continuous-thought idea of feeding a hidden state back as the next input):

    prelude blocks  ->  [ recurrent block x T ]  ->  coda blocks

The recurrent block is applied ``T`` times to the *same* latent state, with the
embedded input optionally re-injected each step. All reasoning happens in
continuous latent space -- no intermediate tokens are decoded -- and ``T`` is a
knob you can turn *up at test time* to spend more compute on harder problems.

With ``adaptive_halting`` the number of steps becomes per-position and learned
(ACT-style ponder): each position accumulates a halting probability and stops
early, and a ponder cost discourages over-thinking. This is off by default.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig
from .memory import NeuralMemory
from .transformer import Block


class RecurrentDepthCore(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.prelude = nn.ModuleList([Block(cfg) for _ in range(cfg.n_prelude)])
        self.recurrent = nn.ModuleList([Block(cfg) for _ in range(cfg.n_recurrent)])
        self.coda = nn.ModuleList([Block(cfg) for _ in range(cfg.n_coda)])
        self.memory = NeuralMemory(cfg.d_model, cfg.d_mem) if cfg.use_memory else None
        self.halt = nn.Linear(cfg.d_model, 1) if cfg.adaptive_halting else None

    def _recur_step(self, state: torch.Tensor, x: torch.Tensor, pad_mask) -> torch.Tensor:
        inp = state + x if self.cfg.inject_input else state
        for blk in self.recurrent:
            inp = blk(inp, pad_mask)
        if self.memory is not None:
            inp = inp + self.memory(inp, pad_mask)
        return inp

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = x
        for blk in self.prelude:
            h = blk(h, pad_mask)

        if self.halt is not None:
            out, aux = self._adaptive(h, x, pad_mask)
        else:
            steps = n_steps if n_steps is not None else self.cfg.recurrent_steps
            state = h
            for _ in range(max(1, steps)):
                state = self._recur_step(state, x, pad_mask)
            out, aux = state, {}

        for blk in self.coda:
            out = blk(out, pad_mask)
        return out, aux

    def _adaptive(self, h: torch.Tensor, x: torch.Tensor, pad_mask):
        # Graves-style Adaptive Computation Time over the recurrent block.
        b, t, _ = h.shape
        device = h.device
        threshold = 1.0 - 1e-2
        still = torch.ones(b, t, device=device)
        accum = torch.zeros(b, t, device=device)
        n_updates = torch.zeros(b, t, device=device)
        weighted = torch.zeros_like(h)
        state = h
        remainder = torch.ones(b, t, device=device)

        max_steps = max(1, self.cfg.max_recurrent_steps)
        for step in range(max_steps):
            state = self._recur_step(state, x, pad_mask)
            p = torch.sigmoid(self.halt(state)).squeeze(-1)  # (B, T)
            is_last = step == max_steps - 1
            would_exceed = (accum + still * p) > threshold
            halt_now = would_exceed | torch.full_like(would_exceed, is_last)
            remainder = 1.0 - accum
            weight = torch.where(halt_now, remainder, still * p)
            weighted = weighted + weight.unsqueeze(-1) * state
            n_updates = n_updates + still
            accum = accum + still * p
            still = still * (~halt_now).float()
            if float(still.sum()) == 0.0:
                break

        ponder = (n_updates + remainder).mean()
        return weighted, {"ponder": ponder}
