"""Test-time neural memory -- the ``infContext`` mechanism.

A fixed-size associative memory ``M`` (a ``d_mem x d_mem`` matrix) that is
written *during the forward pass* by a surprise-gated delta rule, following the
Titans / ATLAS "learn to memorise at test time" line and its DeltaNet-style
linearisation. Reading and writing are O(1) per token and the state size is
independent of sequence length, so effective context is unbounded.

For each token t (causally, reading before writing):
    read_t    = q_t @ M                       # associative recall
    surprise  = v_t - (k_t @ M)               # prediction error under M
    M         = alpha_t * M + eta_t * (k_t outer surprise)

``alpha_t`` (retention/forget) and ``eta_t`` (write rate) are data-dependent
gates produced from the hidden state -- the key idea that lets the memory decide
what to keep. Gradients flow through the whole recurrence, so the projections
learn *how* to use the memory. Padding is a suffix, so padded writes never
affect earlier reads and need no masking.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm


class NeuralMemory(nn.Module):
    def __init__(self, d_model: int, d_mem: int):
        super().__init__()
        self.d_mem = d_mem
        self.norm = RMSNorm(d_model)
        self.to_q = nn.Linear(d_model, d_mem, bias=False)
        self.to_k = nn.Linear(d_model, d_mem, bias=False)
        self.to_v = nn.Linear(d_model, d_mem, bias=False)
        self.gate = nn.Linear(d_model, 2)  # -> (write-rate logit, retention logit)
        self.out = nn.Linear(d_mem, d_model, bias=False)

    def forward(self, h: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = h.shape
        hn = self.norm(h)
        q = F.normalize(self.to_q(hn), dim=-1)
        k = F.normalize(self.to_k(hn), dim=-1)
        v = self.to_v(hn)
        gates = self.gate(hn)
        eta = F.softplus(gates[..., 0])          # (B, T) write rate >= 0
        alpha = torch.sigmoid(gates[..., 1])     # (B, T) retention in (0, 1)

        mem = torch.zeros(b, self.d_mem, self.d_mem, device=h.device, dtype=h.dtype)
        reads = []
        for step in range(t):
            q_t, k_t, v_t = q[:, step], k[:, step], v[:, step]  # (B, d_mem)
            reads.append(torch.einsum("bd,bde->be", q_t, mem))
            pred = torch.einsum("bd,bde->be", k_t, mem)
            surprise = v_t - pred
            update = torch.einsum("bd,be->bde", k_t, surprise)
            a = alpha[:, step].view(b, 1, 1)
            e = eta[:, step].view(b, 1, 1)
            mem = a * mem + e * update

        read = torch.stack(reads, dim=1)  # (B, T, d_mem)
        return self.out(read)
