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
        # Long-term-memory inductive bias: default to *retaining* (alpha ~ 0.95)
        # and writing *sparingly* (eta small), so a binding survives across long
        # spans of uninformative tokens until the model learns what to write. The
        # gate stays data-dependent (small weights), it just starts biased.
        nn.init.normal_(self.gate.weight, std=0.01)
        with torch.no_grad():
            self.gate.bias[0] = -1.0  # write rate: softplus(-1) ~ 0.31
            self.gate.bias[1] = 3.0   # retention: sigmoid(3) ~ 0.95

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

    def build_state(self, h: torch.Tensor) -> torch.Tensor:
        """Write the whole sequence into memory and return the final state ``M``.

        Unlike ``forward`` (which returns the per-position causal reads), this
        exposes the accumulated ``(B, d_mem, d_mem)`` associative matrix so it can
        be queried arbitrarily afterwards -- e.g. iterative dereferencing for
        multi-hop retrieval.
        """
        b, t, _ = h.shape
        hn = self.norm(h)
        k = F.normalize(self.to_k(hn), dim=-1)
        v = self.to_v(hn)
        gates = self.gate(hn)
        eta = F.softplus(gates[..., 0])
        alpha = torch.sigmoid(gates[..., 1])
        mem = torch.zeros(b, self.d_mem, self.d_mem, device=h.device, dtype=h.dtype)
        for step in range(t):
            k_t, v_t = k[:, step], v[:, step]
            surprise = v_t - torch.einsum("bd,bde->be", k_t, mem)
            update = torch.einsum("bd,be->bde", k_t, surprise)
            a = alpha[:, step].view(b, 1, 1)
            e = eta[:, step].view(b, 1, 1)
            mem = a * mem + e * update
        return mem

    def read_state(self, mem: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Query a built state ``mem`` with a hidden vector, returning a read
        projected back to model width. ``hidden`` is ``(B, d_model)``."""
        q = F.normalize(self.to_q(self.norm(hidden)), dim=-1)
        return self.out(torch.einsum("bd,bde->be", q, mem))
