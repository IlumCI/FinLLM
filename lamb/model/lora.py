"""LoRA adapters for the shared backbone -- deeper per-environment specialisation.

The shallow adapter in shared-backbone POET only reshapes the final hidden state,
which cannot change the core's reasoning. LoRA adds a low-rank update ``B @ A`` to
the *core's* Linear layers (attention projections, MLP), so an environment's
specialist adapts the computation itself while the base weights stay shared.

Implemented by injection, not rebuild: a forward hook on each targeted Linear adds
the *active* environment's low-rank delta to that Linear's output. Each
environment owns a :class:`LoRASet` (one ``(A, B)`` per targeted Linear);
``set_active`` selects whose deltas apply for the next forward. ``B`` is
zero-initialised, so a fresh adapter starts as the identity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _key(name: str) -> str:
    return name.replace(".", "__")


class LoRASet(nn.Module):
    """One environment's low-rank deltas, keyed by targeted-Linear name."""

    def __init__(self, shapes: Dict[str, Tuple[int, int]], rank: int, alpha: float):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for name, (fin, fout) in shapes.items():
            k = _key(name)
            a = nn.Parameter(torch.empty(rank, fin))
            nn.init.normal_(a, std=1.0 / rank)
            self.A[k] = a
            self.B[k] = nn.Parameter(torch.zeros(fout, rank))  # zero-init => identity

    def delta(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A[key].t()) @ self.B[key].t() * self.scaling


class LoRAInjector:
    """Registers forward hooks on a backbone's targeted Linears; applies the
    currently-active :class:`LoRASet`. Not an ``nn.Module`` -- it holds no
    parameters, only hook handles and a reference to the active set."""

    def __init__(self, backbone: nn.Module, targets: Sequence[str], rank: int, alpha: float):
        self.rank = rank
        self.alpha = alpha
        self.shapes: Dict[str, Tuple[int, int]] = {}
        self.active: Optional[LoRASet] = None
        self._handles = []
        tset = set(targets)
        for name, mod in backbone.named_modules():
            if isinstance(mod, nn.Linear) and name.split(".")[-1] in tset:
                self.shapes[name] = (mod.in_features, mod.out_features)
                self._handles.append(mod.register_forward_hook(self._make_hook(_key(name))))

    def _make_hook(self, key: str):
        def hook(module, inp, out):
            if self.active is not None and key in self.active.A:
                return out + self.active.delta(key, inp[0])
            return out
        return hook

    def new_set(self) -> LoRASet:
        return LoRASet(self.shapes, self.rank, self.alpha)

    def set_active(self, lora: Optional[LoRASet]) -> None:
        self.active = lora

    def param_count(self) -> int:
        return sum(p.numel() for p in self.new_set().parameters())
