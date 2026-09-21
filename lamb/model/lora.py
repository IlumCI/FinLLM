"""LoRA / DoRA adapters for the shared backbone -- deeper per-environment specialisation.

The shallow adapter only reshapes the final hidden state, which cannot change the
core's reasoning. These adapters add a low-rank update to the *core's* Linear
layers (attention projections, MLP), so an environment's specialist adapts the
computation itself while the base weights stay shared.

* **LoRA**: ``y = x(W + s·BA)^T`` -- an additive low-rank delta on the weight.
* **DoRA** (weight-decomposed, arXiv:2402.09353): decompose the weight into a
  magnitude ``m`` and a direction, apply the low-rank update to the *direction*,
  and train ``m`` separately:
      ``W' = m ⊙ (W + s·BA) / ‖W + s·BA‖_col``.
  Decoupling magnitude from direction recovers a full-fine-tuning-like update and
  beats LoRA at equal rank, which matters most at the low ranks used here. It
  costs one magnitude vector per layer on top of LoRA.

Implemented by injection, not rebuild: a forward hook on each targeted Linear adds
(LoRA) or reparametrises (DoRA) that Linear's output using the *active*
environment's :class:`LoRASet`. ``B`` is zero-initialised and, for DoRA, ``m`` is
initialised to the base column norms, so a fresh adapter is exactly the identity.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _key(name: str) -> str:
    return name.replace(".", "__")


class LoRASet(nn.Module):
    """One environment's low-rank deltas (and DoRA magnitudes), keyed by Linear name."""

    def __init__(self, shapes: Dict[str, Tuple[int, int]], rank: int, alpha: float,
                 dora: bool = False, base_norms: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.scaling = alpha / rank
        self.dora = dora
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self.M = nn.ParameterDict() if dora else None
        for name, (fin, fout) in shapes.items():
            k = _key(name)
            a = nn.Parameter(torch.empty(rank, fin))
            nn.init.normal_(a, std=1.0 / rank)
            self.A[k] = a
            self.B[k] = nn.Parameter(torch.zeros(fout, rank))  # zero-init => identity
            if dora:
                self.M[k] = nn.Parameter(base_norms[k].detach().clone())  # init to ||W||_col


class LoRAInjector:
    """Registers forward hooks on a backbone's targeted Linears and applies the
    currently-active :class:`LoRASet` (LoRA add, or DoRA reparametrisation)."""

    def __init__(self, backbone: nn.Module, targets: Sequence[str], rank: int, alpha: float,
                 dora: bool = False):
        self.rank = rank
        self.alpha = alpha
        self.dora = dora
        self.shapes: Dict[str, Tuple[int, int]] = {}
        self.mods: Dict[str, nn.Module] = {}
        self.active: Optional[LoRASet] = None
        self._handles = []
        tset = set(targets)
        for name, mod in backbone.named_modules():
            if isinstance(mod, nn.Linear) and name.split(".")[-1] in tset:
                k = _key(name)
                self.shapes[name] = (mod.in_features, mod.out_features)
                self.mods[k] = mod
                self._handles.append(mod.register_forward_hook(self._make_hook(k, mod)))

    def _make_hook(self, key: str, module: nn.Module):
        def hook(_module, inp, out):
            act = self.active
            if act is None or key not in act.A:
                return out
            dv = (act.B[key] @ act.A[key]) * act.scaling      # (out, in) directional update
            delta = inp[0] @ dv.t()                            # x @ dv^T
            if not act.dora:
                return out + delta                             # LoRA: additive
            # DoRA: y = (m / ||W + dv||_col) * (x @ (W + dv)^T); norm detached (paper trick).
            wv = module.weight + dv
            norm = wv.norm(dim=1).detach().clamp_min(1e-6)     # (out,)
            return (act.M[key] / norm) * (out + delta)
        return hook

    def new_set(self) -> LoRASet:
        base_norms = None
        if self.dora:
            base_norms = {k: self.mods[k].weight.detach().norm(dim=1) for k in self.mods}
        return LoRASet(self.shapes, self.rank, self.alpha, dora=self.dora, base_norms=base_norms)

    def set_active(self, lora: Optional[LoRASet]) -> None:
        self.active = lora

    def param_count(self) -> int:
        return sum(p.numel() for p in self.new_set().parameters())
