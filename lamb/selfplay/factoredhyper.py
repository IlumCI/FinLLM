"""Factored GRPO hypernetwork proposer for the open-ended space.

On the open-ended grammar the descriptor set grows (append-only), so a proposer
with one output per descriptor (the grid hypernetwork, or a table) cannot keep up.
This proposer instead emits a **factored** distribution -- independent softmaxes
over depth, operand digits, and operator set -- from a fixed-width competence
context (per-depth and per-digit marginal success). A descriptor's probability is
the product of its factors', so a combinatorial descriptor space costs only
``D + G + O`` outputs where a table costs ``D * G * O``. This is the concrete
sense in which the hypernetwork beats the bandit once the space is open-ended.

Trained by GRPO (group-relative advantages over the step's proposals), with a KL
anchor to a uniform-over-admitted factored prior for stability. Implements the
``BaseProposer`` interface over the curriculum's current descriptor list.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .proposer import BaseProposer

_NEG = -1e9


class FactoredHyperProposer(BaseProposer, nn.Module):
    def __init__(
        self,
        curriculum,
        hidden: int = 64,
        eps: float = 0.2,
        kl_coef: float = 0.1,
        entropy_coef: float = 0.05,
        lr: float = 5e-3,
        normalize_std: bool = True,
        rng: Optional[random.Random] = None,
    ):
        nn.Module.__init__(self)
        self.cur = curriculum
        self.D = curriculum.max_depth
        self.G = curriculum.max_digits
        self.O = curriculum.n_ops
        self.eps = eps
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.normalize_std = normalize_std
        self.rng = rng or random.Random()
        self.net = nn.Sequential(
            nn.Linear(self.D + self.G, hidden), nn.Tanh(), nn.Linear(hidden, self.D + self.G + self.O)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    # -- context / factor logits -----------------------------------------
    def _context(self, s: np.ndarray) -> np.ndarray:
        descs = self.cur.descriptors()
        dnum, dden = np.zeros(self.D), np.zeros(self.D)
        gnum, gden = np.zeros(self.G), np.zeros(self.G)
        for i, d in enumerate(descs):
            val = float(s[i]) if i < len(s) else 0.0
            dnum[d.depth - 1] += val
            dden[d.depth - 1] += 1
            gnum[d.digits - 1] += val
            gden[d.digits - 1] += 1
        return np.concatenate([dnum / np.maximum(dden, 1), gnum / np.maximum(gden, 1)])

    def _masked_logsoftmax(self, logits: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        m = torch.as_tensor(mask, dtype=torch.bool)
        masked = torch.where(m, logits, torch.full_like(logits, _NEG))
        return torch.log_softmax(masked, dim=0)

    def _factor_logps(self, s: np.ndarray):
        logits = self.net(torch.as_tensor(self._context(s), dtype=torch.float32))
        dmask, gmask, omask = self.cur.admitted_masks()
        dlp = self._masked_logsoftmax(logits[: self.D], dmask)
        glp = self._masked_logsoftmax(logits[self.D : self.D + self.G], gmask)
        olp = self._masked_logsoftmax(logits[self.D + self.G :], omask)
        return dlp, glp, olp

    # -- BaseProposer interface ------------------------------------------
    def probs(self, s: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            dlp, glp, olp = self._factor_logps(s)
        dlp, glp, olp = (dlp.detach().cpu().numpy(), glp.detach().cpu().numpy(),
                         olp.detach().cpu().numpy())
        descs = self.cur.descriptors()
        scores = np.array([dlp[d.depth - 1] + glp[d.digits - 1] + olp[d.ops_key] for d in descs])
        scores -= scores.max()
        p = np.exp(scores)
        p /= p.sum()
        p = (1.0 - self.eps) * p + self.eps / len(descs)
        return p / p.sum()

    def sample(self, n: int, s: np.ndarray) -> List[int]:
        p = self.probs(s)
        return self.rng.choices(range(len(p)), weights=p.tolist(), k=n)

    def update(self, s: np.ndarray, cells: List[int], rewards: List[float]) -> Dict[str, float]:
        if not cells:
            return {}
        dlp, glp, olp = self._factor_logps(s)
        descs = self.cur.descriptors()
        logp = torch.stack([
            dlp[descs[c].depth - 1] + glp[descs[c].digits - 1] + olp[descs[c].ops_key] for c in cells
        ])
        r = torch.as_tensor(rewards, dtype=torch.float32)
        adv = r - r.mean()
        if self.normalize_std:
            adv = adv / (r.std(unbiased=False) + 1e-6)
        pg = -(adv.detach() * logp).mean()

        # KL to a uniform-over-admitted factored prior + entropy (anti-collapse).
        dmask, gmask, omask = self.cur.admitted_masks()
        kl = self._kl_to_uniform(dlp, dmask) + self._kl_to_uniform(glp, gmask) + self._kl_to_uniform(olp, omask)
        ent = -(dlp.exp() * dlp).sum() - (glp.exp() * glp).sum() - (olp.exp() * olp).sum()
        loss = pg + self.kl_coef * kl - self.entropy_coef * ent

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"prop_pg": float(pg.detach()), "prop_kl": float(kl.detach())}

    @staticmethod
    def _kl_to_uniform(logp: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        m = torch.as_tensor(mask, dtype=torch.float32)
        k = m.sum().clamp_min(1.0)
        log_u = torch.where(m.bool(), torch.log(1.0 / k), torch.zeros_like(logp))
        p = logp.exp() * m
        return (p * (logp - log_u)).sum()
