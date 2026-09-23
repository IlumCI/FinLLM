"""GRPO-trained hypernetwork proposer, anchored to the learning-progress bandit.

This is the ``bandit + GRPO-trained hypernetwork`` design. A small
**hypernetwork** maps the solver's competence state ``s`` (per-cell success rate)
to the parameters of the task-proposal distribution -- here, logits over the
difficulty grid:

    logits = Hypernet(s)         # state-conditioned task distribution
    pi(.|s) = (1-eps)*softmax(logits) + eps*uniform

It is trained by **GRPO**: the cells proposed in a step form one group drawn from
state ``s``; each cell's reward is its learnability (R-Zero-style -- highest where
the solver is uncertain, i.e. ~50% success); advantages are group-relative. A
**KL penalty to the bandit** keeps the policy trust-region-close to a distribution
that provably cannot collapse -- directly countering the documented failure mode
where self-play proposers "drift towards trivial or unsolvable tasks"
(arXiv:2603.02218). Unlike the tabular bandit, the hypernetwork conditions on
state and can generalise across the grid; unlike bare REINFORCE, it is anchored
and cannot collapse.

The hypernetwork here emits the distribution's parameters directly (a lightweight
hypernetwork); emitting the weights of a separate task-generator network is the
heavier variant noted in docs/ROADMAP.md.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .proposer import BaseProposer, learnability


class GRPOHyperProposer(BaseProposer, nn.Module):
    def __init__(
        self,
        n_cells: int,
        hidden: int = 64,
        eps: float = 0.2,
        bandit_temp: float = 6.0,
        kl_coef: float = 0.1,
        entropy_coef: float = 0.01,
        lr: float = 1e-2,
        normalize_std: bool = True,
        rng: Optional[random.Random] = None,
    ):
        nn.Module.__init__(self)
        self.n_cells = n_cells
        self.eps = eps
        self.bandit_temp = bandit_temp
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.normalize_std = normalize_std
        self.rng = rng or random.Random()
        # Hypernetwork: competence state (n_cells) -> task logits (n_cells).
        self.net = nn.Sequential(
            nn.Linear(n_cells, hidden), nn.Tanh(), nn.Linear(hidden, n_cells)
        )
        # Small init so the initial policy is near-uniform (anchor-friendly).
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def _logits(self, s: np.ndarray) -> torch.Tensor:
        return self.net(torch.as_tensor(np.asarray(s, dtype=np.float32)))

    def _bandit_probs_t(self, s: np.ndarray) -> torch.Tensor:
        z = self.bandit_temp * torch.as_tensor(learnability(s), dtype=torch.float32)
        return torch.softmax(z, dim=0)

    def probs(self, s: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            p = torch.softmax(self._logits(s), dim=0).detach().cpu().numpy().astype(np.float64)
        p = (1.0 - self.eps) * p + self.eps / self.n_cells
        return p / p.sum()

    def sample(self, n: int, s: np.ndarray) -> List[int]:
        p = self.probs(s)
        return self.rng.choices(range(self.n_cells), weights=p.tolist(), k=n)

    def update(self, s: np.ndarray, cells: List[int], rewards: List[float]) -> Dict[str, float]:
        if len(cells) == 0:
            return {}
        logits = self._logits(s)
        logp = torch.log_softmax(logits, dim=0)
        p = torch.softmax(logits, dim=0)

        cells_t = torch.as_tensor(cells, dtype=torch.long)
        r = torch.as_tensor(rewards, dtype=torch.float32)
        # GRPO advantage: the step's proposals are one group drawn from state s.
        adv = r - r.mean()
        if self.normalize_std:
            adv = adv / (r.std(unbiased=False) + 1e-6)

        pg = -(adv.detach() * logp[cells_t]).mean()
        pb = self._bandit_probs_t(s)
        kl = (p * (torch.log(p + 1e-12) - torch.log(pb + 1e-12))).sum()  # KL(pi || bandit)
        ent = -(p * torch.log(p + 1e-12)).sum()
        loss = pg + self.kl_coef * kl - self.entropy_coef * ent

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"prop_pg": float(pg.detach()), "prop_kl": float(kl.detach()),
                "prop_entropy": float(ent.detach())}
