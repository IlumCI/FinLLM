"""Task proposers -- the challenger in LAMb's self-play.

All proposers share one interface, so the trainer can swap them freely:

    probs(s)            -> np.ndarray        distribution over difficulty cells
    sample(n, s)        -> list[int]         n cell indices
    entropy(s)          -> float
    update(s, cells, r) -> dict              learn from (state, actions, rewards)

``s`` is the per-cell **solver-competence state** (smoothed success rate). The
default :class:`BanditProposer` is a stateless *learning-progress bandit*
(``softmax(beta * 4 s (1-s))``): it rides the learnable frontier, cannot collapse
(a mastered cell's learnability -> 0, so mass moves on), and has no parameters to
train. The GRPO-trained hypernetwork proposer
(:class:`lamb.selfplay.hyperproposer.GRPOHyperProposer`) implements the same
interface and uses the bandit as its stability anchor.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np


def learnability(s: np.ndarray) -> np.ndarray:
    """Goldilocks learning-progress score 4 s (1-s): peaks at s=0.5, zero at 0/1."""
    s = np.asarray(s, dtype=np.float64)
    return 4.0 * s * (1.0 - s)


class BaseProposer:
    """Interface shared by every proposer. ``update`` is a no-op by default."""

    def probs(self, s: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def sample(self, n: int, s: np.ndarray) -> List[int]:  # pragma: no cover - abstract
        raise NotImplementedError

    def entropy(self, s: np.ndarray) -> float:
        p = self.probs(s)
        return float(-(p * np.log(p + 1e-12)).sum())

    def update(self, s: np.ndarray, cells: List[int], rewards: List[float]) -> Dict[str, float]:
        return {}


class BanditProposer(BaseProposer):
    def __init__(self, n_cells: int, temperature: float = 6.0, eps: float = 0.35,
                 rng: Optional[random.Random] = None):
        self.n_cells = n_cells
        self.temperature = temperature
        self.eps = eps
        self.rng = rng or random.Random()

    def probs(self, s: np.ndarray) -> np.ndarray:
        # Size-agnostic: works over whatever number of cells `s` describes, so an
        # append-only open-ended space grows under it without reconstruction.
        m = len(s)
        z = self.temperature * learnability(s)
        z -= z.max()
        e = np.exp(z)
        p = e / e.sum()
        p = (1.0 - self.eps) * p + self.eps / m
        return p / p.sum()

    def sample(self, n: int, s: np.ndarray) -> List[int]:
        p = self.probs(s)
        return self.rng.choices(range(len(s)), weights=p.tolist(), k=n)


# Backwards-compatible alias.
Proposer = BanditProposer
