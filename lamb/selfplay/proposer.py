"""Task proposer -- the challenger in LAMb's self-play.

A *learning-progress bandit* over a discrete grid of difficulty cells ``(op,
a_digits, b_digits)``. Each cell has a learnability score ``L = 4*s*(1-s)``
(``s`` = the solver's smoothed success rate on that cell), which peaks when the
solver gets it right about half the time and vanishes for cells that are already
mastered (``s->1``) or still impossible (``s->0``). The proposer samples from

    p  =  (1 - eps) * softmax(beta * L)  +  eps * uniform

so probability mass rides the *frontier* -- the band of currently-learnable
cells -- and, because a mastered cell's ``L`` collapses to zero, automatically
moves on to harder cells as the solver improves. The ``eps`` floor guarantees
every cell keeps a trickle of exposure. This is the challenger-solver
co-evolution of Absolute-Zero / R-Zero cast as automatic curriculum learning
(learning-progress-driven task selection); it needs no external data and, unlike
a REINFORCE policy, cannot collapse. A fully neural proposer trained by RL is a
documented upgrade path.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np


class Proposer:
    def __init__(self, n_cells: int, temperature: float = 6.0, eps: float = 0.35,
                 rng: random.Random | None = None):
        self.n_cells = n_cells
        self.temperature = temperature
        self.eps = eps
        self.rng = rng or random.Random()

    def probs(self, learnability: np.ndarray) -> np.ndarray:
        z = self.temperature * np.asarray(learnability, dtype=np.float64)
        z -= z.max()
        e = np.exp(z)
        p = e / e.sum()
        p = (1.0 - self.eps) * p + self.eps / self.n_cells
        return p / p.sum()

    def sample(self, n: int, learnability: np.ndarray) -> Tuple[List[int], np.ndarray]:
        p = self.probs(learnability)
        cells = self.rng.choices(range(self.n_cells), weights=p.tolist(), k=n)
        return cells, p

    def entropy(self, learnability: np.ndarray) -> float:
        p = self.probs(learnability)
        return float(-(p * np.log(p + 1e-12)).sum())
