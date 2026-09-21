"""Behavioural novelty for open-ended admission (novelty search / POET).

Descriptor dedup admits a candidate environment iff its ``(depth, digits, ops)``
tuple is new -- a *structural* test. Two structurally-different environments can
be behaviourally identical (e.g. induce the same solver competence and demand the
same answers), and admitting both just bloats the population without adding
diversity. Behavioural novelty instead characterises each environment by *how it
behaves* and admits a candidate only if it is far, in behaviour space, from the
environments already admitted -- the diversity-maintenance ingredient that
novelty search (Lehman & Stanley) and POET rely on.

Behaviour characterisation (BC) of an environment, under a reference solver:

    [ acc@T1, acc@T2, acc@Tmax, mean_answer_length, gain_from_thinking ]

i.e. the solver's competence across latent-step budgets (behavioural difficulty),
the answer length the tasks demand (structural load), and how much extra latent
compute helps (a behavioural signature of how much reasoning the environment
needs). Novelty is the mean distance to the k nearest BCs in the archive.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np


def behaviour_characterization(
    solve: Callable[[List[str], int], List],
    tasks: Sequence[Tuple[str, str]],
    budgets: Sequence[int] = (1, 2, 4),
) -> np.ndarray:
    """BC vector for an environment. ``solve(problems, n_steps)`` returns predictions
    (strings or ``None``); ``tasks`` is a fixed list of ``(problem, answer)``."""
    problems = [p for p, _ in tasks]
    answers = [a for _, a in tasks]
    accs = []
    for t in budgets:
        preds = solve(problems, t)
        accs.append(sum(1 for pr, a in zip(preds, answers) if pr is not None and pr == a) / max(1, len(answers)))
    mean_len = np.mean([len(a.lstrip("-")) for a in answers]) / 10.0
    gain = accs[-1] - accs[0]
    return np.array(accs + [float(mean_len), float(gain)], dtype=np.float64)


class NoveltyArchive:
    def __init__(self, k: int = 3, threshold: float = 0.15):
        self.k = k
        self.threshold = threshold
        self.bcs: List[np.ndarray] = []

    def novelty(self, bc: np.ndarray) -> float:
        if not self.bcs:
            return float("inf")
        dists = sorted(float(np.linalg.norm(bc - b)) for b in self.bcs)
        knn = dists[: self.k]
        return sum(knn) / len(knn)

    def is_novel(self, bc: np.ndarray) -> bool:
        return self.novelty(bc) >= self.threshold

    def add(self, bc: np.ndarray) -> None:
        self.bcs.append(bc)

    def __len__(self) -> int:
        return len(self.bcs)
