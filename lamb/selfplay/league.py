"""Solver league -- historical self-play for Red Queen coevolution.

A ring of periodically frozen solver snapshots. Two jobs, both essential to
turning LAMb's reactive autocurriculum into a genuine coevolutionary arms race:

* **Relative fitness.** The Red Queen signature is that the current solver keeps
  *dominating its own past* on an ever-advancing task distribution. We measure
  this directly: current vs. best-past accuracy on the *current frontier*
  (``dominance`` > 0 means the solver is still pulling ahead where the proposer
  now pushes -- escalation, not saturation).
* **Forgetting detection.** Coevolution's classic pathology is cycling/
  catastrophic forgetting. We track whether the current solver still solves the
  easy tasks an *old* snapshot had mastered (``forgetting`` > 0 means regression).

Snapshots are cheap for a tiny model. This is the historical-self-play ingredient
that Digital Red Queen (arXiv:2601.03335) identifies as sufficient (with
generation + diversity) to instantiate Red Queen dynamics; the population-of-
pairs and transfer of POET/MCC are the later, heavier steps on the roadmap.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch

from .._native import verify
from ..model.lamb import LAMb
from ..tokenizer import ArithmeticTokenizer


class League:
    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.snapshots: List[Tuple[int, LAMb]] = []

    def __len__(self) -> int:
        return len(self.snapshots)

    def maybe_snapshot(self, model: LAMb, step: int, every: int) -> bool:
        """Freeze a copy of the solver every ``every`` steps. Keeps the oldest
        snapshot (the baseline, for forgetting) plus the most recent ones."""
        if every <= 0 or step <= 0 or step % every != 0:
            return False
        snap = copy.deepcopy(model).eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self.snapshots.append((step, snap))
        if len(self.snapshots) > self.capacity:
            self.snapshots = [self.snapshots[0]] + self.snapshots[-(self.capacity - 1):]
        return True

    @torch.no_grad()
    def _accuracy(
        self, model: LAMb, tok: ArithmeticTokenizer, problems: List[str],
        answers: List[str], device: str, max_answer_len: int, n_steps: Optional[int],
    ) -> float:
        if not problems:
            return 0.0
        preds = model.solve(problems, tok, max_answer_len=max_answer_len, n_steps=n_steps, device=device)
        ok = sum(1 for pr, a in zip(preds, answers) if pr is not None and pr == a)
        return ok / len(problems)

    @torch.no_grad()
    def relative_fitness(
        self,
        current: LAMb,
        tok: ArithmeticTokenizer,
        frontier: List[Tuple[str, str]],
        retention: List[Tuple[str, str]],
        device: str = "cpu",
        max_answer_len: int = 12,
        n_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compare the current solver to its history.

        ``frontier`` / ``retention`` are ``(problem, exact_answer)`` lists: the
        current proposer's frontier, and a fixed easy set. Returns dominance
        (current minus best-past on the frontier) and forgetting (oldest minus
        current on the easy set, floored at 0).
        """
        fp = [p for p, _ in frontier]
        fa = [a for _, a in frontier]
        rp = [p for p, _ in retention]
        ra = [a for _, a in retention]

        cur_f = self._accuracy(current, tok, fp, fa, device, max_answer_len, n_steps)
        best_past_f = 0.0
        for _, snap in self.snapshots:
            best_past_f = max(best_past_f, self._accuracy(snap, tok, fp, fa, device, max_answer_len, n_steps))

        cur_r = self._accuracy(current, tok, rp, ra, device, max_answer_len, n_steps)
        oldest_r = cur_r
        if self.snapshots:
            oldest_r = self._accuracy(self.snapshots[0][1], tok, rp, ra, device, max_answer_len, n_steps)

        return {
            "current_frontier": cur_f,
            "league_best_frontier": best_past_f,
            "dominance": cur_f - best_past_f,
            "current_retention": cur_r,
            "oldest_retention": oldest_r,
            "forgetting": max(0.0, oldest_r - cur_r),
            "league_size": float(len(self.snapshots)),
        }
