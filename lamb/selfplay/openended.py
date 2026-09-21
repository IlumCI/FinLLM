"""Curricula: the fixed grid, and the open-ended grammar with MCC admission.

Both expose one interface so the trainer is agnostic to which is in use:

    descriptors()                 -> list (indices are stable / append-only)
    sample(idx, seed)             -> (problem, exact_answer)
    label(idx) / complexity(idx)  -> logging / frontier score
    admit(success_ema)            -> grow the space (no-op for the fixed grid)
    mastered_frontier(...)        -> current mastered complexity

The open-ended curriculum realises **minimal criterion coevolution**: a
descriptor's harder neighbours (deeper, wider, richer ops) are admitted *only
once the descriptor itself is mastered*. The exact verifier guarantees every
admitted descriptor is well-posed, so the reachable task space grows without
bound, gated purely by the solver's own competence -- the mechanism that keeps
``dominance`` positive instead of saturating on a fixed grid.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import TrainConfig
from .grammar import Descriptor, TaskGrammar
from .verifier import Verifier


class FixedGridCurriculum:
    """Wraps the original ``(op, a_digits, b_digits)`` grid."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.verifier = Verifier()
        self._descs: List[Tuple[str, int, int]] = cfg.difficulty_grid()

    def descriptors(self) -> List[Tuple[str, int, int]]:
        return self._descs

    def sample(self, idx: int, seed: int) -> Tuple[str, str]:
        op, a, b = self._descs[idx]
        return self.verifier.sample(op, a, b, seed)

    def label(self, idx: int) -> str:
        op, a, b = self._descs[idx]
        return f"{op}{a}x{b}"

    def complexity(self, idx: int) -> float:
        _, a, b = self._descs[idx]
        return float(a + b)

    def admit(self, success_ema: Dict[int, float]) -> int:
        return 0

    def mastered_frontier(self, success_ema: Dict[int, float]) -> float:
        thr = self.cfg.mastery_threshold
        vals = [self.complexity(i) for i in range(len(self._descs)) if success_ema.get(i, 0.0) >= thr]
        return max(vals) if vals else 0.0

    def answer_budget(self) -> int:
        span = 2 * self.cfg.max_digits + 2 if "*" in self.cfg.ops else self.cfg.max_digits + 2
        return span + 1

    def marginals(self, success_ema: Dict[int, float]) -> Optional[np.ndarray]:
        return None


class OpenEndedCurriculum:
    """Grammar-backed, append-only descriptor space with MCC admission."""

    def __init__(self, cfg: TrainConfig, grammar: Optional[TaskGrammar] = None):
        self.cfg = cfg
        self.grammar = grammar or TaskGrammar()
        self.max_depth = cfg.oe_max_depth
        self.max_digits = cfg.oe_max_digits
        self.n_ops = len(self.grammar.ops_sets)
        self._descs: List[Descriptor] = []
        self._index: Dict[Descriptor, int] = {}
        # Seed: the simplest tasks only (depth 1, narrowest operands, first op set).
        for g in range(1, cfg.start_digits + 1):
            self._add(Descriptor(1, g, 0))

    # -- space management -------------------------------------------------
    def _valid(self, d: Descriptor) -> bool:
        return 1 <= d.depth <= self.max_depth and 1 <= d.digits <= self.max_digits and 0 <= d.ops_key < self.n_ops

    def _add(self, d: Descriptor) -> bool:
        if d in self._index or not self._valid(d):
            return False
        self._index[d] = len(self._descs)
        self._descs.append(d)
        return True

    def descriptors(self) -> List[Descriptor]:
        return self._descs

    def sample(self, idx: int, seed: int) -> Tuple[str, str]:
        return self.grammar.sample(self._descs[idx], seed)

    def label(self, idx: int) -> str:
        return self._descs[idx].label()

    def complexity(self, idx: int) -> float:
        d = self._descs[idx]
        return float(2 * d.depth + d.digits + d.ops_key)

    def admit(self, success_ema: Dict[int, float]) -> int:
        """MCC: unlock harder neighbours of every mastered descriptor."""
        thr = self.cfg.mastery_threshold
        added = 0
        for i, d in enumerate(list(self._descs)):
            if success_ema.get(i, 0.0) < thr:
                continue
            for nb in (
                Descriptor(d.depth + 1, d.digits, d.ops_key),
                Descriptor(d.depth, d.digits + 1, d.ops_key),
                Descriptor(d.depth, d.digits, d.ops_key + 1),
            ):
                added += int(self._add(nb))
        return added

    def mastered_frontier(self, success_ema: Dict[int, float]) -> float:
        thr = self.cfg.mastery_threshold
        vals = [self.complexity(i) for i in range(len(self._descs)) if success_ema.get(i, 0.0) >= thr]
        return max(vals) if vals else 0.0

    def answer_budget(self) -> int:
        return self.grammar.max_answer_len(self.max_depth, self.max_digits)

    # -- factored proposer support ---------------------------------------
    def marginals(self, success_ema: Dict[int, float]) -> np.ndarray:
        """Per-depth and per-digit mean competence (fixed-width context vector)."""
        depth_num = np.zeros(self.max_depth)
        depth_den = np.zeros(self.max_depth)
        dig_num = np.zeros(self.max_digits)
        dig_den = np.zeros(self.max_digits)
        for i, d in enumerate(self._descs):
            s = success_ema.get(i, 0.0)
            depth_num[d.depth - 1] += s
            depth_den[d.depth - 1] += 1
            dig_num[d.digits - 1] += s
            dig_den[d.digits - 1] += 1
        depth_m = depth_num / np.maximum(depth_den, 1)
        dig_m = dig_num / np.maximum(dig_den, 1)
        return np.concatenate([depth_m, dig_m])

    def admitted_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Which depths / digits / op-sets currently contain an admitted descriptor."""
        dmask = np.zeros(self.max_depth, dtype=bool)
        gmask = np.zeros(self.max_digits, dtype=bool)
        omask = np.zeros(self.n_ops, dtype=bool)
        for d in self._descs:
            dmask[d.depth - 1] = True
            gmask[d.digits - 1] = True
            omask[d.ops_key] = True
        return dmask, gmask, omask

    def index_of(self, d: Descriptor) -> Optional[int]:
        return self._index.get(d)


def build_curriculum(cfg: TrainConfig):
    return OpenEndedCurriculum(cfg) if cfg.open_ended else FixedGridCurriculum(cfg)
