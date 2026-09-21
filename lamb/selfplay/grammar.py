"""Generative task grammar -- the open-ended task space (Red Queen Step 2).

A task is no longer a cell in a fixed grid; it is drawn from a grammar of nested
arithmetic expressions whose complexity grows without bound along three axes:

    depth    -- nesting depth of the balanced expression tree (1 = ``a op b``,
                2 = ``(a op b) op (c op d)``, ...), which is what forces more
                latent-reasoning steps;
    digits   -- operand width;
    ops_key  -- which operator set is allowed (``+ -`` first; ``* `` unlocked
                later, since it makes answers blow up).

The exact answer comes from the same native evaluator used as the reward oracle,
so every generated task is well-posed by construction (the ``minimal criterion``
checker in :mod:`lamb.selfplay.openended`). Values that would overflow ``i128``
are rejected and resampled, so deep ``*`` chains degrade gracefully.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .._native import evaluate

OPS_SETS: Tuple[Tuple[str, ...], ...] = (("+", "-"), ("+", "-", "*"))


@dataclass(frozen=True)
class Descriptor:
    depth: int
    digits: int
    ops_key: int

    def label(self) -> str:
        return f"d{self.depth}g{self.digits}o{self.ops_key}"


class TaskGrammar:
    def __init__(self, ops_sets: Sequence[Sequence[str]] = OPS_SETS):
        self.ops_sets = [tuple(o) for o in ops_sets]

    def _num(self, digits: int, rng: random.Random) -> int:
        d = max(1, digits)
        if d == 1:
            return rng.randint(0, 9)
        return rng.randint(10 ** (d - 1), 10 ** d - 1)

    def _build(self, depth: int, digits: int, ops: Sequence[str], rng: random.Random) -> str:
        if depth <= 1:
            a, b = self._num(digits, rng), self._num(digits, rng)
            return f"{a}{rng.choice(ops)}{b}"
        left = self._build(depth - 1, digits, ops, rng)
        right = self._build(depth - 1, digits, ops, rng)
        return f"({left}){rng.choice(ops)}({right})"

    def sample(self, descriptor: Descriptor, seed: int) -> Tuple[str, str]:
        """Return a ``(expression, exact_answer)`` pair for the descriptor."""
        rng = random.Random(seed)
        ops = self.ops_sets[descriptor.ops_key]
        for _ in range(6):  # resample if a value overflows i128
            expr = self._build(descriptor.depth, descriptor.digits, ops, rng)
            val = evaluate(expr)
            if val is not None:
                return expr, str(val)
        # Degrade to a single binary op that cannot overflow at these widths.
        a, b = self._num(descriptor.digits, rng), self._num(descriptor.digits, rng)
        expr = f"{a}+{b}"
        return expr, str(a + b)

    def max_answer_len(self, max_depth: int, max_digits: int) -> int:
        # Generous upper bound on answer token length across the reachable space.
        return min(24, 2 + max_digits * (2 ** min(max_depth, 3)))
