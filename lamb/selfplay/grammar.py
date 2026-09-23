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
from typing import Callable, List, Optional, Sequence, Tuple

from .._native import evaluate
from ..holdout import is_heldout

OPS_SETS: Tuple[Tuple[str, ...], ...] = (("+", "-"), ("+", "-", "*"),
                                         ("+", "-", "*", "/"))

_MAX_ABS = 2 ** 126  # stay inside the i128 range the native verifier uses


def _apply(op: str, a: int, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0 or a % b != 0:
            raise ValueError("inexact division should have been avoided upstream")
        return a // b
    raise ValueError(f"unsupported operator {op!r}")


def _exact_division_operands(digits: int, rng: random.Random,
                             num: "Callable[[int, random.Random], int]") -> Tuple[int, int]:
    """A dividend/divisor pair that divides exactly.

    Constructed rather than rejected: picking two operands and resampling until the
    division happens to be exact wastes most draws and biases the distribution
    toward small divisors. Choosing the divisor and quotient and multiplying gives
    a uniform, exact pair in one go. The dividend can exceed the nominal digit
    width, which is correct -- "96 split into boxes of 12" is a normal thing for a
    problem to say.
    """
    b = 0
    while b == 0:
        b = num(digits, rng)
    return b * num(digits, rng), b


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
            op = rng.choice(ops)
            if op == "/":
                a, b = _exact_division_operands(digits, rng, self._num)
            else:
                a, b = self._num(digits, rng), self._num(digits, rng)
            return f"{a}{op}{b}"
        left = self._build(depth - 1, digits, ops, rng)
        right = self._build(depth - 1, digits, ops, rng)
        return f"({left}){rng.choice(ops)}({right})"

    def _build_traced(self, depth: int, digits: int, ops: Sequence[str],
                      rng: random.Random) -> Tuple[str, int, List[int]]:
        """``(expr, value, trace)`` where ``trace`` is the sub-expression values in
        post-order, *excluding* the root (which is the answer)."""
        if depth <= 1:
            # Draw order is load-bearing: operands then operator, exactly as before
            # division existed. Choosing the operator first reshuffles every problem
            # this grammar has ever produced, silently invalidating every
            # measurement taken against it. Division consumes *extra* draws after
            # the fact, so op sets without "/" stay byte-identical.
            a, b = self._num(digits, rng), self._num(digits, rng)
            op = rng.choice(ops)
            if op == "/":
                a, b = _exact_division_operands(digits, rng, self._num)
            return f"{a}{op}{b}", _apply(op, a, b), []
        lexpr, lval, ltrace = self._build_traced(depth - 1, digits, ops, rng)
        rexpr, rval, rtrace = self._build_traced(depth - 1, digits, ops, rng)
        op = rng.choice(ops)
        if op == "/" and (rval == 0 or lval % rval != 0):
            # An internal division's operands are already fixed by the subtrees, so
            # exactness cannot be arranged -- substitute rather than resample the
            # whole tree, which would bias against deep expressions entirely.
            op = "*"
        # post-order: everything each child computed, then the child's own value
        return f"({lexpr}){op}({rexpr})", _apply(op, lval, rval), ltrace + [lval] + rtrace + [rval]

    def sample_with_trace(self, descriptor: Descriptor, seed: int,
                          exclude_heldout: bool = False) -> Tuple[str, str, List[int]]:
        """``(expression, exact_answer, trace)`` -- the gold *numeric* reasoning trace.

        ``trace`` holds the intermediate sub-expression values in evaluation order,
        excluding the final answer. It is generated exactly and for free by the
        grammar itself: no language, no external data, no human annotation. This is
        what supervises each latent position in the LOTUS-style parallel latent
        block, standing in for the gold chain-of-thought tokens that method uses.
        """
        rng = random.Random(seed)
        ops = self.ops_sets[descriptor.ops_key]
        for _ in range(32):  # resample on i128 overflow, or if it lands in the eval partition
            expr, val, trace = self._build_traced(descriptor.depth, descriptor.digits, ops, rng)
            if abs(val) >= _MAX_ABS or any(abs(t) >= _MAX_ABS for t in trace):
                continue
            if exclude_heldout and is_heldout(expr):
                continue
            return expr, str(val), trace
        for _ in range(32):   # degrade to a binary op that cannot overflow
            a, b = self._num(descriptor.digits, rng), self._num(descriptor.digits, rng)
            expr = f"{a}+{b}"
            if not (exclude_heldout and is_heldout(expr)):
                return expr, str(a + b), []
        return expr, str(a + b), []

    def sample(self, descriptor: Descriptor, seed: int,
               exclude_heldout: bool = False) -> Tuple[str, str]:
        """Return a ``(expression, exact_answer)`` pair for the descriptor."""
        rng = random.Random(seed)
        ops = self.ops_sets[descriptor.ops_key]
        for _ in range(32):  # resample on overflow, or if it lands in the eval partition
            expr = self._build(descriptor.depth, descriptor.digits, ops, rng)
            val = evaluate(expr)
            if val is None:
                continue
            if exclude_heldout and is_heldout(expr):
                continue
            return expr, str(val)
        # Degrade to a single binary op that cannot overflow at these widths.
        for _ in range(32):
            a, b = self._num(descriptor.digits, rng), self._num(descriptor.digits, rng)
            expr = f"{a}+{b}"
            if not (exclude_heldout and is_heldout(expr)):
                return expr, str(a + b)
        return expr, str(a + b)

    def sample_heldout(self, descriptor: Descriptor, seed: int,
                       tries: int = 256) -> Tuple[str, str]:
        """A problem drawn from the **evaluation** partition.

        The mirror of ``sample(..., exclude_heldout=True)``: training rejects this
        partition and evaluation draws only from it, so the two never meet however
        long training runs. Falls back to an unfiltered draw only if the descriptor's
        space is so small that the partition is effectively empty.
        """
        rng = random.Random(seed)
        last = self.sample(descriptor, seed)
        for _ in range(tries):
            expr, ans = self.sample(descriptor, rng.randint(0, 2 ** 31 - 1))
            last = (expr, ans)
            if is_heldout(expr):
                return expr, ans
        return last

    def max_answer_len(self, max_depth: int, max_digits: int) -> int:
        # Generous upper bound on answer token length across the reachable space.
        return min(24, 2 + max_digits * (2 ** min(max_depth, 3)))
