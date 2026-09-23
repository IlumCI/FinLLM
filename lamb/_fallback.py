"""Pure-Python fallbacks for the ``lamb_core`` Rust extension.

These mirror the native kernels so the package is fully functional (if slower)
when the compiled extension is unavailable. Correctness -- not speed -- is the
contract here; ``lamb._native`` dispatches to the Rust versions when present.

The evaluator uses a restricted ``ast`` walk rather than ``eval`` so that
untrusted expression strings can never execute arbitrary code.
"""

from __future__ import annotations

import ast
import operator
import random
from typing import List, Optional, Tuple

import numpy as np

USING_RUST = False

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _exact_div(a: int, b: int) -> Optional[int]:
    """Integer division, or ``None`` when it is not exact.

    The contract here is ``Optional[int]``, so a non-exact quotient has no
    representation and must be rejected rather than rounded -- a rounded answer
    would be a wrong answer presented as a right one. Problems needing true
    rationals go through :mod:`lamb.rational`, which represents them exactly.
    """
    if b == 0 or a % b != 0:
        return None
    return a // b


_BIN[ast.Div] = _exact_div          # registered after the helper it points at


def _eval_node(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if left is None or right is None:
            return None
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        val = _eval_node(node.operand)
        return None if val is None else _UN[type(node.op)](val)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
        node.value, bool
    ):
        return node.value
    return None


def evaluate(expr: str) -> Optional[int]:
    """Exactly evaluate an integer expression over ``+ - * ( )``; ``None`` if invalid."""
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    return _eval_node(tree)


def verify(expr: str, answer: str) -> bool:
    """True iff ``answer`` is the exact integer value of ``expr``."""
    value = evaluate(expr)
    if value is None:
        return False
    try:
        return int(answer.strip()) == value
    except (ValueError, TypeError):
        return False


def _int_with_digits(rng: random.Random, digits: int) -> int:
    d = max(1, min(18, int(digits)))
    if d == 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (d - 1), 10 ** d - 1)


def sample_problem(op: str, a_digits: int, b_digits: int, seed: int) -> Tuple[str, str]:
    """Sample a ``(problem, answer)`` pair at the given difficulty."""
    rng = random.Random(seed)
    a = _int_with_digits(rng, a_digits)
    b = _int_with_digits(rng, b_digits)
    opc = op[0] if op else "+"
    if opc == "-":
        return f"{a}-{b}", str(a - b)
    if opc == "*":
        return f"{a}*{b}", str(a * b)
    return f"{a}+{b}", str(a + b)


class TopKStore:
    """Brute-force top-k dot-product store; numpy mirror of the Rust ``TopKStore``."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self._keys: List[np.ndarray] = []
        self._ids: List[int] = []

    def add(self, key, id: int) -> None:  # noqa: A002 - match Rust signature
        vec = np.asarray(key, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"key has shape {vec.shape}, expected ({self.dim},)")
        self._keys.append(vec)
        self._ids.append(int(id))

    def add_batch(self, keys, ids) -> None:
        for key, id_ in zip(keys, ids):
            self.add(key, id_)

    def query(self, query, k: int) -> List[Tuple[int, float]]:
        if not self._ids or k == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            return []
        mat = np.stack(self._keys)
        scores = mat @ q
        order = np.argsort(-scores)[: int(k)]
        return [(int(self._ids[i]), float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self._ids)
