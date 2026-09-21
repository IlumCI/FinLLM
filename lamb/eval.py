"""Evaluation utilities: held-out arithmetic accuracy and test-time scaling."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from ._native import sample_problem, verify
from .model.lamb import LAMb
from .tokenizer import ArithmeticTokenizer


@torch.no_grad()
def evaluate(
    model: LAMb,
    tokenizer: ArithmeticTokenizer,
    grid: List[Tuple[str, int, int]],
    n_per_cell: int = 16,
    device: str = "cpu",
    seed: int = 1234,
    n_steps: Optional[int] = None,
    max_answer_len: int = 20,
) -> Dict[str, object]:
    """Solve fresh problems for every difficulty cell and score with the verifier.

    Returns ``overall`` accuracy and a ``per_cell`` map. ``n_steps`` overrides the
    latent thinking budget, so passing a larger value probes test-time compute
    scaling on a fixed model.
    """
    model.eval()
    rng = random.Random(seed)
    problems: List[str] = []
    cell_of: List[int] = []
    for ci, (op, a, b) in enumerate(grid):
        for _ in range(n_per_cell):
            expr, _ = sample_problem(op, a, b, rng.randint(0, 2**31 - 1))
            problems.append(expr)
            cell_of.append(ci)

    preds = model.solve(
        problems, tokenizer, max_answer_len=max_answer_len, n_steps=n_steps, device=device
    )

    correct = defaultdict(int)
    total = defaultdict(int)
    n_ok = 0
    for ci, expr, pred in zip(cell_of, problems, preds):
        ok = verify(expr, pred) if pred is not None else False
        correct[ci] += int(ok)
        total[ci] += 1
        n_ok += int(ok)

    per_cell = {grid[ci]: correct[ci] / total[ci] for ci in total}
    return {"overall": n_ok / max(1, len(problems)), "per_cell": per_cell}


@torch.no_grad()
def length_generalization(
    model: LAMb,
    tokenizer: ArithmeticTokenizer,
    ops: Tuple[str, ...],
    max_test_digits: int,
    train_max_digits: int,
    n_per_cell: int = 48,
    device: str = "cpu",
    seed: int = 4321,
    n_steps: Optional[int] = None,
) -> Dict[int, float]:
    """Exact-match accuracy by operand digit width, including widths beyond training.

    The single most diagnostic benchmark for the number-native design: train on
    ``<= train_max_digits`` and read off how accuracy holds (or decays) as operands
    grow to ``max_test_digits``. Each key is a digit width ``d`` (both operands ``d``
    wide); widths ``> train_max_digits`` are extrapolation.
    """
    out: Dict[int, float] = {}
    budget = 2 * max_test_digits + 2
    for d in range(1, max_test_digits + 1):
        grid = [(op, d, d) for op in ops]
        res = evaluate(
            model, tokenizer, grid, n_per_cell=n_per_cell, device=device, seed=seed + d,
            n_steps=n_steps, max_answer_len=budget,
        )
        out[d] = float(res["overall"])
    return out


@torch.no_grad()
def test_time_scaling(
    model: LAMb,
    tokenizer: ArithmeticTokenizer,
    grid: List[Tuple[str, int, int]],
    steps_list: List[int],
    n_per_cell: int = 16,
    device: str = "cpu",
) -> Dict[int, float]:
    """Accuracy as a function of the latent thinking budget ``T`` (same weights)."""
    out: Dict[int, float] = {}
    for t in steps_list:
        res = evaluate(model, tokenizer, grid, n_per_cell=n_per_cell, device=device, n_steps=t)
        out[t] = float(res["overall"])
    return out
