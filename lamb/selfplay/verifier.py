"""Verifier -- the exact, deterministic reward oracle for self-play.

Wraps the native (Rust) exact-arithmetic kernels. All reward in LAMb's
self-improvement loop flows from :meth:`Verifier.check`; there is no learned
reward model and no human-labelled data.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .._native import backend, sample_problem, verify


class Verifier:
    def sample(self, op: str, a_digits: int, b_digits: int, seed: int) -> Tuple[str, str]:
        """Return a ``(problem, exact_answer)`` pair at the given difficulty."""
        return sample_problem(op, a_digits, b_digits, seed)

    def check(self, problem: str, answer: Optional[str]) -> bool:
        """True iff ``answer`` exactly solves ``problem``."""
        if answer is None:
            return False
        return verify(problem, answer)

    @property
    def backend(self) -> str:
        return backend()
