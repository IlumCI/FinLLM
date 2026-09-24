"""Dispatch layer selecting the Rust ``lamb_core`` kernels or the Python fallback.

Import ``lamb._native`` and use ``evaluate``, ``verify``, ``sample_problem`` and
``TopKStore`` from here; the rest of the package never imports ``lamb_core``
directly, so behaviour is identical with or without the compiled extension
(``USING_RUST`` records which path is active).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import _fallback

try:  # pragma: no cover - trivial import guard
    import lamb_core as _rust

    USING_RUST = bool(getattr(_rust, "USING_RUST", True))
except Exception:  # noqa: BLE001 - any failure means: fall back to Python
    _rust = None
    USING_RUST = False


def _rust_handles(expr: str) -> bool:
    """Can the compiled kernel be trusted with this expression?

    Division routes to the Python path even when the Rust kernels are present. The
    compiled extension predates ``/`` -- its lexer has no token for it -- and an
    already-installed build cannot be assumed to know it, so dispatching on the
    operator is robust against a stale ``.so`` in a way that rebuilding is not. It
    costs nothing measurable: problem generation profiles at 0.2% of a training step.

    **This is one predicate because it used to be two, and they drifted.** The guard
    was written into :func:`evaluate` and not into :func:`verify`, so on any machine
    with the extension built, ``verify("48/2", "24")`` reached a lexer that rejects
    ``/`` and returned **False** -- scoring every division problem wrong and silently
    corrupting the only reward signal the self-play loop has. CI runs the Python
    backend, so nothing ever saw it. A condition restated in two places is a condition
    that will disagree in one of them.
    """
    return _rust is not None and "/" not in expr


def evaluate(expr: str) -> Optional[int]:
    """Exact value of an integer expression, or ``None``."""
    if _rust_handles(expr):
        return _rust.evaluate(expr)
    return _fallback.evaluate(expr)


def verify(expr: str, answer: str) -> bool:
    if _rust_handles(expr):
        return _rust.verify(expr, answer)
    return _fallback.verify(expr, answer)


def sample_problem(op: str, a_digits: int, b_digits: int, seed: int) -> Tuple[str, str]:
    # Same reasoning, on the operator rather than the expression: a kernel with no
    # ``/`` token cannot generate a division problem either.
    if _rust is not None and op != "/":
        return _rust.sample_problem(op, int(a_digits), int(b_digits), int(seed))
    return _fallback.sample_problem(op, a_digits, b_digits, seed)


TopKStore = _rust.TopKStore if _rust is not None else _fallback.TopKStore


def backend() -> str:
    return "rust" if USING_RUST else "python"


__all__ = ["evaluate", "verify", "sample_problem", "TopKStore", "USING_RUST",
           "backend"]
