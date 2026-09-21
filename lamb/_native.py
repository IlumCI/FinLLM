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


def evaluate(expr: str) -> Optional[int]:
    return _rust.evaluate(expr) if _rust is not None else _fallback.evaluate(expr)


def verify(expr: str, answer: str) -> bool:
    return _rust.verify(expr, answer) if _rust is not None else _fallback.verify(expr, answer)


def sample_problem(op: str, a_digits: int, b_digits: int, seed: int) -> Tuple[str, str]:
    if _rust is not None:
        return _rust.sample_problem(op, int(a_digits), int(b_digits), int(seed))
    return _fallback.sample_problem(op, a_digits, b_digits, seed)


TopKStore = _rust.TopKStore if _rust is not None else _fallback.TopKStore


def backend() -> str:
    return "rust" if USING_RUST else "python"


__all__ = ["evaluate", "verify", "sample_problem", "TopKStore", "USING_RUST", "backend"]
