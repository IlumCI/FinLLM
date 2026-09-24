"""Native kernels (Rust or Python fallback) and cross-implementation parity."""

from __future__ import annotations

import pytest

from lamb import _fallback, _native


def test_backend_reports_something():
    assert _native.backend() in {"rust", "python"}


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2+3", 5),
        ("2+3*4", 14),
        ("(2+3)*4", 20),
        ("-5+2", -3),
        ("7- -3", 10),
        ("123+877", 1000),
    ],
)
def test_evaluate_matches_fallback(expr, expected):
    assert _native.evaluate(expr) == expected
    assert _fallback.evaluate(expr) == expected


@pytest.mark.parametrize("expr", ["", "2+", "2 2", "2/3", "(2+3", "abc"])
def test_evaluate_rejects_garbage(expr):
    assert _native.evaluate(expr) is None
    assert _fallback.evaluate(expr) is None


def test_verify():
    assert _native.verify("123+877", "1000")
    assert not _native.verify("123+877", "999")
    assert _native.verify("-4*-4", "16")
    assert not _native.verify("1+1", "banana")


@pytest.mark.parametrize("op", ["+", "-", "*"])
def test_sampled_problems_are_solvable(op):
    for seed in range(300):
        expr, ans = _native.sample_problem(op, 3, 2, seed)
        assert _native.verify(expr, ans), f"{expr} != {ans}"


def test_sample_problem_digit_widths():
    expr, _ = _native.sample_problem("+", 3, 1, 7)
    left = expr.split("+")[0]
    assert len(left) == 3


def test_topk_store_retrieval():
    store = _native.TopKStore(3)
    store.add([1.0, 0.0, 0.0], 10)
    store.add([0.0, 1.0, 0.0], 20)
    store.add([0.9, 0.1, 0.0], 30)
    out = store.query([1.0, 0.0, 0.0], 2)
    assert len(out) == 2
    assert out[0][0] == 10  # nearest
    assert out[1][0] == 30
    assert len(store) == 3


def test_topk_fallback_matches():
    for Store in (_native.TopKStore, _fallback.TopKStore):
        s = Store(2)
        s.add([1.0, 0.0], 1)
        s.add([0.0, 1.0], 2)
        top = s.query([0.8, 0.2], 1)
        assert top[0][0] == 1


def test_division_dispatches_the_same_way_for_evaluate_and_verify():
    """The guard was on ``evaluate`` and not on ``verify``, and they disagreed.

    The Rust lexer has no ``/`` token, so on a machine with the extension built
    ``verify("48/2", "24")`` reached it and returned **False** -- every division
    problem in self-play scored wrong, silently corrupting the only reward signal the
    loop has. CI runs the Python backend, so the asymmetry was invisible; the commit
    that added division guarded the function it was looking at and not the one that
    produces the reward.

    This test passes trivially on the Python backend and is the one that would have
    caught it on a Rust build, which is the whole point of writing it against the
    *dispatch* rather than against either implementation.
    """
    from lamb._native import evaluate, verify

    for expr, want in (("48/2", 24), ("81/9", 9), ("(6+6)/(2+1)", 4), ("(48/2)+3", 27)):
        assert evaluate(expr) == want, f"{expr} evaluated wrongly"
        assert verify(expr, str(want)) is True, f"{expr} failed to verify its own value"
        assert verify(expr, str(want + 1)) is False


def test_inexact_division_is_none_rather_than_a_rounded_answer():
    """``7/2`` has no integer value, and a ring cannot hold a rounded one. The
    generator constructs exact quotients precisely so this never has to be decided
    downstream."""
    from lamb._native import evaluate, verify

    assert evaluate("7/2") is None
    assert verify("7/2", "3") is False
    assert verify("7/2", "4") is False
