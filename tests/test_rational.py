"""Exact rationals over the residue ring: division, and decimals.

Two gaps blocked every natural-language benchmark this project targets, and both
were fatal rather than inconvenient -- one silently produced wrong numbers. These
tests pin the fix and, as much as possible, pin the *failure modes* that motivated
it, so a regression reads as the original bug rather than as a new one.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
import torch

from lamb.algebra import ResidueAlgebra, ResidueSystem, order10
from lamb.rational import RATIONAL_MODULI, RationalAlgebra


def test_small_divisors_have_no_modular_inverse_which_is_why_this_exists():
    """The motivating failure. Moduli were chosen for short digit-periods -- powers
    of 2 and 5 and divisors of 10^k-1 -- and that is exactly the set which makes
    small divisors non-invertible. Not one divisor from 2 to 12 has an inverse, and
    'half as many' is the most common operation in the target benchmark."""
    import math

    s = ResidueSystem()
    invertible = [d for d in range(2, 13)
                  if all(math.gcd(d, p) == 1 for p in s.moduli)]
    assert invertible == []


def test_rational_moduli_keep_short_digit_periods():
    """Rationals need a big ring because denominators multiply, but the periodicity
    argument still has to hold or the encoder stops extrapolating."""
    assert max(order10(p) for p in RATIONAL_MODULI) <= 6
    R = RationalAlgebra()
    assert R.headroom() > 1e15


def test_division_by_values_with_no_inverse():
    R = RationalAlgebra()
    seven = R.encode([Fraction(7)])
    for d in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12):
        got = R.decode(R.div(seven, R.encode([Fraction(d)])))[0]
        assert got == Fraction(7, d)


def test_mixed_scales_were_the_silent_bug():
    """3.25 + 7 used to give 3.32: the parser tracked scale and the algebra did not.
    A confidently wrong number is worse than a missing feature."""
    R = RationalAlgebra()
    got = R.decode(R.add(R.encode([Fraction(325, 100)]), R.encode([Fraction(7)])))[0]
    assert got == Fraction(41, 4) == Fraction("10.25")


def test_encode_scaled_is_the_join_with_the_text_parser():
    """A scale is not something the arithmetic has to track; it is a denominator."""
    from lamb.bridge import extract_quantities

    R = RationalAlgebra()
    qs = extract_quantities("It costs $3.25 and he has $7.")
    r = R.encode_scaled([q.value for q in qs], [q.scale for q in qs])
    assert R.decode(r) == [Fraction(13, 4), Fraction(7)]


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
def test_exactness_against_python_fractions(op):
    R = RationalAlgebra()
    rng = random.Random(0)
    for _ in range(40):
        x = Fraction(rng.randint(-500, 500), rng.choice([1, 1, 2, 4, 5, 10, 100]))
        y = Fraction(rng.randint(-500, 500), rng.choice([1, 1, 2, 3, 4, 100]))
        if op == "/" and y == 0:
            continue
        want = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else None}[op]
        assert R.decode(R.compose(R.encode([x]), R.encode([y]), op))[0] == want


def test_a_chain_of_operations_stays_exact():
    """Grade-school problems are chains, so per-operation exactness is not enough."""
    R = RationalAlgebra()
    val = R.encode([Fraction(48)])
    val = R.div(val, R.encode([Fraction(2)]))          # half as many
    val = R.add(val, R.encode([Fraction(325, 100)]))   # plus $3.25
    val = R.mul(val, R.encode([Fraction(3)]))
    assert R.decode(val)[0] == (Fraction(48, 2) + Fraction(325, 100)) * 3


def test_denominator_growth_is_observable():
    """Fractions cannot be reduced in residue form, so denominators only grow.
    Watching that is how a chain approaching the ring is noticed, rather than
    discovered from a silently wrong answer."""
    R = RationalAlgebra()
    v = R.encode([Fraction(1)])
    seen = []
    for _ in range(4):
        v = R.div(v, R.encode([Fraction(100)]))
        seen.append(R.denominator_magnitude(v)[0])
    assert seen == [100, 10_000, 1_000_000, 100_000_000]
    assert all(d < R.headroom() for d in seen)


def test_a_zero_denominator_raises_rather_than_returning_a_number():
    """Division by zero is undetectable in the ring -- a zero denominator is a legal
    residue vector. It must fail at decode instead of producing a confident wrong
    answer; guarding it is the program's job."""
    R = RationalAlgebra()
    bad = R.div(R.encode([Fraction(5)]), R.encode([Fraction(0)]))
    with pytest.raises(ZeroDivisionError):
        R.decode(bad)


def test_rational_operations_are_differentiable():
    """Division is multiplication with the operands swapped, so it stays in the
    differentiable part of the ring -- no modular inverse, no decode in the loop."""
    R = RationalAlgebra()
    a = R.encode([Fraction(3)])
    num = a[0].clone().requires_grad_(True)
    out = R.div((num, a[1]), R.encode([Fraction(2)]))
    out[0].sum().backward()
    assert num.grad is not None
