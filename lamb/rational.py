"""Exact rational arithmetic in the residue ring -- division, and decimals.

:mod:`lamb.algebra` gives exact ``+``, ``-`` and ``*``. That is not enough for the
benchmarks this project is aiming at, and the two gaps are both fatal rather than
inconvenient:

* **Division is absent, and unavailable.** In a residue system you divide by
  multiplying with a modular inverse, which exists only for divisors coprime to
  every modulus, and gives the true quotient only when the division is exact.
  Worse, the moduli were chosen for short digit-periods -- powers of 2 and 5 and
  divisors of ``10^k - 1`` -- which is precisely the set that makes small divisors
  *non*-invertible. Under ``(2,5,9,11,7,13,37)`` not one divisor from 2 to 12 has an
  inverse, and "half as many" is the single most common operation in GSM8K.
* **Mixed scales are silently wrong.** ``extract_quantities`` returns ``3.25`` as
  ``(325, scale=2)`` and ``7`` as ``(7, scale=0)``; adding those in the ring gives
  ``332``, i.e. 3.32. A confidently wrong number is worse than a missing feature.

Both dissolve under one change: carry a value as a **pair** ``(numerator,
denominator)``, each an ordinary residue vector.

    a/b + c/d = (ad + cb) / bd          a/b * c/d = ac / bd
    a/b - c/d = (ad - cb) / bd          a/b / c/d = ad / bc

Division becomes multiplication with the operands swapped -- exact, closed, needing
no modular inverse, and differentiable because every step is still ``+``, ``-`` or
``*`` on distributions. Decimals stop needing scale bookkeeping entirely, because
``3.25`` *is* ``325/100``.

The cost is that a fraction cannot be reduced in residue form, so denominators grow
multiplicatively. That sets the ring size rather than the design: :data:`RATIONAL_MODULI`
gives ~9e15 with every digit-period still at most 6, which covers the worst chain a
grade-school problem produces (five two-decimal values, denominator 1e10) with five
orders of magnitude to spare. Growth is the thing to watch when the chain gets long,
and :meth:`RationalAlgebra.denominator_magnitude` is there to watch it with.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import torch

from .algebra import ResidueAlgebra, ResidueSystem

# Short digit-period (powers of 2 and 5, and divisors of 10^k - 1 for k <= 6) *and*
# a large product, which rationals need because denominators multiply.
RATIONAL_MODULI: Tuple[int, ...] = (64, 125, 27, 11, 7, 13, 37, 101, 41, 271)

# A value is (numerator, denominator), each packed (..., K, P).
Rat = Tuple[torch.Tensor, torch.Tensor]


class RationalAlgebra:
    """Exact rationals over a residue ring, differentiable throughout."""

    def __init__(self, system: Optional[ResidueSystem] = None, device: str = "cpu"):
        self.sys = system or ResidueSystem(RATIONAL_MODULI)
        self.alg = ResidueAlgebra(self.sys, device=device)

    # -- encoding ---------------------------------------------------------
    def encode(self, values: Sequence[Fraction], device: str = "cpu") -> Rat:
        """Exact rationals in, packed ``(N, K, P)`` numerator and denominator out."""
        nums = [int(v.numerator) for v in values]
        dens = [int(v.denominator) for v in values]
        return (self.alg.pack(self.sys.split(self.sys.onehot(nums, device))),
                self.alg.pack(self.sys.split(self.sys.onehot(dens, device))))

    def encode_scaled(self, values: Sequence[int], scales: Sequence[int],
                      device: str = "cpu") -> Rat:
        """From :class:`lamb.bridge.Quantity`: ``value`` at ``10**scale``.

        This is the join between the text parser and the algebra, and it is where
        the mixed-scale bug is fixed at the root: a scale is not a property the
        arithmetic has to track, it is just a denominator.
        """
        return self.encode([Fraction(int(v), 10 ** int(s))
                            for v, s in zip(values, scales)], device)

    # -- the four operations ----------------------------------------------
    def _mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.alg.compose_packed(x, y, "*")

    def add(self, a: Rat, b: Rat) -> Rat:
        (an, ad), (bn, bd) = a, b
        return (self.alg.compose_packed(self._mul(an, bd), self._mul(bn, ad), "+"),
                self._mul(ad, bd))

    def sub(self, a: Rat, b: Rat) -> Rat:
        (an, ad), (bn, bd) = a, b
        return (self.alg.compose_packed(self._mul(an, bd), self._mul(bn, ad), "-"),
                self._mul(ad, bd))

    def mul(self, a: Rat, b: Rat) -> Rat:
        return (self._mul(a[0], b[0]), self._mul(a[1], b[1]))

    def div(self, a: Rat, b: Rat) -> Rat:
        """``a / b``. Multiplication with the operands swapped -- no inverse needed.

        Division by zero is not detectable here: a zero denominator is a legal
        residue vector like any other, and decoding it raises rather than returning
        a wrong number. Guarding it is the program's job, not the ring's.
        """
        return (self._mul(a[0], b[1]), self._mul(a[1], b[0]))

    def compose(self, a: Rat, b: Rat, op: str) -> Rat:
        return {"+": self.add, "-": self.sub, "*": self.mul, "/": self.div}[op](a, b)

    # -- decoding ----------------------------------------------------------
    def decode(self, r: Rat) -> List[Fraction]:
        """CRT the numerator and denominator, then divide once, exactly."""
        num = self.sys.decode(self.alg.unpack(r[0]))
        den = self.sys.decode(self.alg.unpack(r[1]))
        return [Fraction(n, d) for n, d in zip(num, den)]

    def denominator_magnitude(self, r: Rat) -> List[int]:
        """The denominators as integers -- the quantity that limits chain length.

        Fractions cannot be reduced in residue form, so this only grows. Watching it
        is how you know a chain is approaching the ring rather than discovering it
        from a silently wrong answer.
        """
        return [abs(d) for d in self.sys.decode(self.alg.unpack(r[1]))]

    def headroom(self) -> int:
        return self.sys.M // 2
