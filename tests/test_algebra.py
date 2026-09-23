"""The residue algebra: exactness, extrapolation, and the two easy mistakes.

Everything here is checked **without a model**. The point of moving arithmetic into
the representation is that composition stops being a learned approximation, so its
correctness should be provable by inspection rather than measured -- and if it is
not exact here, no amount of training will rescue it.
"""

from __future__ import annotations

import random

import torch

from lamb.algebra import DEFAULT_MODULI, ResidueAlgebra, ResidueSystem, order10


def test_round_trip_is_exact_over_the_whole_signed_range():
    s = ResidueSystem()
    lo, hi = s.bounds()
    rng = random.Random(0)
    vals = [0, 1, -1, lo, hi] + [rng.randint(lo, hi) for _ in range(5000)]
    assert all(s.crt(s.residues(v)) == v for v in vals)


def test_composition_is_exact_far_beyond_any_training_magnitude():
    """The structural claim: the algebra does not degrade with magnitude.

    A model trained on 1-digit operands has still never seen a 5-digit number --
    but composition is not a thing the model does, so it is exact anyway.
    """
    s = ResidueSystem()
    alg = ResidueAlgebra(s)
    rng = random.Random(1)
    for digits in (1, 2, 3, 5):
        lo, hi = (0, 9) if digits == 1 else (10 ** (digits - 1), 10 ** digits - 1)
        for _ in range(200):
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            for op, want in (("+", a + b), ("-", a - b), ("*", a * b)):
                if s.representable(want):
                    assert alg.compose_exact(a, b, op) == want, (a, op, b)


def test_subtraction_is_not_reversed():
    """Regression. Indexing the cross-correlation the wrong way round computes
    ``y - x``, which equals ``x - y`` exactly when the two are equal -- so it passes
    about a tenth of random cases and reads as a near miss rather than a sign error.
    """
    s = ResidueSystem()
    alg = ResidueAlgebra(s)
    for a, b in ((10, 3), (3, 10), (-5, 7), (100, 1)):
        assert alg.compose_exact(a, b, "-") == a - b
    # the asymmetric cases are the ones a reversed index gets wrong
    assert alg.compose_exact(10, 3, "-") != alg.compose_exact(3, 10, "-")


def test_out_of_range_wraps_rather_than_saturating():
    """A value past the ring is a *different number*, not a large one. Training on
    a wrapped target teaches arithmetic that is wrong, so callers must mask on
    ``representable`` rather than clip -- this pins why."""
    s = ResidueSystem()
    lo, hi = s.bounds()
    assert not s.representable(hi + 1)
    assert s.crt(s.residues(hi + 1)) != hi + 1


def test_default_moduli_have_short_digit_coefficient_periods():
    """The design constraint that is easy to miss.

    ``n mod p`` from digits is ``sum_i digit_i * (10^i mod p)``: the coefficients
    repeat every ``ord_p(10)`` positions, and that period is how many digit
    positions a run must *see* before the modulus is learnable at all. Primes like
    17/19/23 look reasonable and have periods of 16/18/22, which makes them nearly
    useless at the widths training reaches.
    """
    assert max(order10(p) for p in DEFAULT_MODULI) <= 6
    assert order10(9) == 1 and order10(11) == 2      # digit sum, alternating sum
    assert order10(17) == 16 and order10(23) == 22   # the trap


def test_moduli_must_be_coprime():
    import pytest

    with pytest.raises(ValueError):
        ResidueSystem((4, 6))


def test_composition_is_differentiable_into_the_coder():
    s = ResidueSystem()
    alg = ResidueAlgebra(s)
    lg = torch.randn(4, s.n_units, requires_grad=True)
    out = alg.compose(lg, s.onehot([5, 5, 5, 5]), "+")
    tgt = s.targets([12, 13, 14, 15])
    loss = sum(torch.nn.functional.nll_loss(torch.log(o.clamp_min(1e-9)), tgt[:, k])
               for k, o in enumerate(out))
    loss.backward()
    assert float(lg.grad.norm()) > 0.0


def test_summing_a_composed_distribution_is_not_a_loss():
    """Both inputs are normalised, so the composition is too and its sum is 1
    whatever the logits. A gradient check against that sum reports zero and looks
    like a broken graph; it is a broken *test*."""
    s = ResidueSystem()
    alg = ResidueAlgebra(s)
    lg = torch.randn(2, s.n_units, requires_grad=True)
    out = torch.cat(alg.compose(lg, s.onehot([3, 3]), "+"), dim=-1)
    assert abs(float(out.detach().sum()) - 2 * len(s.moduli)) < 1e-4
