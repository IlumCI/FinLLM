"""The residue algebra: exactness, extrapolation, and the two easy mistakes.

Everything here is checked **without a model**. The point of moving arithmetic into
the representation is that composition stops being a learned approximation, so its
correctness should be provable by inspection rather than measured -- and if it is
not exact here, no amount of training will rescue it.
"""

from __future__ import annotations

import random

import pytest
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


def test_multiplication_tables_follow_the_data_not_the_constructor():
    """Regression: the first GPU run would have died on a device mismatch.

    The tables are plain tensors held by an ordinary object, not buffers on an
    nn.Module, so ``module.to("cuda")`` does not move them -- it moves parameters
    and registered buffers only. Building them once at construction leaves them
    behind when the activations move. They must be built per device, from the
    device of the data.
    """
    alg = ResidueAlgebra(ResidueSystem())
    assert alg._mul_cache == {}                       # nothing built eagerly
    a = ResidueSystem().onehot([7])
    alg.compose(a, a, "*", logits=False)
    assert torch.device("cpu") in alg._mul_cache      # built on the data's device
    tables = alg.mul_idx_for(torch.device("cpu"))
    assert alg.mul_idx_for(torch.device("cpu")) is tables   # and cached, not rebuilt


def test_half_precision_inputs_do_not_corrupt_the_result():
    """Under autocast the activations arrive in fp16. These are distributions over
    small rings: the tail underflows, the convolutions lose mass, and a
    renormalised-but-wrong distribution decodes to a *different integer* -- which in
    a residue system is not a near miss. The algebra therefore runs in fp32."""
    s = ResidueSystem()
    alg = ResidueAlgebra(s)
    for a, b, op in ((123, 456, "*"), (9999, 1234, "+"), (77, 9999, "-")):
        A, B = s.onehot([a]).half(), s.onehot([b]).half()
        want = {"+": a + b, "-": a - b, "*": a * b}[op]
        assert s.decode(alg.compose(A, B, op, logits=False)) == [want]


def _rrns(n_redundant=3):
    from lamb.algebra import RedundantResidueSystem
    return RedundantResidueSystem((16, 25, 27, 11) + (37, 7, 41, 101)[:n_redundant],
                                  n_core=4)


def test_detection_catches_every_single_residue_error():
    """CRT has no locality, so a corrupted residue lands essentially uniformly over
    the ring and therefore outside the legitimate range."""
    s = _rrns()
    rng = random.Random(0)
    for _ in range(2000):
        v = rng.randint(-s.legit, s.legit)
        res = s.residues(v)
        assert not s.detect(res)                      # clean values never flagged
        k = rng.randrange(len(s.moduli))
        res[k] = (res[k] + rng.randint(1, s.moduli[k] - 1)) % s.moduli[k]
        assert s.detect(res)


def test_single_errors_are_corrected_exactly_with_enough_redundancy():
    """Dropping each modulus in turn identifies the culprit, because the survivors
    still over-determine the value. This takes answer accuracy from q**K to
    q**K + K*q**(K-1)*(1-q) -- at the measured q=0.985 and K=7, 0.900 to 0.996."""
    s = _rrns(3)
    rng = random.Random(1)
    for _ in range(1500):
        v = rng.randint(-s.legit, s.legit)
        res = s.residues(v)
        k = rng.randrange(len(s.moduli))
        res[k] = (res[k] + rng.randint(1, s.moduli[k] - 1)) % s.moduli[k]
        val, faulty = s.correct(res)
        assert val == v and faulty == k


def test_the_failure_mode_is_refusal_not_a_wrong_answer():
    """Under-provisioned redundancy loses corrections but must never invent one: a
    confidently wrong number is worse than an admitted failure, and this whole
    representation is being used precisely where no verifier can catch it."""
    s = _rrns(2)                                      # deliberately too few
    rng = random.Random(2)
    refused = corrected = 0
    for _ in range(1500):
        v = rng.randint(-s.legit, s.legit)
        res = s.residues(v)
        k = rng.randrange(len(s.moduli))
        res[k] = (res[k] + rng.randint(1, s.moduli[k] - 1)) % s.moduli[k]
        val, _ = s.correct(res)
        if val is None:
            refused += 1
        else:
            assert val == v                           # never a wrong value
            corrected += 1
    assert refused > 0 and corrected > 0              # it does lose some


def test_clean_residues_are_returned_untouched():
    s = _rrns()
    for v in (0, 1, -1, 12345, -59_400):
        val, faulty = s.correct(s.residues(v))
        assert val == v and faulty is None


def test_n_core_must_leave_redundancy():
    from lamb.algebra import RedundantResidueSystem

    with pytest.raises(ValueError):
        RedundantResidueSystem((16, 25, 27), n_core=3)
