"""The language peripheral: exact number extraction, and a length-invariant bridge.

The point of this module is that almost none of it is learned. The numbers come out
of the text by rule, the encoder is frozen and cached, and only the resampler and
the program heads carry gradients -- so these tests are mostly about the *exactness*
of the parts that are not trained, which is where a silent error would do real
damage.
"""

from __future__ import annotations

import torch

from lamb.bridge import (LanguageFront, Quantity, Resampler, extract_quantities,
                         registers_from_quantities)


def test_integers_come_out_exactly():
    q = extract_quantities("Natalia sold clips to 48 of her friends, then 24 more.")
    assert [x.value for x in q] == [48, 24]
    assert all(x.scale == 0 for x in q)


def test_decimals_are_carried_as_scaled_integers_not_floats():
    """The algebra is a ring of integers; a rounded operand is a wrong operand, and
    a float operand is a rounded one. 3.25 is (325, scale=2)."""
    q = extract_quantities("It costs $3.25 and weighs 0.5 kg.")
    assert (q[0].value, q[0].scale) == (325, 2)
    assert (q[1].value, q[1].scale) == (5, 1)
    assert abs(q[0].approx - 3.25) < 1e-9


def test_thousands_separators_and_signs():
    q = extract_quantities("The debt was -1,250 dollars, revenue 12,000,500.")
    assert [x.value for x in q] == [-1250, 12000500]


def test_percent_is_flagged_rather_than_silently_converted():
    """20% is not the number 20 and not the number 0.2 until the program says which.
    Flagging it leaves that decision where it belongs."""
    q = extract_quantities("A 20% discount on 50 items.")
    assert q[0].percent is True and q[0].value == 20
    assert q[1].percent is False


def test_excess_decimal_places_truncate_rather_than_round():
    q = extract_quantities("pi is about 3.14159")
    assert (q[0].value, q[0].scale) == (314, 2)


def test_no_numbers_is_not_an_error():
    assert extract_quantities("How many apples does she have left?") == []


def test_positions_are_kept_so_a_quantity_can_be_traced_to_its_span():
    text = "She had 48 clips."
    q = extract_quantities(text)[0]
    assert text[q.start:q.end] == "48"


def test_register_file_pads_and_reports_how_many_were_real():
    """A program pointing at a slot no number went into is not a worse program, so
    the count has to travel with the values."""
    qs = [extract_quantities(t) for t in ("48 and 24", "just 7", "nothing here")]
    vals, scales, counts = registers_from_quantities(qs, 6)
    assert counts == [2, 1, 0]
    assert vals[0][:2] == [48, 24] and vals[0][2:] == [0, 0, 0, 0]
    assert all(len(v) == 6 for v in vals)
    assert all(len(s) == 6 for s in scales)


def test_the_register_file_carries_the_scale_because_dropping_it_was_the_bug():
    """``3.25`` and ``7`` in the same problem used to enter the registers as ``325``
    and ``7``, so adding them gave ``332``. The parser tracked scale and the loader
    threw it away one line later -- the mixed-scale failure :mod:`lamb.rational` was
    written to kill, reintroduced at the one seam nothing downstream could catch.

    The fix is not arithmetic, it is plumbing: a scale is a denominator, and
    ``encode_scaled`` is where the two lists meet.
    """
    from fractions import Fraction

    from lamb.rational import RationalAlgebra

    qs = [extract_quantities("It costs $3.25 and he has $7.")]
    vals, scales, counts = registers_from_quantities(qs, 4)
    assert counts == [2]
    assert vals[0][:2] == [325, 7]
    assert scales[0][:2] == [2, 0]          # the half that used to be discarded

    R = RationalAlgebra()
    r = R.encode_scaled(vals[0], scales[0])
    got = R.decode(r)
    assert got[:2] == [Fraction(13, 4), Fraction(7)]
    # and the sum is 10.25, not 332
    assert R.decode(R.add((r[0][:1], r[1][:1]), (r[0][1:2], r[1][1:2])))[0] \
        == Fraction(41, 4)


def test_percent_reaches_the_register_file_as_a_flag_not_a_conversion():
    """``20%`` is neither 20 nor 0.2 until the program says which. With ``1`` and
    ``100`` available as constants the choice is expressible as a program, so the
    loader's job is to carry the flag, not to make the decision."""
    from lamb.bridge import quantity_percent_flags

    qs = [extract_quantities("A 20% discount on 50 items.")]
    vals, _, _ = registers_from_quantities(qs, 4)
    flags = quantity_percent_flags(qs, 4)
    assert vals[0][:2] == [20, 50]
    assert flags[0][:2] == [True, False]
    assert flags[0][2:] == [False, False]


def test_constants_occupy_the_same_slots_in_every_row():
    """"Half as many", "twice", "a third", "20% off" each name an operation whose
    other operand is never written down, so the parser cannot find it. Preloading it
    keeps the decision in the program -- ``x / 2``, not a parser deciding that
    "half" means 0.5 -- and a constant is only useful if its *address* is stable, so
    it goes in the low slots and the flags have to move with it."""
    from lamb.bridge import DEFAULT_CONSTANTS, quantity_percent_flags

    qs = [extract_quantities("A 20% discount on 50 items."),
          extract_quantities("nothing here")]
    n = len(DEFAULT_CONSTANTS)
    vals, scales, counts = registers_from_quantities(qs, 16, DEFAULT_CONSTANTS)
    assert vals[0][:n] == list(DEFAULT_CONSTANTS) == vals[1][:n]
    assert vals[0][n:n + 2] == [20, 50]
    assert scales[0][:n] == [0] * n
    assert counts == [n + 2, n]      # constants count as real operands
    flags = quantity_percent_flags(qs, 16, DEFAULT_CONSTANTS)
    assert flags[0][n] is True and not any(flags[0][:n])
    assert not any(flags[1])


def test_resampler_output_is_fixed_regardless_of_problem_length():
    """The whole reason for a resampler: everything downstream costs K, not T."""
    r = Resampler(d_enc=64, d_model=32, n_latents=16, n_heads=4, n_layers=2)
    for T in (8, 137, 1024):
        assert r(torch.randn(3, T, 64)).shape == (3, 16, 32)


def test_padding_is_actually_masked_out():
    """If padding leaked into the attention, a batch's results would depend on how
    its neighbours were padded -- which is the kind of bug that shows up as
    irreproducible numbers rather than as a crash."""
    torch.manual_seed(0)
    r = Resampler(d_enc=64, d_model=32, n_latents=8, n_heads=4, n_layers=1).eval()
    enc = torch.randn(2, 20, 64)
    pad = torch.zeros(2, 20, dtype=torch.bool)
    pad[:, 12:] = True
    with torch.no_grad():
        a = r(enc, pad)
        enc2 = enc.clone()
        enc2[:, 12:] = torch.randn(2, 8, 64) * 100      # garbage in the padded region
        b = r(enc2, pad)
    assert torch.allclose(a, b, atol=1e-5)


def test_gradients_reach_the_bridge_but_nothing_upstream_of_it():
    """The encoder is frozen and cached, so embeddings enter as data. The bridge is
    the only thing here that learns."""
    front = LanguageFront(d_enc=64, d_model=32, n_latents=8, n_heads=4, n_layers=1)
    enc = torch.randn(2, 16, 64)                     # cached, no grad
    out = front(enc)
    out.pow(2).mean().backward()
    assert float(front.resampler.queries.grad.norm()) > 0.0
    assert enc.grad is None


def test_written_numerals_are_found_and_ambiguous_words_are_not():
    """Worth 7.9 points of GSM8K program coverage, measured -- a problem whose
    quantities are partly spelled out has an incomplete register file, and no program
    over an incomplete file can be right.

    ``a``/``an`` are deliberately excluded. They mean one often enough to tempt and
    mean nothing often enough ("a discount", "an hour later") that admitting them would
    flood a fixed-width register file with ones and push real quantities out of it --
    which is the more expensive failure of the two.
    """
    from lamb.bridge import all_quantities, extract_lexical_quantities

    q = extract_lexical_quantities("She sold three dozen and seventeen more.")
    assert [x.value for x in q] == [3, 12, 17]
    assert all(x.scale == 0 for x in q)

    assert extract_lexical_quantities("a discount on an hour") == []
    # substrings must not match: "often" contains "ten", "hundreds" contains "hundred"
    assert extract_lexical_quantities("often hundreds of items") == []


def test_all_quantities_interleaves_digits_and_words_in_reading_order():
    """Register slots are filled first-N-in-text-order, so the order is load-bearing:
    a quantity that sorts to the back can be pushed out of a fixed-width file."""
    from lamb.bridge import all_quantities

    q = all_quantities("He had 12 apples, then three more, then 4.")
    assert [x.value for x in q] == [12, 3, 4]


def test_operation_words_are_left_to_the_program_not_the_parser():
    """"half" is an operation whose other operand is implicit, not a quantity. The
    constant register 2 plus a ``/`` expresses it, which keeps the decision where the
    percent flag already puts it. Measured: the constant `2` is worth 7.9 points of
    coverage, the same as the entire written-numeral table."""
    from lamb.bridge import all_quantities

    q = all_quantities("half as many as 48, and twice 6")
    assert [x.value for x in q] == [48, 6]
