"""Number-native tokenizer: round-trips, Abacus positions, no answer leakage."""

from __future__ import annotations

import pytest

from lamb import ArithmeticTokenizer


@pytest.mark.parametrize(
    "problem,answer",
    [("729+69", "798"), ("50-53", "-3"), ("12*11", "132"), ("0+0", "0"), ("9+9", "18")],
)
def test_answer_round_trip(problem, answer):
    tok = ArithmeticTokenizer()
    enc = tok.encode(problem, answer)
    decoded = tok.decode_answer(enc.ids[enc.ans_start :])
    assert decoded == answer


def test_no_value_leakage_on_answer():
    tok = ArithmeticTokenizer()
    enc = tok.encode("41+58", "99")
    # Every token at or after the answer start must have zero value / mask.
    for j in range(enc.ans_start, len(enc.ids)):
        assert enc.value[j] == 0.0
        assert enc.value_mask[j] == 0.0
    # Problem operand digits carry their magnitude.
    assert max(enc.value[: enc.ans_start]) == 58.0


def test_abacus_positions_reset_per_number():
    tok = ArithmeticTokenizer(reverse_digits=True)
    enc = tok.encode("123+45", "168")
    # Digit runs get 1,2,3,... and non-digits get 0.
    assert 0 in enc.abacus
    assert max(enc.abacus) == 3  # longest number here has 3 digits
    # BOS is non-digit -> 0.
    assert enc.abacus[0] == 0


def test_reverse_digits_is_lsb_first():
    tok = ArithmeticTokenizer(reverse_digits=True)
    enc = tok.encode("12+0", "12")
    # "12" -> reversed digits "2","1"; first emitted digit is the least significant.
    first_digit_id = enc.ids[1]
    assert tok.is_digit_id(first_digit_id)
    assert tok.digit_value(first_digit_id) == 2


def test_prompt_encoding_stops_at_equals():
    tok = ArithmeticTokenizer()
    enc = tok.encode_prompt("7+8")
    assert enc.ids[-1] == tok.EQ
    assert enc.ans_start == len(enc.ids)


def test_decode_handles_no_digits():
    tok = ArithmeticTokenizer()
    assert tok.decode_answer([tok.EOS]) is None
