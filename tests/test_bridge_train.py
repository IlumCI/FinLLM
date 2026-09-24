"""GSM8K through the bridge: the parts that are exact, and the coverage that caps
everything else.

Almost none of this module is learned, so almost all of these tests are about
whether the *unlearned* parts are right -- which is where a silent error does real
damage. In particular: a program recovered from someone else's annotations is a
hypothesis, and if it does not execute to that dataset's own answer, training on it
teaches the wrong program perfectly.
"""

from __future__ import annotations

import os
from fractions import Fraction

import pytest
import torch

from lamb.algebra import ResidueSystem
from lamb.bridge import DEFAULT_CONSTANTS
from lamb.bridge_train import (Example, align_program, build_examples,
                               parse_annotation_steps, parse_final_answer,
                               verify_alignment, _frac)
from lamb.rational import RATIONAL_MODULI
from lamb.regmachine import RATIONAL_OPS, RegisterMachine


def test_literals_become_exact_rationals_not_floats():
    """A rounded operand is a wrong operand in a ring, and a float operand is a
    rounded one. ``3.25`` is ``325/100``, decided by the parser and not by IEEE."""
    assert _frac("3.25") == Fraction(13, 4)
    assert _frac("1,250") == Fraction(1250)
    assert _frac("-7") == Fraction(-7)
    assert _frac("0.1") == Fraction(1, 10)          # not 0.1000000000000000055...
    assert _frac("20%") == Fraction(20)             # the % is the program's problem
    assert _frac("") is None and _frac("abc") is None


def test_the_final_answer_line_parses():
    sol = "She sold 48/2 = <<48/2=24>> clips in May.\n#### 72"
    assert parse_final_answer(sol) == Fraction(72)
    assert parse_final_answer("no marker here") is None


def test_annotations_give_a_program_and_the_rejects_are_counted():
    """GSM8K's ``<<a op b=c>>`` is very nearly the emitted program -- an operation,
    its operands and its result, in evaluation order. That is the premise the
    roadmap said a natural-language problem could not supply, and for this dataset
    it is simply wrong.

    A compound left side is *decomposed* rather than rejected: ``<<48*3+5=149>>``
    contributes two instructions. It used to be thrown away, and that was the single
    largest cause of alignment failure -- 52.1% of the 39% that failed -- because the
    result of a discarded step is never written to a register, so every later step
    that reads it fails too. Decomposing lifted coverage 0.420 -> 0.614.
    """
    steps, rejected = parse_annotation_steps(
        "<<48/2=24>> then <<48+24=72>> and <<48*3+5=149>>")
    assert [(a, op, b, c) for a, op, b, c in steps] == [
        (Fraction(48), "/", Fraction(2), Fraction(24)),
        (Fraction(48), "+", Fraction(24), Fraction(72)),
        (Fraction(48), "*", Fraction(3), Fraction(144)),
        (Fraction(144), "+", Fraction(5), Fraction(149)),
    ]
    assert rejected == 0


def test_compound_annotations_respect_precedence_and_associativity():
    """The decomposition is a recursive-descent parser, not a regex, because the point
    is the *structure*. ``5-2*2`` must emit the multiplication first.

    The lexer originally allowed a signed literal, which made it swallow a *binary*
    minus: ``5-2*2`` tokenised to ``["5", "-2", "*", "2"]``, failed to parse, and
    rejected a perfectly good annotation. It survived one test round because the
    obvious cases -- ``48*3+5``, ``100*0.2`` -- contain no subtraction.
    """
    def steps_of(s):
        st, rej = parse_annotation_steps(s)
        assert rej == 0, s
        return [(str(a), op, str(b), str(c)) for a, op, b, c in st]

    assert steps_of("<<5-2*2=1>>") == [("2", "*", "2", "4"), ("5", "-", "4", "1")]
    assert steps_of("<<(2+3)*4=20>>") == [("2", "+", "3", "5"), ("5", "*", "4", "20")]
    assert steps_of("<<20-5-3=12>>") == [("20", "-", "5", "15"), ("15", "-", "3", "12")]
    assert steps_of("<<100/4/5=5>>") == [("100", "/", "4", "25"), ("25", "/", "5", "5")]
    assert steps_of("<<-5+8=3>>") == [("-5", "+", "8", "3")]
    # a decimal stays exact through the chain: 0.2 is 1/5, never a float
    assert steps_of("<<100*0.2=20>>") == [("100", "*", "1/5", "20")]


def test_an_annotation_that_does_not_compute_its_own_result_is_dropped():
    """GSM8K rounds in places. An annotation that disagrees with itself cannot
    supervise anything, and keeping it would train the wrong arithmetic."""
    steps, rejected = parse_annotation_steps("<<10/3=3>>")
    assert steps == [] and rejected == 1


def test_alignment_turns_values_into_addresses():
    """An annotation names operands by value; an instruction names them by address.
    Alignment is that lookup, and it fails when a number the annotation uses is not
    in the register file -- a fact about extraction, not about the model."""
    regs = [Fraction(1), Fraction(2), Fraction(48), Fraction(0)]
    steps = [(Fraction(48), "/", Fraction(2), Fraction(24))]
    prog = align_program(regs, steps, n_operands=4, n_instr=3)
    assert prog is not None
    assert prog[0] == (RATIONAL_OPS.index("/"), 2, 1)
    # padded to the fixed shape by multiplying through the constant 1, so the answer
    # still lands in the last register
    assert len(prog) == 3
    assert all(RATIONAL_OPS[i[0]] == "*" and i[2] == 0 for i in prog[1:])

    missing = [(Fraction(99), "+", Fraction(1), Fraction(100))]
    assert align_program(regs, missing, n_operands=4, n_instr=3) is None
    assert align_program(regs, [], n_operands=4, n_instr=3) is None


def test_alignment_will_not_point_at_a_padding_slot():
    """Padding is zero, and zero is a value an annotation can name.

    The file is padded to a fixed width and those slots are masked out of the pointer
    distribution, so resolving an operand of ``0`` to a padding index emits a program
    that points where it is not allowed to read: unrepresentable at execution, and
    invisible to a value-matching search because the value genuinely matches. The
    search therefore runs over *readable* slots, and the readable set is not a
    prefix -- the row's operands, then the results written so far, with the padding in
    between excluded.
    """
    # slots 0..2 are real (1, 2, 48); slot 3 is padding that happens to hold 0
    regs = [Fraction(1), Fraction(2), Fraction(48), Fraction(0)]
    steps = [(Fraction(48), "-", Fraction(0), Fraction(48))]
    assert align_program(regs, steps, 4, 3, count=4) is not None   # 0 is real here
    assert align_program(regs, steps, 4, 3, count=3) is None       # 0 is padding


def test_a_recovered_program_actually_executes_to_the_stated_answer():
    """The question to ask before a single gradient step, and it involves no model.

    ``x * 1`` is the padding, and it is an exact identity in this ring rather than
    an approximate one: it multiplies numerator and denominator by one, so it does
    not even grow the denominator.
    """
    machine = RegisterMachine(8, n_operands=4, n_instr=3,
                              system=ResidueSystem(RATIONAL_MODULI), rational=True)
    ex = Example(text="", answer=Fraction(24),
                 values=[1, 2, 48, 0], scales=[0, 0, 0, 0], count=3,
                 program=align_program([Fraction(1), Fraction(2), Fraction(48),
                                        Fraction(0)],
                                       [(Fraction(48), "/", Fraction(2),
                                         Fraction(24))], 4, 3))
    assert verify_alignment([ex], machine) == 1.0


def test_a_decimal_survives_the_whole_path():
    """The mixed-scale bug, checked where it would actually bite: ``3.25`` reaches
    the registers as ``(325, scale=2)`` and comes back out of the algebra as
    ``13/4``, not as ``325`` and not as a float."""
    machine = RegisterMachine(8, n_operands=4, n_instr=2,
                              system=ResidueSystem(RATIONAL_MODULI), rational=True)
    regs = [Fraction(1), Fraction(2), Fraction(13, 4), Fraction(7)]
    steps = [(Fraction(13, 4), "+", Fraction(7), Fraction(41, 4))]
    ex = Example(text="", answer=Fraction(41, 4),
                 values=[1, 2, 325, 7], scales=[0, 0, 2, 0], count=4,
                 program=align_program(regs, steps, 4, 2))
    assert ex.fractions == regs          # value+scale round-trips to the rational
    assert verify_alignment([ex], machine) == 1.0


def test_build_examples_reports_the_ceiling_rather_than_hiding_it():
    """Three coverage rates, and every one of them caps what any model on top could
    reach. ``operand_miss`` is extraction recall: if the numbers are not in the
    register file, no program over it can be right."""
    rows = [
        {"question": "Natalia sold 48 clips, then half as many.",
         "answer": "She sold <<48/2=24>> in May.\n#### 24"},
        {"question": "A bag costs $3.25 and he buys 7.",
         "answer": "That is <<3.25*7=22.75>> dollars.\n#### 22.75"},
        {"question": "No numbers at all here.",
         "answer": "Nothing.\n#### 0"},
    ]
    ex, st = build_examples(rows, n_operands=20, n_instr=4, constants=DEFAULT_CONSTANTS)
    assert int(st["n"]) == 3
    assert st["no_answer"] == 0.0
    # "half as many" is served by the constant register 2, not by the parser
    # inventing a 0.5 -- so 48/2 aligns even though "2" is nowhere in the text.
    assert ex[0].program is not None
    assert 0.0 <= st["operand_miss"] <= 1.0
    assert 0.0 <= st["program_coverage"] <= 1.0


def test_padding_needs_a_real_identity_register():
    """The pad instruction is ``x * 1``, so slot 0 has to hold one. With no constants
    it holds the first quantity instead, and padding by multiplying through whatever
    that happens to be is precisely the silently wrong program the register machine
    exists to make unrepresentable."""
    regs = [Fraction(48), Fraction(2), Fraction(0), Fraction(0)]
    steps = [(Fraction(48), "/", Fraction(2), Fraction(24))]
    assert align_program(regs, steps, 4, 3, count=2) is None   # slot 0 is 48, not 1
    assert align_program(regs, steps, 4, 1, count=2) is not None  # no padding needed


def test_constants_are_what_make_half_as_many_expressible():
    """The sharpest case for preloading constants. "half as many as 48" needs a
    ``2`` that the text never writes down, and by this repo's own note it is the
    single most common operation in GSM8K. Without the constant the alignment fails
    on a problem the parser read perfectly."""
    rows = [{"question": "Natalia sold 48 clips, then half as many.",
             "answer": "<<48/2=24>>\n#### 24"}]
    with_c, _ = build_examples(rows, 20, 4, constants=DEFAULT_CONSTANTS)
    without, _ = build_examples(rows, 20, 4, constants=())
    assert with_c[0].program is not None
    assert without[0].program is None


def test_a_chain_that_does_not_end_on_the_answer_is_dropped():
    """The check that changed the pipeline, and the number that justified it.

    GSM8K leaves its final step unannotated, or writes arithmetic it does not itself
    compute, often enough that **a sixth of usable chains end somewhere other than the
    answer**.
    Keeping them meant 17.6% of "recovered" programs executed to the wrong number --
    which is worse than having no program at all: it is supervision toward a wrong
    answer, and a model would learn it perfectly.

    So the chain is truncated at the *last* step whose result is the answer, and a
    chain that never reaches it is dropped. Coverage falls from 0.533 to 0.420 on the
    train split and the execution rate goes 0.824 -> 1.000. That is the trade, and it
    is the right way round: less supervision that is correct beats more that is not.
    """
    rows = [
        # ends on the answer: kept
        {"question": "48 clips, then half as many.",
         "answer": "<<48/2=24>> then <<48+24=72>>\n#### 72"},
        # the final step is inexact, so it is dropped and the chain never reaches 72
        {"question": "48 clips, then half as many.",
         "answer": "<<48/2=24>> then <<70/3=72>>\n#### 72"},
        # the answer appears mid-chain; truncation recovers it
        {"question": "48 clips, then half as many.",
         "answer": "<<48/2=24>> and <<24+1=25>>\n#### 24"},
    ]
    ex, st = build_examples(rows, n_operands=20, n_instr=4)
    assert ex[0].program is not None
    assert ex[1].program is None
    assert ex[2].program is not None
    assert st["answer_not_in_chain"] == pytest.approx(1 / 3)

    # and every program that survives executes to the stated answer -- by construction
    machine = RegisterMachine(8, n_operands=20, n_instr=4,
                              system=ResidueSystem(RATIONAL_MODULI), rational=True)
    assert verify_alignment(ex, machine) == 1.0


@pytest.mark.skipif(not os.environ.get("LAMB_BRIDGE_ENCODER_TEST"),
                    reason="downloads a model; set LAMB_BRIDGE_ENCODER_TEST=1 to run")
def test_encode_dataset_caches_token_states_not_a_pooled_vector():
    """Opt-in, because it downloads an encoder.

    The default suite stays offline on purpose: every other test here is about the
    *unlearned* parts being exact, and a test that can fail because a model host is
    down tells you nothing about that. Verified against transformers 5.17 --
    ``last_hidden_state`` and the padding call are what the cache format depends on,
    and both are what a major version could move.
    """
    from lamb.bridge import encode_dataset

    out = encode_dataset(["Natalia sold clips to 48 of her friends.", "Short one."],
                         max_len=32, batch_size=2)
    # token states, not one pooled vector -- the resampler cross-attends, so
    # collapsing the problem before it arrives throws away what it is there to read
    assert out["enc"].shape == (2, 32, 384)
    assert out["enc"].dtype is torch.float16          # the cache is on disk
    assert out["pad_mask"].shape == (2, 32)
    assert int(out["pad_mask"][1].sum()) > int(out["pad_mask"][0].sum())
    assert out["meta"]["truncated"] == 0


def test_the_bridge_says_what_is_missing_rather_than_failing_obscurely():
    """``transformers`` is an optional extra, so the failure has to name the fix."""
    import lamb.bridge as bridge

    src = bridge.encode_dataset.__doc__ or ""
    assert "truncated" in src          # the number a silent cut-off would hide
    import inspect
    body = inspect.getsource(bridge.encode_dataset)
    assert "--extra bridge" in body and "ImportError" in body


def test_constants_and_register_width_are_one_decision_not_two():
    """The sharp edge in the extraction config, and the naive version is harmful.

    Unit constants (60, 24, 7, ...) are worth +6.6 points of coverage because 59.6% of
    the remaining alignment failures need an integer the problem never writes -- "an
    hour ... 50 minutes" requires 60. But every constant occupies a register slot, and
    in a 12-wide file ten constants fill 97% of rows and push the problem's own
    quantities out: coverage *halves*, 0.614 -> 0.249.

    So the pair is tested together. Adding a constant without widening the file is a
    regression that looks like a feature.
    """
    from lamb.bridge import DEFAULT_CONSTANTS

    # The annotation deliberately uses the *last two* quantities. A narrow file keeps
    # only the first few, so the squeeze has to be visible in alignment rather than
    # merely in the counts -- an earlier version of this test used the first two
    # quantities, which survive the squeeze, and passed under both widths while
    # demonstrating nothing.
    rows = [{"question": f"He had 41 apples, 43 pears, {i} plums and {i + 1} figs.",
             "answer": f"<<{i}+{i + 1}={2 * i + 1}>>\n#### {2 * i + 1}"}
            for i in range(200, 229)]
    narrow, _ = build_examples(rows, 12, 4, DEFAULT_CONSTANTS)
    wide, _ = build_examples(rows, 20, 4, DEFAULT_CONSTANTS)
    # ten constants leave only two quantity slots at width 12, so most of the problem's
    # own numbers are pushed out of the file entirely; at 20 they all fit
    assert all(e.count == 12 for e in narrow)    # saturated: 10 constants + 2 quantities
    assert all(e.count == 14 for e in wide)      # 10 constants + 4 quantities
    # the quantities the annotation needs were pushed out of the narrow file entirely
    assert not any(e.program for e in narrow)
    assert all(e.program for e in wide)

    # and the file must refuse rather than silently drop a constant
    import pytest as _pytest
    with _pytest.raises(ValueError):
        build_examples(rows, 8, 4, DEFAULT_CONSTANTS)
