"""The latent ALU: slot addressing, leaf-up composition, and depth.

The load-bearing properties are that a latent slot means the same thing the
grammar's trace means, that composition climbs from the leaves (which is the whole
reason depth is supposed to be free), and that the answer the algebra produces is
exact given correct leaves -- at depths no training run would reach.
"""

from __future__ import annotations

import pytest
import torch

from lamb import ArithmeticTokenizer, LotusConfig
from lamb.algebra import ResidueSystem
from lamb.alu import (LatentALU, is_leaf, parse_expr, slot_order, slot_pairs)
from lamb.config import ModelConfig
from lamb.lotus import LotusTrainer
from lamb.selfplay.grammar import Descriptor, TaskGrammar


def _value(t):
    op, l, r = t
    a = l if isinstance(l, int) else _value(l)
    b = r if isinstance(r, int) else _value(r)
    return {"+": a + b, "-": a - b, "*": a * b}[op]


def _render(t):
    if is_leaf(t):
        op, a, b = t
        return f"{a}{op}{b}"
    op, l, r = t
    return f"({_render(l)}){op}({_render(r)})"


def test_parse_round_trips_and_slots_match_the_grammar_trace():
    """Slot ``i`` must be trace entry ``i`` -- otherwise every target is misaligned
    and the model is being taught the wrong value for the right position."""
    g = TaskGrammar()
    for depth in (1, 2, 3):
        for ops_key in (0, 1):
            for seed in range(40):
                expr, ans, trace = g.sample_with_trace(Descriptor(depth, 1, ops_key), seed)
                t = parse_expr(expr)
                if depth > 1:
                    assert _render(t) == expr
                assert [_value(s) for s in slot_order(t)] == trace
                assert _value(t) == int(ans)


def test_repeated_subexpressions_get_distinct_slots():
    """Regression. ``(1+2)-(1+2)`` has two equal-but-distinct children; a
    value-keyed lookup hands both the same slot and silently reads one latent
    twice. It only bites on problems with a repeated sub-expression, so a smoke
    test sails past it."""
    t = parse_expr("(1+2)-(1+2)")
    pairs = slot_pairs(t)
    assert [i for _, i in pairs] == [0, 1]
    _, l, r = t
    assert l == r and l is not r          # equal by value, distinct objects
    assert [s for s, _ in pairs][0] is l and [s for s, _ in pairs][1] is r


def test_slot_indices_are_a_contiguous_post_order_range():
    for expr in ("(6+6)-(4-8)", "((1+2)-(3+4))*((5-6)+(7+8))"):
        idx = [i for _, i in slot_pairs(parse_expr(expr))]
        assert idx == sorted(idx) == list(range(len(idx)))


def test_composition_from_oracle_leaves_is_exact_at_untrained_depths():
    """Depth is free *for the algebra*: given correct leaves, a depth-6 expression
    (64 operands) composes exactly, with nothing learned about depth."""
    alu = LatentALU(d_model=8, system=ResidueSystem())
    g = TaskGrammar()
    for depth in (2, 3, 4, 5, 6):
        checked = 0
        for seed in range(25):
            expr, ans, _ = g.sample_with_trace(Descriptor(depth, 1, 0), seed)
            if not alu.sys.representable(int(ans)):
                continue
            t = parse_expr(expr)
            pairs = slot_pairs(t)
            codes = torch.zeros(len(pairs), alu.sys.n_units)
            for sub, idx in pairs:
                if is_leaf(sub):
                    codes[idx] = alu.sys.onehot([_value(sub)])[0] * 30.0
            blocks = alu._from_leaves(codes, t, pairs)
            assert alu.sys.crt([int(b.argmax(-1)) for b in blocks]) == int(ans)
            checked += 1
        assert checked > 0


def _trainer(**kw):
    torch.manual_seed(0)
    base = dict(steps=4, batch_size=16, n_latent=8, loops=3, depth=2, eval_tasks=16,
                trace_coef=0.0, alu_coef=1.0, alu_consistency_coef=0.0, device="cpu")
    base.update(kw)
    return LotusTrainer(LotusConfig(**base), ArithmeticTokenizer(),
                        ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3))


def test_root_slot_is_depth_independent():
    """The root goes in the *last* slot, not slot T. T moves with depth (2, 6, 14),
    so indexing the answer by it would put it somewhere different at every depth and
    silently break any transfer between depths."""
    for depth, n_latent in ((2, 8), (3, 16)):
        tr = _trainer(depth=depth, n_latent=n_latent)
        assert tr.root_slot == n_latent - 1
        assert tr.n_inter == 2 ** depth - 2


def test_alu_and_token_trace_cannot_both_claim_the_slots():
    """One value per slot and one digit token per slot are different meanings for
    the same hidden state; running both puts contradictory targets on it."""
    with pytest.raises(ValueError):
        _trainer(trace_coef=0.5, alu_coef=1.0)


def test_alu_supervision_trains_and_reaches_the_head():
    tr = _trainer(steps=30, batch_size=32)
    first = tr._train_step(0)
    last = first
    for s in range(1, 30):
        last = tr._train_step(s)
    assert last["alu"] < first["alu"]
    assert float(tr.reasoner.alu.head.weight.grad.norm()) > 0.0


def test_evaluation_reports_both_answers_and_splits_the_diagnostic():
    """Leaf and root residue accuracy come apart hard -- the algebra composes from
    leaves, so the model has no reason to learn the root -- and their average is a
    number about nothing."""
    tr = _trainer(n_latent=8)
    r = tr.algebraic_accuracy(32)
    for k in ("algebraic", "readout", "leaf_residue", "root_residue", "in_range"):
        assert k in r and 0.0 <= r[k] <= 1.0


def test_out_of_range_rows_are_masked_not_clipped():
    tr = _trainer(depth=2, digits=6, n_latent=8, alu_moduli=(5, 7))   # tiny ring
    tasks = tr._sample_batch(24)
    _, mask, _, keep, _raw = tr._alu_batch(tasks)
    assert float(keep.float().mean()) < 1.0       # some rows leave the ring
    assert float(mask[~keep].sum()) == 0.0        # and are supervised on nothing


def test_scalar_control_composes_and_trains():
    """The control for "why not just regress the value?".

    It has to actually work as a baseline, or the comparison is rigged: exact
    composition from oracle scalars, and a loss that falls.
    """
    from lamb.alu import ScalarALU

    m = ScalarALU(d_model=8)
    t = parse_expr("(6+6)-(4-8)")
    vals = torch.tensor([[12.0, -4.0, 0.0, 0.0]])
    assert float(m.compose_tree(vals, [t])[0]) == 16.0

    tr = _trainer(steps=20, batch_size=32, alu_mode="scalar")
    first = tr._train_step(0)
    last = first
    for s in range(1, 20):
        last = tr._train_step(s)
    assert last["alu"] < first["alu"]
    r = tr.algebraic_accuracy(32)
    assert "mean_abs_value_error" in r      # the scalar arm's own diagnostic
