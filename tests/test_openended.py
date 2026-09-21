"""Open-ended grammar, MCC admission, and the factored hypernetwork proposer."""

from __future__ import annotations

import random

import numpy as np

from lamb import ArithmeticTokenizer, ModelConfig, TrainConfig
from lamb._native import verify
from lamb.model.lamb import build_model
from lamb.selfplay.factoredhyper import FactoredHyperProposer
from lamb.selfplay.grammar import Descriptor, TaskGrammar
from lamb.selfplay.loop import SelfPlayTrainer
from lamb.selfplay.openended import OpenEndedCurriculum


def test_grammar_generates_verifiable_nested_tasks():
    g = TaskGrammar()
    for depth in (1, 2, 3):
        for seed in range(50):
            expr, ans = g.sample(Descriptor(depth, 2, 0), seed)
            assert verify(expr, ans), f"{expr} != {ans}"
            if depth > 1:
                assert "(" in expr  # nesting present


def test_grammar_mult_stays_exact():
    g = TaskGrammar()
    for seed in range(50):
        expr, ans = g.sample(Descriptor(2, 2, 1), seed)  # ops set includes '*'
        assert verify(expr, ans)


def _cfg():
    return TrainConfig(open_ended=True, start_digits=1, oe_max_depth=3, oe_max_digits=3)


def test_mcc_admission_unlocks_neighbours_of_mastered():
    cur = OpenEndedCurriculum(_cfg())
    assert len(cur.descriptors()) == 1                 # seed: d1g1o0
    added = cur.admit({0: 1.0})                         # master the seed
    assert added >= 1
    labels = {d.label() for d in cur.descriptors()}
    assert {"d2g1o0", "d1g2o0", "d1g1o1"} <= labels    # deeper, wider, richer-ops
    # An unmastered space admits nothing.
    cur2 = OpenEndedCurriculum(_cfg())
    assert cur2.admit({0: 0.1}) == 0


def test_complexity_increases_with_depth():
    cur = OpenEndedCurriculum(_cfg())
    cur.admit({0: 1.0})
    by_label = {cur.label(i): cur.complexity(i) for i in range(len(cur.descriptors()))}
    assert by_label["d2g1o0"] > by_label["d1g1o0"]


def test_factored_proposer_handles_growth():
    cur = OpenEndedCurriculum(_cfg())
    cur.admit({0: 1.0})  # a handful of descriptors
    prop = FactoredHyperProposer(cur, rng=random.Random(0))
    n = len(cur.descriptors())
    s = np.zeros(n)
    p = prop.probs(s)
    assert len(p) == n and abs(p.sum() - 1.0) < 1e-6 and p.min() > 0.0
    cells = prop.sample(24, s)
    assert all(0 <= c < n for c in cells)
    metrics = prop.update(s, cells, [0.5] * len(cells))
    assert "prop_pg" in metrics

    # Grow the space; the fixed-size factored net still produces a valid
    # distribution over the larger descriptor set (a table could not).
    cur.admit({i: 1.0 for i in range(n)})
    n2 = len(cur.descriptors())
    assert n2 > n
    p2 = prop.probs(np.zeros(n2))
    assert len(p2) == n2 and abs(p2.sum() - 1.0) < 1e-6


def test_open_ended_trainer_runs_with_factored_proposer():
    import torch

    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    tcfg = TrainConfig(
        batch_size=16, ops=("+", "-"), proposer_warmup=2, open_ended=True,
        proposer_kind="factored_hyper", admit_every=3, oe_max_depth=3, oe_max_digits=3,
        red_queen=True, league_snapshot_every=8,
    )
    trainer = SelfPlayTrainer(tcfg, build_model(ModelConfig(d_model=48, recurrent_steps=2), tok), tok)
    for _ in range(20):
        stats = trainer.train_step()
    assert np.isfinite(stats.loss)
    assert "n_tasks" in stats.extra
    assert trainer._n() >= 1
