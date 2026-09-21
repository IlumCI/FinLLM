"""POET population: neighbours/complexity, reproduction, transfer, capacity."""

from __future__ import annotations

import torch

from lamb import ArithmeticTokenizer, POETConfig
from lamb.poet import POETTrainer, complexity, neighbours
from lamb.selfplay.grammar import Descriptor


def test_neighbours_and_complexity():
    nb = neighbours(Descriptor(1, 1, 0), max_depth=3, max_digits=3, n_ops=2)
    assert {x.label() for x in nb} == {"d2g1o0", "d1g2o0", "d1g1o1"}
    assert complexity(Descriptor(2, 1, 0)) > complexity(Descriptor(1, 1, 0))
    # respects caps
    assert neighbours(Descriptor(3, 3, 1), max_depth=3, max_digits=3, n_ops=2) == []


def _cfg(**kw):
    base = dict(agent_d_model=48, agent_recurrent_steps=2, opt_steps=1, batch_size=16,
                eval_tasks=8, max_depth=3, max_digits=3)
    base.update(kw)
    return POETConfig(**base)


def test_poet_runs_and_respects_capacity():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    cfg = _cfg(iters=12, init_members=2, pop_capacity=4, transfer_every=3,
              reproduce_every=4, reproduce_threshold=0.0)
    trainer = POETTrainer(cfg, tok)
    assert len(trainer.population) == 2
    for _ in range(cfg.iters):
        trainer.iterate()
    assert len(trainer.population) <= cfg.pop_capacity
    assert trainer.attempted_frontier() >= complexity(Descriptor(1, 1, 0))


def test_poet_reproduction_spawns_children():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    cfg = _cfg(init_members=1, pop_capacity=8, mc_high=1.0, reproduce_threshold=0.5,
               behavioural_novelty=False)  # isolate descriptor reproduction here
    trainer = POETTrainer(cfg, tok)
    trainer.population[0].success = 1.0  # a competent parent
    before = len(trainer.population)
    trainer._reproduce()
    labels = {m.env.label() for m in trainer.population}
    assert len(trainer.population) > before
    assert {"d2g1o0", "d1g2o0", "d1g1o1"} <= labels  # all novel neighbours admitted


def test_poet_transfer_runs():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    trainer = POETTrainer(_cfg(init_members=2, pop_capacity=8), tok)
    trainer._transfer()  # must not raise
    assert trainer.transfers >= 0
