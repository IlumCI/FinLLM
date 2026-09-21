"""RULER/BABILong-style suite: task validity, retrieval, and multi-hop tracking."""

from __future__ import annotations

import random

import torch

from lamb.ruler_bench import HopMemoryModel, RulerConfig, RulerTask, score, train


def test_ruler_task_is_well_posed():
    cfg = RulerConfig(n_literals=8, n_vars=16, d_model=32, d_mem=48, hops=3)
    task = RulerTask(cfg)
    batch, target = task.generate(16, 24, chain_len=2, n_distractor=3, rng=random.Random(0), front=True)
    assert (batch["kind"][:, -1] == 2).all()          # query at the end
    assert int((batch["kind"] == 1).sum()) > 0        # assignments present
    assert (target < cfg.n_literals).all()            # answers are literals


def test_retrieval_and_multihop_tracking():
    torch.manual_seed(0)
    cfg = RulerConfig(n_literals=8, n_vars=24, d_model=48, d_mem=64, hops=3)
    task = RulerTask(cfg)
    model = HopMemoryModel(cfg)
    train(model, task, steps=800, length=40, max_chain=3, n_distractor=3, seed=0)

    # NIAH retrieval works (chance is 1/8 = 0.125).
    niah = score(model, task, length=40, chain_len=1, n_distractor=3, hops=1)
    assert niah > 0.4, f"NIAH retrieval too low: {niah}"

    # 2-hop chain: iterative dereferencing resolves it; a single read cannot.
    multi = score(model, task, length=40, chain_len=2, n_distractor=3, hops=2)
    single = score(model, task, length=40, chain_len=2, n_distractor=3, hops=1)
    assert multi > 0.4, f"multi-hop tracking too low: {multi}"
    assert multi > single + 0.15, f"multi-hop ({multi}) should beat single read ({single})"
