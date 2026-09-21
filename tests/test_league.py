"""Solver league and Red Queen coevolution (relative fitness, novelty)."""

from __future__ import annotations

import numpy as np
import torch

from lamb import ArithmeticTokenizer, ModelConfig, TrainConfig
from lamb._native import sample_problem
from lamb.data import collate
from lamb.model.lamb import build_model
from lamb.selfplay.league import League
from lamb.selfplay.loop import SelfPlayTrainer


def _tiny():
    return ModelConfig(d_model=64, n_heads=4, d_ff=128, recurrent_steps=2)


def test_league_capacity_keeps_baseline_and_recent():
    tok = ArithmeticTokenizer()
    model = build_model(_tiny(), tok)
    league = League(capacity=4)
    for step in (10, 20, 30, 40, 50, 60):
        league.maybe_snapshot(model, step, every=10)
    steps = [s for s, _ in league.snapshots]
    assert len(league) == 4
    assert steps[0] == 10       # oldest baseline retained (for forgetting)
    assert steps[-1] == 60      # most recent retained


def test_league_no_snapshot_off_schedule():
    tok = ArithmeticTokenizer()
    league = League()
    assert not league.maybe_snapshot(build_model(_tiny(), tok), step=7, every=10)
    assert len(league) == 0


def test_relative_fitness_detects_progress():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    model = build_model(_tiny(), tok)
    league = League(4)
    league.maybe_snapshot(model, step=10, every=10)  # frozen while still untrained

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(200):
        batch = collate([tok.encode(*sample_problem("+", 1, 1, s + step * 1000)) for s in range(48)], tok.PAD)
        loss, _ = model.compute_loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

    tasks = [sample_problem("+", 1, 1, 700000 + i) for i in range(48)]
    rf = league.relative_fitness(model, tok, tasks, tasks, max_answer_len=4)
    assert rf["current_frontier"] > 0.3          # the trained solver can add
    assert rf["dominance"] >= 0.0                # ... and dominates its untrained past
    assert rf["forgetting"] == 0.0               # nothing to forget vs an untrained snapshot


def test_trainer_red_queen_runs():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    tcfg = TrainConfig(
        batch_size=32, max_digits=1, ops=("+", "-"), proposer_warmup=3,
        red_queen=True, league_snapshot_every=10, sample_train_depth=True, train_max_steps=3,
    )
    trainer = SelfPlayTrainer(tcfg, build_model(_tiny(), tok), tok)
    assert trainer.league is not None
    for _ in range(45):
        stats = trainer.train_step()

    assert len(trainer.league) >= 2                 # snapshots accumulated
    assert "league_size" in stats.extra
    assert sum(trainer.visits.values()) > 0.0       # novelty visitation is tracked
    rq = trainer.red_queen_report()
    assert rq is not None and "dominance" in rq and "forgetting" in rq
