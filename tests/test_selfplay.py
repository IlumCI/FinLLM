"""Self-play loop smoke tests and proposer-policy properties."""

from __future__ import annotations

import random

import numpy as np

from lamb import ArithmeticTokenizer, ModelConfig, TrainConfig
from lamb.model.lamb import build_model
from lamb.selfplay.hyperproposer import GRPOHyperProposer
from lamb.selfplay.loop import SelfPlayTrainer
from lamb.selfplay.proposer import BanditProposer, learnability


# -- bandit proposer (input is the per-cell success state s) ---------------
def test_bandit_never_collapses():
    prop = BanditProposer(n_cells=8, temperature=6.0, eps=0.35)
    s = np.zeros(8)
    s[3] = 0.5  # maximal learnability at cell 3
    p = prop.probs(s)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p.min() > 0.0            # every cell retains probability
    assert p.argmax() == 3         # the learnable cell is preferred


def test_bandit_shifts_off_mastered_cells():
    prop = BanditProposer(n_cells=4, temperature=6.0, eps=0.1)
    s = np.array([1.0, 0.5, 0.0, 1.0])  # cell0 mastered, cell1 the frontier
    p = prop.probs(s)
    assert p[1] > p[0]             # prefers the frontier over the mastered cell


# -- GRPO hypernetwork proposer -------------------------------------------
def test_hyperproposer_interface():
    hp = GRPOHyperProposer(n_cells=6, eps=0.2, rng=random.Random(0))
    s = np.zeros(6)
    s[2] = 0.5
    p = hp.probs(s)
    assert abs(p.sum() - 1.0) < 1e-6 and p.min() > 0.0
    cells = hp.sample(32, s)
    assert len(cells) == 32 and all(0 <= c < 6 for c in cells)
    assert hp.entropy(s) > 0.1


def test_hyperproposer_targets_the_frontier():
    import torch

    torch.manual_seed(0)
    hp = GRPOHyperProposer(n_cells=6, eps=0.2, kl_coef=0.3, lr=5e-3, rng=random.Random(0))
    # cell 2 is the frontier (s=0.5 -> max learnability); the rest are mastered or
    # unsolved (learnability ~0).
    s = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 1.0])
    reward = learnability(s)
    p_before = hp.probs(s)[2]
    for _ in range(150):
        cells = hp.sample(32, s)
        hp.update(s, cells, [reward[c] for c in cells])
    p = hp.probs(s)
    assert int(p.argmax()) == 2         # GRPO steers mass onto the learnable frontier
    assert p[2] > p_before              # ... and raised it relative to the start
    assert p.min() > 0.0                # eps floor keeps every cell reachable (coverage)


# -- full trainer smoke, both new modes -----------------------------------
def test_self_play_runs_and_fills_buffer():
    import torch

    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    mcfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, recurrent_steps=3)
    tcfg = TrainConfig(
        batch_size=48, max_digits=1, ops=("+", "-"), proposer_warmup=5,
        sample_train_depth=True, train_max_steps=3,
    )
    model = build_model(mcfg, tok)
    trainer = SelfPlayTrainer(tcfg, model, tok)

    first = trainer.train_step()
    for _ in range(220):
        stats = trainer.train_step()

    assert np.isfinite(stats.loss)
    assert stats.buffer_size > 0            # verified solved traces are retained
    assert stats.proposer_entropy > 0.1     # 2-cell policy stays spread (max is ln 2)
    assert stats.token_acc > first.token_acc  # solver improved over the run


def test_grpo_hyper_and_solver_modes_run():
    import torch

    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    mcfg = ModelConfig(d_model=64, n_heads=4, d_ff=128, recurrent_steps=2)
    tcfg = TrainConfig(
        batch_size=32, max_digits=1, ops=("+", "-"), proposer_warmup=2,
        proposer_kind="grpo_hyper", solver_algo="grpo", grpo_warmup=3,
        grpo_problems=6, grpo_group_size=3, grpo_ref_update_every=5,
        sample_train_depth=True, train_max_steps=3,
    )
    model = build_model(mcfg, tok)
    trainer = SelfPlayTrainer(tcfg, model, tok)
    assert trainer.ref_model is not None
    for _ in range(12):
        stats = trainer.train_step()

    assert np.isfinite(stats.loss)
    assert stats.proposer_entropy > 0.1
    assert "grpo_reward" in stats.extra   # GRPO term engaged after its warmup
