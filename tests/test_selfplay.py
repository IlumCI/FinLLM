"""Self-play loop smoke test and proposer-policy properties."""

from __future__ import annotations

import numpy as np

from lamb import ArithmeticTokenizer, ModelConfig, TrainConfig
from lamb.model.lamb import build_model
from lamb.selfplay.loop import SelfPlayTrainer
from lamb.selfplay.proposer import Proposer


def test_proposer_never_collapses():
    """Even with one cell maximally learnable, the eps-floor keeps others alive."""
    prop = Proposer(n_cells=8, temperature=6.0, eps=0.35)
    learn = np.zeros(8)
    learn[3] = 1.0  # one dominant cell
    p = prop.probs(learn)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p.min() > 0.0  # every cell retains probability
    assert p.argmax() == 3  # but the learnable cell is preferred


def test_proposer_shifts_off_mastered_cells():
    prop = Proposer(n_cells=4, temperature=6.0, eps=0.1)
    # frontier at cell 1 (s=0.5 -> learnability 1.0); cell 0 mastered (s=1 -> 0)
    learn = np.array([0.0, 1.0, 0.25, 0.0])
    p = prop.probs(learn)
    assert p[1] > p[0]  # prefers the learnable frontier over the mastered cell


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
    assert stats.buffer_size > 0            # verified solved traces are being retained
    assert stats.proposer_entropy > 0.1     # 2-cell policy stays spread (max is ln 2)
    assert stats.token_acc > first.token_acc  # solver improved over the run
