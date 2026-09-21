"""Behavioural-novelty admission: BC vectors, the archive, and POET integration."""

from __future__ import annotations

import numpy as np

from lamb.selfplay.novelty import NoveltyArchive, behaviour_characterization


def test_novelty_archive_distance_gate():
    arch = NoveltyArchive(k=1, threshold=0.5)
    a = np.zeros(5)
    assert arch.is_novel(a)          # empty archive -> everything is novel
    arch.add(a)
    assert not arch.is_novel(a.copy())  # identical BC -> distance 0 -> not novel
    b = np.ones(5)
    assert arch.is_novel(b)          # far in behaviour space -> novel
    arch.add(b)
    assert len(arch) == 2


def test_behaviour_characterization_distinguishes_behaviour():
    tasks = [("1+1", "2"), ("2+2", "4")]
    truth = {"1+1": "2", "2+2": "4"}
    perfect = lambda probs, t: [truth[p] for p in probs]   # noqa: E731
    wrong = lambda probs, t: ["0" for _ in probs]          # noqa: E731
    bc_p = behaviour_characterization(perfect, tasks, budgets=(1, 2))
    bc_w = behaviour_characterization(wrong, tasks, budgets=(1, 2))
    assert bc_p.shape == (4,)                    # 2 budgets + mean_len + gain
    assert bc_p[0] == 1.0 and bc_w[0] == 0.0     # accuracy component differs
    assert np.linalg.norm(bc_p - bc_w) > 0.5     # behaviourally distinct


def test_poet_behavioural_novelty_runs():
    import torch

    from lamb import ArithmeticTokenizer, POETConfig
    from lamb.poet import POETTrainer

    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    cfg = POETConfig(iters=16, agent_d_model=48, agent_recurrent_steps=2, opt_steps=1, batch_size=16,
                     eval_tasks=8, bc_tasks=8, init_members=2, pop_capacity=6, transfer_every=4,
                     reproduce_every=4, reproduce_threshold=0.0, behavioural_novelty=True,
                     novelty_threshold=0.2, max_depth=3, max_digits=3)
    trainer = POETTrainer(cfg, tok)
    assert len(trainer.archive) == 2         # seeded with the initial members' BCs
    for _ in range(cfg.iters):
        trainer.iterate()
    assert trainer.novelty_rejects >= 0
    assert len(trainer.archive) >= 2
