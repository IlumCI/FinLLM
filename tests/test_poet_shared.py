"""Shared-backbone POET: adapter identity-init, efficiency, growth, specialisation."""

from __future__ import annotations

import torch

from lamb import ArithmeticTokenizer, POETConfig
from lamb.poet_shared import Adapter, SharedBackbonePOETTrainer


def test_adapter_starts_as_identity():
    a = Adapter(32, 8)
    h = torch.randn(2, 5, 32)
    assert torch.allclose(a(h), h)  # zero-init up-projection => identity at start


def _cfg(**kw):
    base = dict(agent_d_model=48, agent_recurrent_steps=2, adapter_rank=8, opt_steps=1,
                batch_size=16, eval_tasks=8, max_depth=3, max_digits=3)
    base.update(kw)
    return POETConfig(**base)


def test_param_efficiency_and_growth():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    cfg = _cfg(init_members=2, pop_capacity=4, transfer_every=3, reproduce_every=4,
              reproduce_threshold=0.0)
    tr = SharedBackbonePOETTrainer(cfg, tok)
    assert tr.adapter_params() * 20 < tr.backbone_params()   # adapters are tiny
    assert len(tr.members) == 2
    for _ in range(12):
        tr.iterate()
    assert len(tr.members) <= cfg.pop_capacity
    assert tr.attempted_frontier() >= 2


def test_adapter_specialises_after_training():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    tr = SharedBackbonePOETTrainer(_cfg(init_members=1, pop_capacity=4, opt_steps=2), tok)
    tr._optimize()
    assert float(tr.members[0].adapter.up.weight.detach().abs().sum()) > 0  # learned a specialisation
