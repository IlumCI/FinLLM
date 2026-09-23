"""On-policy RL over the latent block -- the machinery, not the (negative) result.

These assert the experiment is *well-posed*: the sampled budget action really does
change the latent computation, the log-probs correspond to what was sampled, and a
GRPO step runs end to end. The scientific finding (RL is inert here) lives in
docs/ROADMAP.md; a test suite should not pin an accuracy number from one seed.
"""

from __future__ import annotations

import torch

from lamb import ArithmeticTokenizer, LotusConfig
from lamb.coconut import coconut_collate
from lamb.config import ModelConfig
from lamb.latent_rl import BUDGETS, LatentPolicy, answer_tensors, evaluate, grpo_step
from lamb.lotus import LotusTrainer


def _setup(use_switch=True, steps=6):
    torch.manual_seed(0)
    cfg = LotusConfig(steps=20, batch_size=16, n_latent=8, loops=3, eval_tasks=16, device="cpu")
    mcfg = ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3)
    tr = LotusTrainer(cfg, ArithmeticTokenizer(), mcfg)
    for s in range(steps):
        tr._train_step(s)
    return tr, LatentPolicy(tr.reasoner, use_switch=use_switch)


def test_answer_tensors_match_the_sampled_sequence():
    tok = ArithmeticTokenizer()
    seqs = [[10, 11, tok.EOS], [12]]
    aids, aab, apad, targets, tmask = answer_tensors(tok, seqs, "cpu")
    # inputs are the content; targets are content + EOS
    assert aids[0, 0].item() == 10 and aids[0, 1].item() == 11
    assert targets[0, 2].item() == tok.EOS
    assert targets[1, 1].item() == tok.EOS
    assert float(tmask[0].sum()) == 3 and float(tmask[1].sum()) == 2
    assert bool(apad[1, 1])          # shorter row is padded


def test_budget_action_actually_changes_the_latent_computation():
    """If the sampled action were a no-op, the RL experiment would be meaningless."""
    tr, pol = _setup()
    tasks = tr._eval_set(6)
    prompt, *_ = coconut_collate([(e, "0") for e, _, _ in tasks], tr.tok, "cpu")
    seqs = pol.rollout(prompt, tr.tok, tr.max_ans, greedy=True)
    aids, aab, apad, _, _ = answer_tensors(tr.tok, seqs, "cpu")
    lo = torch.zeros(6, dtype=torch.long)                      # smallest budget
    hi = torch.full((6,), len(BUDGETS) - 1, dtype=torch.long)  # largest budget
    with torch.no_grad():
        a = pol.forward_logits(prompt, aids, aab, apad, lo)
        b = pol.forward_logits(prompt, aids, aab, apad, hi)
    assert float((a - b).abs().max()) > 1e-5


def test_switch_gives_a_well_defined_categorical_logprob():
    tr, pol = _setup()
    tasks = tr._eval_set(5)
    prompt, *_ = coconut_collate([(e, "0") for e, _, _ in tasks], tr.tok, "cpu")
    sl = pol.switch_logits(prompt)
    assert sl.shape == (5, len(BUDGETS))
    logp = torch.log_softmax(sl, dim=-1)
    assert torch.isfinite(logp).all()
    assert torch.allclose(logp.exp().sum(-1), torch.ones(5), atol=1e-5)


def test_no_switch_arm_has_no_switch_policy():
    tr, pol = _setup(use_switch=False)
    tasks = tr._eval_set(4)
    prompt, *_ = coconut_collate([(e, "0") for e, _, _ in tasks], tr.tok, "cpu")
    assert pol.switch_logits(prompt) is None
    assert pol.greedy_budget(prompt) is None


def test_grpo_step_runs_and_reports_metrics():
    tr, pol = _setup(steps=10)
    ref = LatentPolicy(tr.reasoner, use_switch=True)
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-4)
    tasks = tr._eval_set(6)
    m = grpo_step(pol, ref, tr.tok, tr.verifier, [e for e, _, _ in tasks], opt,
                  group_size=4, temperature=1.0, kl_coef=0.02, max_len=tr.max_ans,
                  device="cpu", entropy_coef=0.01)
    assert "reward" in m and 0.0 <= m["reward"] <= 1.0
    assert "kept" in m
    assert sum(m.get(f"mode{b}", 0.0) for b in BUDGETS) > 0.99   # a mode was chosen


def test_evaluate_runs_for_both_arms():
    tr, pol = _setup()
    tasks = tr._eval_set(8)
    acc = evaluate(pol, tasks, tr.tok, tr.verifier, tr.max_ans, "cpu")
    assert 0.0 <= acc <= 1.0
