"""GRPO utilities and the solver GRPO/RLVR objective."""

from __future__ import annotations

import torch

from lamb import ArithmeticTokenizer, ModelConfig
from lamb.model.lamb import build_model
from lamb.selfplay.grpo import (
    dynamic_keep_mask,
    grpo_solver_loss,
    group_advantages,
)
from lamb.selfplay.verifier import Verifier


def test_group_advantages_center_and_scale():
    r = torch.tensor([1.0, 0.0, 1.0, 0.0])
    g = torch.tensor([0, 0, 1, 1])
    adv = group_advantages(r, g, normalize_std=False)
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 0.5, -0.5]))
    # zero-mean per group when std-normalized too
    adv_n = group_advantages(r, g, normalize_std=True)
    assert abs(float(adv_n[:2].mean())) < 1e-6


def test_dynamic_keep_mask_drops_uniform_groups():
    r = torch.tensor([1.0, 1.0, 1.0, 0.0])
    g = torch.tensor([0, 0, 1, 1])
    keep = dynamic_keep_mask(r, g)
    assert keep.tolist() == [False, False, True, True]


def test_tokenizer_round_trips_sampled_answer_ids():
    tok = ArithmeticTokenizer()
    # emulate a rollout that produced the digits of 42 (reversed) then EOS
    enc_ref = tok.encode("40+2", "42")
    answer_ids = enc_ref.ids[enc_ref.ans_start :]  # includes EOS
    enc = tok.build_from_answer_ids("40+2", answer_ids)
    assert tok.decode_answer(enc.ids[enc.ans_start :]) == "42"


def test_grpo_solver_loss_runs_and_is_differentiable():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    model = build_model(ModelConfig(d_model=64, n_heads=4, d_ff=128, recurrent_steps=2), tok)
    ref = build_model(ModelConfig(d_model=64, n_heads=4, d_ff=128, recurrent_steps=2), tok)
    ref.load_state_dict(model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    # Train briefly so the solver sometimes gets a problem right -> non-uniform
    # groups survive dynamic sampling and a real GRPO loss is produced.
    from lamb._native import sample_problem
    from lamb.data import collate

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(150):
        batch = collate([tok.encode(*sample_problem("+", 1, 1, s + step * 1000)) for s in range(48)], tok.PAD)
        loss, _ = model.compute_loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

    problems = [sample_problem("+", 1, 1, 900000 + i)[0] for i in range(8)]
    loss, metrics, solved = grpo_solver_loss(
        model, ref, tok, Verifier(), problems, group_size=4, temperature=1.0, max_answer_len=4
    )
    assert 0.0 <= metrics["grpo_reward"] <= 1.0
    if loss is not None:  # at least some group had reward variance
        assert torch.isfinite(loss)
        loss.backward()
