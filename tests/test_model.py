"""LAMb model: shapes, loss/backward, generation, adaptive halting, memory flag."""

from __future__ import annotations

import torch

from lamb import ArithmeticTokenizer, ModelConfig
from lamb._native import sample_problem
from lamb.data import collate
from lamb.model.lamb import build_model


def _tiny_cfg(**kw) -> ModelConfig:
    base = dict(d_model=64, n_heads=4, d_ff=128, recurrent_steps=2)
    base.update(kw)
    return ModelConfig(**base)


def _batch(tok, op="+", a=1, b=1, n=8):
    exs = [tok.encode(*sample_problem(op, a, b, s)) for s in range(n)]
    return collate(exs, tok.PAD)


def test_forward_and_loss_backward():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    model = build_model(_tiny_cfg(), tok)
    batch = _batch(tok)
    logits, aux = model.forward(
        batch["input_ids"], batch["abacus_ids"], batch["value"], batch["value_mask"], batch["pad_mask"]
    )
    b, t = batch["input_ids"].shape
    assert logits.shape == (b, t, tok.vocab_size)
    loss, metrics = model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert 0.0 <= metrics["token_acc"] <= 1.0


def test_n_steps_override_changes_compute():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    model = build_model(_tiny_cfg(), tok)
    batch = _batch(tok)
    l2, _ = model.compute_loss(batch, n_steps=2)
    l8, _ = model.compute_loss(batch, n_steps=8)
    # Different latent budgets should produce different (finite) losses.
    assert torch.isfinite(l2) and torch.isfinite(l8)
    assert abs(float(l2.detach()) - float(l8.detach())) > 0


def test_adaptive_halting_runs():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    model = build_model(_tiny_cfg(adaptive_halting=True, max_recurrent_steps=4), tok)
    batch = _batch(tok)
    loss, metrics = model.compute_loss(batch)
    assert torch.isfinite(loss)
    assert "ponder" in metrics
    loss.backward()


def test_memory_enabled_forward():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    model = build_model(_tiny_cfg(use_memory=True, d_mem=32), tok)
    batch = _batch(tok)
    loss, _ = model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()


def test_solve_returns_strings():
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    model = build_model(_tiny_cfg(), tok)
    out = model.solve(["1+2", "10+20", "7+8"], tok, max_answer_len=6)
    assert len(out) == 3
    assert all(o is None or isinstance(o, str) for o in out)


def test_learns_single_digit_addition():
    """A short train should let the tiny model actually solve 1-digit addition."""
    torch.manual_seed(0)
    tok = ArithmeticTokenizer()
    model = build_model(_tiny_cfg(recurrent_steps=3), tok)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(220):
        batch = collate([tok.encode(*sample_problem("+", 1, 1, s + step * 1000)) for s in range(64)], tok.PAD)
        loss, _ = model.compute_loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    probs = [sample_problem("+", 1, 1, 555000 + i) for i in range(40)]
    preds = model.solve([p for p, _ in probs], tok, max_answer_len=4)
    acc = sum(pr == a for (p, a), pr in zip(probs, preds)) / len(probs)
    assert acc > 0.5, f"expected >0.5 on 1-digit addition, got {acc}"
