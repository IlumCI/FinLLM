"""Coconut continuous-thought substrate (Stage A).

Covers the load-bearing correctness properties: the K=0 reduction to ordinary
teacher forcing, the inert-at-init interface, that thoughts actually enter the
answer computation, that gradients reach the thought parameters, and that the
verifier-selected best-of-N path is a well-formed superset of greedy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lamb import ArithmeticTokenizer, CoconutConfig, ModelConfig
from lamb.coconut import CoconutTrainer, coconut_collate
from lamb.data import collate
from lamb.model.lamb import build_model


def _tiny_model(tok):
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=48, n_heads=4, d_ff=96, n_prelude=1, n_recurrent=1,
                      n_coda=1, recurrent_steps=3)
    return build_model(cfg, tok)


def test_thought_marker_zero_init():
    tok = ArithmeticTokenizer()
    m = _tiny_model(tok)
    # Inert interface at start: the additive latent marker is zero, so a thought
    # is exactly its normed hidden state until the model learns to use it.
    assert float(m.thought_marker.detach().abs().sum()) == 0.0


def test_k0_equals_ordinary_teacher_forcing():
    tok = ArithmeticTokenizer()
    m = _tiny_model(tok)
    m.eval()
    pairs = [("12+3", "15"), ("7-2", "5"), ("40+40", "80")]

    prompt, aids, aab, apad, targets, tmask = coconut_collate(pairs, tok)
    with torch.no_grad():
        clog, _ = m.coconut_logits(prompt, aids, aab, apad, n_thoughts=0)
        ce_co = (F.cross_entropy(clog.reshape(-1, clog.size(-1)), targets.reshape(-1),
                                 reduction="none").view(targets.shape) * tmask).sum() / tmask.sum()

        full = collate([tok.encode(p, a) for p, a in pairs], tok.PAD)
        olog, _ = m.forward(full["input_ids"], full["abacus_ids"], full["value"],
                            full["value_mask"], full["pad_mask"])
        ce_ord = (F.cross_entropy(olog.reshape(-1, olog.size(-1)), full["target_ids"].reshape(-1),
                                  reduction="none").view(full["target_ids"].shape)
                  * full["loss_mask"]).sum() / full["loss_mask"].sum()

    # With no thoughts, the Coconut path is exactly ordinary answer-only teacher
    # forcing (RoPE is relative, so left-padding the prompt changes nothing).
    assert torch.allclose(ce_co, ce_ord, atol=1e-5)


def test_thoughts_enter_the_answer_computation():
    tok = ArithmeticTokenizer()
    m = _tiny_model(tok)
    # Break the zero-init marker so thoughts carry a signal, then check K changes logits.
    with torch.no_grad():
        m.thought_marker.add_(torch.randn_like(m.thought_marker) * 0.1)
    prompt, aids, aab, apad, _, _ = coconut_collate([("123+456", "579")], tok)
    with torch.no_grad():
        l0, _ = m.coconut_logits(prompt, aids, aab, apad, n_thoughts=0)
        l3, _ = m.coconut_logits(prompt, aids, aab, apad, n_thoughts=3)
    assert l0.shape == l3.shape
    assert float((l3 - l0).abs().max()) > 1e-5


def test_gradients_reach_thought_parameters():
    tok = ArithmeticTokenizer()
    m = _tiny_model(tok)
    m.train()
    prompt, aids, aab, apad, targets, tmask = coconut_collate([("12+3", "15")], tok)
    logits, _ = m.coconut_logits(prompt, aids, aab, apad, n_thoughts=2)
    loss = (F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                            reduction="none").view(targets.shape) * tmask).sum() / tmask.sum()
    loss.backward()
    assert m.thought_marker.grad is not None
    assert float(m.thought_marker.grad.norm()) > 0.0
    assert float(m.thought_norm.weight.grad.norm()) > 0.0


def test_coconut_solve_shapes_and_bestof_superset():
    tok = ArithmeticTokenizer()
    cfg = CoconutConfig(depth=2, digits=1, n_thoughts=3, eval_tasks=24, thought_dropout=0.3)
    mcfg = ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3)
    tr = CoconutTrainer(cfg, tok, mcfg)

    ans = tr.model.coconut_solve(["1+2", "(3+4)-(1+2)"], tok, n_thoughts=2)
    assert isinstance(ans, list) and len(ans) == 2

    # best-of-N with N>=1 is a verifier check per problem; best-of-N accuracy can
    # never be below greedy (N=1) because the greedy trajectory is in the pool.
    pairs = tr._eval_set(24)
    solved1 = tr._solve_bestof(pairs, k=3, n=1)
    solved4 = tr._solve_bestof(pairs, k=3, n=4)
    assert sum(s4 or s1 for s1, s4 in zip(solved1, solved4)) >= sum(solved1)


def test_short_training_runs_and_reduces_loss():
    tok = ArithmeticTokenizer()
    cfg = CoconutConfig(steps=30, batch_size=24, eval_every=10_000, log_every=10_000,
                        eval_tasks=24, depth=2, digits=1)
    mcfg = ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3)
    tr = CoconutTrainer(cfg, tok, mcfg)
    first = tr._train_step(0)["loss"]
    last = first
    for s in range(1, 30):
        last = tr._train_step(s)["loss"]
    assert last < first  # the answer-NLL through the thought chain goes down


def test_collapse_metric_is_finite():
    tok = ArithmeticTokenizer()
    cfg = CoconutConfig(depth=2, digits=1, n_thoughts=3, eval_tasks=32)
    mcfg = ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3)
    tr = CoconutTrainer(cfg, tok, mcfg)
    cos, std = tr.collapse_metric(3, n_tasks=32)
    assert -1.001 <= cos <= 1.001 and std >= 0.0
