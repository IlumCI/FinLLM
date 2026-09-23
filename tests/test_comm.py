"""Stage B -- latent inter-agent communication.

Covers the load-bearing properties: the channel bottleneck width, that a blanked
message is a constant (the no-information ablation is well-defined), that the task
is well-posed, that the message actually enters the listener's computation, and --
the differentiable-inter-agent-learning claim -- that the listener's loss
back-propagates into the speaker through the message.
"""

from __future__ import annotations

import torch

from lamb import ArithmeticTokenizer, CommConfig
from lamb.comm import Channel, CommTask, CommTrainer
from lamb.comm_transfer import held_out_partner
from lamb.coconut import _masked_ce
from lamb._native import evaluate, verify


def _trainer(**kw):
    torch.manual_seed(0)
    base = dict(steps=6, batch_size=16, d_model=48, recurrent_steps=3, eval_tasks=32,
                n_msg=2, n_listen_thoughts=1)
    base.update(kw)
    return CommTrainer(CommConfig(**base), ArithmeticTokenizer())


def test_channel_bottleneck_width():
    assert Channel(96, 8).width == 8         # throttled
    assert Channel(96, 0).width == 96        # 0 => full
    assert Channel(96, 200).width == 96      # clamped to d


def test_blank_message_is_constant():
    ch = Channel(32, 8)
    with torch.no_grad():
        out = ch(torch.zeros(4, 2, 32))      # zeroed message
    flat = out.reshape(-1, 32)
    assert torch.allclose(flat, flat[:1].expand_as(flat), atol=1e-6)  # every row identical


def test_task_is_wellposed_and_split():
    task = CommTask(1, 1, ("+", "-"), 0)
    for a_view, b_view, answer, full in task.sample(20):
        assert evaluate(full) == int(answer)   # exact
        assert verify(full, answer)             # verifier agrees
        # the listener's half never contains the speaker's operand X literally,
        # so it cannot reconstruct the answer without the message.
        assert b_view[0] in "+-" and a_view.isdigit()


def test_message_shape_and_enters_listener():
    tr = _trainer()
    msg = tr._message(["7", "3"])
    assert msg.shape == (2, tr.cfg.n_msg, tr.cfg.d_model)
    # real message vs a zeroed one produce different listener prefixes => the
    # channel is wired into the listener's computation.
    with torch.no_grad():
        x_real, _, _ = tr._listen_prefix(["+3", "-2"], msg)
        x_blank, _, _ = tr._listen_prefix(["+3", "-2"], torch.zeros_like(msg))
    assert float((x_real - x_blank).abs().max()) > 1e-5


def test_gradients_flow_speaker_and_channel():
    tr = _trainer()
    tr.speaker.train(); tr.listener.train(); tr.channel.train()
    samples = tr.task.sample(8)
    messages = tr._message([s[0] for s in samples])
    logits, targets, tmask = tr._answer_logits(samples, messages)
    _masked_ce(logits, targets, tmask).backward()
    # DIAL: the listener's answer loss reaches the speaker *through the message*.
    sp = sum(float(p.grad.norm()) for p in tr.speaker.parameters() if p.grad is not None)
    ch = sum(float(p.grad.norm()) for p in tr.channel.parameters() if p.grad is not None)
    assert sp > 0.0 and ch > 0.0


def test_accuracy_and_blank_run():
    tr = _trainer()
    comm = tr.accuracy(n_tasks=32)
    blank = tr.accuracy(n_tasks=32, blank=True)
    assert 0.0 <= comm <= 1.0 and 0.0 <= blank <= 1.0


def test_short_training_reduces_loss():
    tr = _trainer(steps=30, batch_size=24)
    first = tr._train_step(0)
    last = first
    for s in range(1, 30):
        last = tr._train_step(s)
    assert last < first


def test_foreign_speaker_produces_different_message():
    tr = _trainer()
    other = _trainer(seed=1)  # an independently initialised pair
    with torch.no_grad():
        m_self = tr._message(["7"])
        m_other = tr._message(["7"], speaker=other.speaker)
    assert m_self.shape == m_other.shape
    assert float((m_self - m_other).abs().max()) > 1e-5  # a different partner speaks differently


def test_held_out_partner_harness_runs():
    tok = ArithmeticTokenizer()
    cfg = CommConfig(steps=6, batch_size=16, d_model=48, recurrent_steps=3, eval_tasks=24)
    r = held_out_partner(cfg, tok, seeds=(0, 1), fresh_steps=6, n_tasks=24)
    assert set(r) >= {"A_matched", "A_swapped", "B_matched", "B_swapped", "blank", "fresh_partner"}
    assert all(0.0 <= v <= 1.0 for v in r.values())
