"""LOTUS-style parallel supervised latent reasoning (Stage A, restructured).

The load-bearing properties: the gold numeric trace is exact, the latent block is
genuinely *parallel* (cost independent of the latent count -- the whole reason to
prefer it over the autoregressive Coconut loop), the latents are never decoded at
inference, and the per-position supervision actually trains.
"""

from __future__ import annotations

import torch

from lamb import ArithmeticTokenizer, LotusConfig
from lamb._native import evaluate
from lamb.config import ModelConfig
from lamb.lotus import LotusTrainer, trace_targets
from lamb.selfplay.grammar import Descriptor, TaskGrammar


def _trainer(**kw):
    torch.manual_seed(0)
    base = dict(steps=8, batch_size=16, n_latent=8, loops=3, eval_tasks=24, device="cpu")
    base.update(kw)
    mcfg = ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3)
    return LotusTrainer(LotusConfig(**base), ArithmeticTokenizer(), mcfg)


def test_gold_trace_is_exact_and_well_shaped():
    g = TaskGrammar()
    for depth in (1, 2, 3):
        d = Descriptor(depth=depth, digits=1, ops_key=0)
        for seed in range(25):
            expr, ans, trace = g.sample_with_trace(d, seed)
            assert evaluate(expr) == int(ans)          # answer agrees with the exact verifier
            assert len(trace) == 2 ** depth - 2        # post-order, root excluded
    # a concrete case: (a op b) op (c op d) -> the two sub-results, in order
    expr, ans, trace = TaskGrammar().sample_with_trace(Descriptor(2, 1, 0), 0)
    assert expr == "(6+6)-(4-8)" and ans == "16" and trace == [12, -4]


def test_trace_tokenisation():
    tok = ArithmeticTokenizer()
    # 12 -> LSB-first digits "2","1"; -4 -> minus then "4"
    assert trace_targets(tok, [12, -4]) == [11, 10, 5, 13]
    assert trace_targets(tok, []) == []


def test_latent_block_is_parallel_cost_independent_of_latent_count():
    """One core pass per loop, no matter how many latents -- the scalability claim.

    The Coconut path needed one sequential forward *per thought*; this needs
    ``loops + 1`` forwards whether there are 4 latents or 64.
    """
    counts = {}
    for n_latent in (4, 64):
        tr = _trainer(n_latent=n_latent, loops=3)
        calls = {"n": 0}
        orig = tr.reasoner.model.core.forward

        def counting(*a, _orig=orig, **k):
            calls["n"] += 1
            return _orig(*a, **k)

        tr.reasoner.model.core.forward = counting
        tasks = tr._sample_batch(4)
        prompt, aids, aab, apad, *_ = tr._collate(tasks)
        tr.reasoner(prompt, aids, aab, apad)
        tr.reasoner.model.core.forward = orig
        counts[n_latent] = calls["n"]
    assert counts[4] == counts[64] == 4          # loops(3) + 1 answer pass
    assert counts[64] == 4                        # 64 latents still cost 4 passes


def test_latents_are_never_decoded_at_inference():
    """Decoding starts *after* the latent block: the trace is never emitted."""
    tr = _trainer(n_latent=8)
    seen = {}
    import lamb.lotus as lotus_mod
    orig = lotus_mod._decode_answer

    def spy(model, x, pad, tok, max_len, device):
        seen["ctx_len"] = x.size(1)
        return orig(model, x, pad, tok, max_len, device)

    lotus_mod._decode_answer = spy
    try:
        out = tr.reasoner.solve(["(1+2)-(3+4)", "(5+5)-(1+1)"], tr.tok, tr.max_ans, device="cpu")
    finally:
        lotus_mod._decode_answer = orig
    assert len(out) == 2
    # context handed to the decoder = prompt + all latent positions
    assert seen["ctx_len"] > tr.cfg.n_latent      # latents are in context...
    # ...and the decoder only ever appends answer tokens, so nothing latent is emitted
    assert all(o is None or isinstance(o, str) for o in out)


def test_gradients_reach_latent_parameters():
    tr = _trainer()
    tr.reasoner.train()
    tasks = tr._sample_batch(8)
    prompt, aids, aab, apad, targets, tmask, tr_ids, tr_mask = tr._collate(tasks)
    from lamb.coconut import _masked_ce
    ans_logits, latent_logits, _sw, _ = tr.reasoner(prompt, aids, aab, apad)
    (_masked_ce(ans_logits, targets, tmask) + _masked_ce(latent_logits, tr_ids, tr_mask)).backward()
    assert float(tr.reasoner.latent_emb.weight.grad.norm()) > 0.0
    assert float(tr.reasoner.latent_marker.grad.norm()) > 0.0
    assert float(tr.reasoner.latent_norm.weight.grad.norm()) > 0.0


def test_trace_coef_zero_is_the_answer_only_ablation():
    tr = _trainer(trace_coef=0.0, switch_coef=0.0)   # isolate the answer term alone
    m = tr._train_step(0)
    assert m["trace"] == 0.0            # no per-position supervision applied
    assert m["switch"] == 0.0
    assert m["loss"] == m["ans"]


def test_boundary_tokens_are_new_ids_outside_the_answer_alphabet():
    tok = ArithmeticTokenizer()
    assert tok.BOT == tok._digit0 + tok.base and tok.EOT == tok.BOT + 1
    assert tok.vocab_size == tok.EOT + 1
    # every pre-existing id is untouched, and boundaries are never digits/signs
    assert not tok.is_digit_id(tok.BOT) and not tok.is_digit_id(tok.EOT)
    # a stray boundary token can never corrupt a decoded answer
    assert tok.decode_answer([tok.BOT, 10, tok.EOT, tok.EOS]) == "1"


def test_switch_logits_give_the_latent_segment_a_probability():
    """The entry boundary is a *predicted* token -- the hook on-policy RL needs."""
    tr = _trainer()
    tasks = tr._sample_batch(6)
    prompt, aids, aab, apad, *_ = tr._collate(tasks)
    _, _, switch_logits, _ = tr.reasoner(prompt, aids, aab, apad)
    assert switch_logits is not None
    assert switch_logits.shape == (6, tr.tok.vocab_size)
    logp = torch.log_softmax(switch_logits, dim=-1)[:, tr.tok.BOT]
    assert torch.isfinite(logp).all()          # a well-defined log-probability


def test_boundaries_can_be_ablated():
    tr = _trainer(use_boundaries=False)
    assert tr.reasoner.bot_id is None and tr.reasoner.eot_id is None
    tasks = tr._sample_batch(4)
    prompt, aids, aab, apad, *_ = tr._collate(tasks)
    _, _, switch_logits, _ = tr.reasoner(prompt, aids, aab, apad)
    assert switch_logits is None
    assert tr._train_step(0)["switch"] == 0.0
    assert tr.reasoner.boundary_states(prompt) is None


def test_exit_boundary_is_off_by_default():
    """Entry marker on; exit marker off (measured: it blocks the answer readout)."""
    tr = _trainer()
    assert tr.reasoner.bot_id is not None
    assert tr.reasoner.eot_id is None
    opt_in = _trainer(use_exit_boundary=True)
    assert opt_in.reasoner.eot_id is not None      # still available for ablation


def test_boundary_states_are_a_probe_attachment_point():
    tr = _trainer(n_latent=8)
    tasks = tr._sample_batch(5)
    prompt, *_ = tr._collate(tasks)
    states = tr.reasoner.boundary_states(prompt)
    assert states.shape == (5, tr.reasoner.model.cfg.d_model)
    assert torch.isfinite(states).all()


def test_training_reduces_loss_and_learns_the_trace():
    tr = _trainer(steps=40, batch_size=24, trace_coef=0.5)
    first = tr._train_step(0)
    last = first
    for s in range(1, 40):
        last = tr._train_step(s)
    assert last["loss"] < first["loss"]
    assert last["trace"] < first["trace"]   # the latent block is learning the intermediates
