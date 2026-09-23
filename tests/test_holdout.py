"""The train/eval split must be of the *problem space*, not of the random seeds.

This exists because seed separation silently failed: the depth-2/1-digit grammar
has ~80k expressions, a 1000-step run at batch 64 draws 64k of them, and 53% of a
seed-separated "held-out" set had in fact been trained on -- with the contamination
growing as training got longer, which biases exactly the longer-vs-shorter
comparisons the project makes. These tests pin the fix.
"""

from __future__ import annotations

from lamb import ArithmeticTokenizer, CoconutConfig, CommConfig, LotusConfig
from lamb.coconut import CoconutTrainer
from lamb.comm import CommTask
from lamb.config import ModelConfig
from lamb.holdout import is_heldout, is_trainable
from lamb.lotus import LotusTrainer
from lamb.selfplay.grammar import Descriptor, TaskGrammar

_MCFG = ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3)


def test_partition_is_deterministic_and_disjoint():
    g = TaskGrammar()
    d = Descriptor(2, 1, 0)
    probs = [g.sample_with_trace(d, s)[0] for s in range(2000)]
    assert all(is_heldout(p) == is_heldout(p) for p in probs)        # stable
    assert all(is_heldout(p) != is_trainable(p) for p in probs)      # a true partition
    frac = sum(is_heldout(p) for p in probs) / len(probs)
    assert 0.10 < frac < 0.20                                        # ~15% reserved


def test_grammar_train_sampler_never_emits_heldout_problems():
    g = TaskGrammar()
    d = Descriptor(2, 1, 0)
    assert not any(is_heldout(g.sample_with_trace(d, s, exclude_heldout=True)[0])
                   for s in range(1500))
    assert not any(is_heldout(g.sample(d, s, exclude_heldout=True)[0])
                   for s in range(1500))


def test_lotus_train_and_eval_streams_do_not_overlap():
    """The regression this file exists for: zero overlap, however long training runs."""
    tr = LotusTrainer(LotusConfig(device="cpu", eval_tasks=128), ArithmeticTokenizer(), _MCFG)
    ev = {e for e, _, _ in tr._eval_set(128)}
    train = {e for e, _, _ in tr._sample_batch(4000)}    # far more than the eval set
    assert all(is_heldout(e) for e in ev)
    assert len(ev & train) == 0


def test_coconut_train_and_eval_streams_do_not_overlap():
    tr = CoconutTrainer(CoconutConfig(device="cpu", eval_tasks=128), ArithmeticTokenizer(), _MCFG)
    ev = {p for p, _ in tr._eval_set(128)}
    train = {p for p, _ in tr._sample_batch(4000)}
    assert all(is_heldout(p) for p in ev)
    assert len(ev & train) == 0


def test_comm_task_splits_are_disjoint():
    train = CommTask(1, 1, ("+", "-"), 0, split="train")
    ev = CommTask(1, 1, ("+", "-"), 99_991, split="eval")
    trp = {s[3] for s in train.sample(2000)}
    evp = {s[3] for s in ev.sample(200)}
    assert all(is_heldout(p) for p in evp)
    assert not any(is_heldout(p) for p in trp)
    assert len(trp & evp) == 0


def test_eval_set_is_bounded_on_a_tiny_partition():
    """A 1-digit comm space has only ~22 held-out problems; asking for more must
    terminate (repeating) rather than spin forever."""
    ev = CommTask(1, 1, ("+", "-"), 7, split="eval")
    got = ev.sample(100)
    assert len(got) == 100
    assert all(is_heldout(s[3]) for s in got)
