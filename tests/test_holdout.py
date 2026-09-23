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


def test_population_comm_trains_and_evaluates_on_disjoint_problems():
    """comm_pop holds out *pairings*; it must also hold out *problems*.

    Holding out only the ``(i, i)`` pairings measures a zero-shot partner on
    problems the population trained on, which is not the zero-shot-coordination
    question it claims to answer.
    """
    from lamb import ArithmeticTokenizer, CommConfig
    from lamb.comm_pop import PopulationComm

    cfg = CommConfig(steps=2, batch_size=8, pop_speakers=2, pop_listeners=2, device="cpu")
    pc = PopulationComm(cfg, ArithmeticTokenizer())
    train = {s[3] for _ in range(40) for s in pc.task.sample(32)}
    ev = {s[3] for s in pc._eval_set(256)}
    assert train and ev
    assert not (train & ev)


def test_comm_sampler_tops_up_without_recursing_into_its_own_padding():
    """A split that merely needs more draws must not be padded with duplicates."""
    from lamb.comm import CommTask

    from collections import Counter

    train = CommTask(2, 2, ("+", "-"), 0, split="train")
    got = train.sample(256)
    assert len(got) == 256
    # The 2-digit training partition holds ~13.7k problems, so 256 i.i.d. draws
    # collide a couple of times by the birthday effect and no more. The bug this
    # guards against padded the tail by *cycling* a short prefix, which shows up as
    # a low distinct count and a high multiplicity, not as a couple of collisions.
    counts = Counter(s[3] for s in got)
    assert len(counts) >= 245, len(counts)
    assert max(counts.values()) <= 3, counts.most_common(3)


def test_long_context_benchmarks_do_not_need_a_hash_partition():
    """The contamination problem was the *arithmetic grammar*, not seed separation.

    The needle/RULER generators draw from a combinatorial space of episodes so
    large that two seed-separated streams never collide -- unlike the depth-2
    1-digit grammar's 80k expressions, where a single run covered 80% of the space.
    Pinning this keeps the lesson from being over-generalised into a rewrite of
    benchmarks that were never affected.
    """
    import random

    import torch

    from lamb.memory_bench import RecallConfig, RecallTask
    from lamb.ruler_bench import RulerConfig, RulerTask

    torch.manual_seed(0)

    def recall_sigs(seed, batches, front):
        task, rng, out = RecallTask(RecallConfig()), random.Random(seed), set()
        for _ in range(batches):
            b, y = task.batch(32, 48, 8, rng, front=front)
            for i in range(32):
                out.add((tuple(b["key_ids"][i].tolist()), tuple(b["val_ids"][i].tolist()),
                         tuple(b["kind"][i].tolist()), int(y[i])))
        return out

    def ruler_sigs(seed, batches, front):
        task, rng, out = RulerTask(RulerConfig()), random.Random(seed), set()
        for _ in range(batches):
            b, y = task.generate(32, 64, 3, 8, rng, front=front)
            for i in range(32):
                out.add((tuple(b["kind"][i].tolist()), tuple(b["lhs"][i].tolist()),
                         tuple(b["rhs"][i].tolist()), int(y[i])))
        return out

    for train, ev in ((recall_sigs(0, 40, False), recall_sigs(999, 10, True)),
                      (ruler_sigs(0, 40, False), ruler_sigs(999, 10, True))):
        assert len(train) > 1000 and len(ev) > 250
        assert not (train & ev)


def test_division_problems_are_exact_and_the_old_op_sets_are_unchanged():
    """Adding division must not reshuffle the problems every prior measurement used.

    Draw order in the grammar is load-bearing: operands then operator. Choosing the
    operator first -- the natural way to write the division branch -- changes every
    problem the grammar has ever produced for a given seed, which would silently
    invalidate comparison with everything measured before it.
    """
    from lamb._native import evaluate
    from lamb.selfplay.grammar import Descriptor, TaskGrammar

    g = TaskGrammar()
    # the canonical value asserted elsewhere in the suite, unchanged
    expr, ans, trace = g.sample_with_trace(Descriptor(2, 1, 0), 0)
    assert (expr, ans, trace) == ("(6+6)-(4-8)", "16", [12, -4])

    seen_div = 0
    for depth in (1, 2, 3):
        for seed in range(150):
            e, a, _ = g.sample_with_trace(Descriptor(depth, 1, 2), seed)
            assert evaluate(e) == int(a), (e, a)      # every division is exact
            seen_div += "/" in e
    assert seen_div > 100                              # division actually appears
