"""Paired multi-seed comparison of the Stage A arms, with error bars.

Every headline number this repo reported was a single run. A single run cannot
distinguish a real effect from initialisation noise, and the spread between runs
here is several accuracy points -- the same scale as some of the differences that
were being claimed. One of those claims (the SWITCH entry boundary) did not
survive a clean evaluation split; the rest were never given the chance to fail.

So: run every arm at several seeds and report the *paired* difference. Pairing
matters. Within a seed the arms share a model-initialisation seed and the same
held-out evaluation set (``lamb.holdout`` partitions by problem, so the set is a
function of the seed alone), which removes the two largest nuisance terms and
leaves a difference estimate far tighter than comparing independent means.

The confidence interval is a percentile bootstrap over seeds rather than a
t-interval: with 5 seeds the normality assumption is doing more work than the
data can support, and the bootstrap at least makes no distributional claim.

Task spaces -- the second thing a single setting cannot tell you. Every previous
number came from ``depth-2/1-digit``, a space of only 80k expressions, of which a
1000-step run at batch 64 draws 64k. Held-out problems are genuinely unseen
(that is what the hash partition guarantees), but the model has still seen nearly
every *other* problem in the space, so the measurement sits close to the
memorisation regime. ``d2g2`` (5.2e8 expressions) and ``d3g1`` (1.3e10) put the
training set at 0.01% and 0.0005% of the space, where memorisation is not
available and only the structure can generalise. If the restructure's advantage
is real it should survive there; if it was an artefact of a small space it should
not.

Usage::

    python -m lamb.study --task d2g1 --seeds 5 --workers 4
    python -m lamb.study --task d2g2 --steps 1500
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# -- task spaces ----------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    """A task size plus the latent budget it needs.

    ``n_latent`` must cover the gold trace, or the supervision is silently
    truncated and the trace arm is handicapped by construction. The p95/max trace
    lengths measured over the grammar are 4/4 (d2g1), 6/6 (d2g2) and 12/14 (d3g1)
    tokens, so d3g1 needs twice the latent block. The *same* budget is used for
    both LOTUS arms within a task, so the comparison stays matched.
    """

    depth: int
    digits: int
    ops_key: int
    n_latent: int
    space: float          # analytic size of the expression space, for context

    def label(self) -> str:
        return f"d{self.depth}g{self.digits}o{self.ops_key}"


TASKS: Dict[str, TaskSpec] = {
    # the original setting: small enough that training covers most of the space
    "d2g1": TaskSpec(depth=2, digits=1, ops_key=0, n_latent=8, space=8.0e4),
    # wide operands: same reasoning shape, memorisation unavailable
    "d2g2": TaskSpec(depth=2, digits=2, ops_key=0, n_latent=8, space=5.25e8),
    # deeper nesting: 6 intermediates instead of 2, so the trace carries more
    "d3g1": TaskSpec(depth=3, digits=1, ops_key=0, n_latent=16, space=1.28e10),
    # multiplication unlocked
    "d2g1x": TaskSpec(depth=2, digits=1, ops_key=1, n_latent=8, space=2.7e5),
    # division unlocked -- needs rational registers, since the integer ring has no
    # division at all. The leaf count is ~390 (300 for +,-,* and 90 constructed
    # exact quotients), so depth 2 is ~390^2 * 4.
    "d2g1d": TaskSpec(depth=2, digits=1, ops_key=2, n_latent=8, space=6.1e5),
}

# Wall-clock cost per training step, measured on this box at d_model=96, batch 64,
# depth-2/1-digit: LOTUS 0.564 s, Coconut 0.335 s. The arms are matched on *core
# forward passes* -- Coconut's K=3 thoughts plus an answer pass is 4, LOTUS's
# loops=3 plus an answer pass is 4 -- but not on FLOPs, because LOTUS carries the
# latent block in every sequence. So at equal steps LOTUS gets ~1.68x the compute,
# and a win could be bought rather than structural. ``coconut-long`` is the control
# that removes the confound: the same Coconut, given the extra steps instead.
COMPUTE_MATCH = 0.564 / 0.335

ARMS = ("coconut", "coconut-long", "lotus-answer", "lotus-trace")
# Opt-in arms, not run by default -- they cost a full sweep each and neither has
# been shown to do anything yet. ``lotus-space`` adds the supervised-contrastive
# term over the latent manifold (arXiv:2606.20075's second supervision dimension).
EXTRA_ARMS = ("lotus-space", "lotus-alu")
# The register machine: the latents emit a *program* and the algebra executes it.
# These exist because ROADMAP 3a-vii -- the one learned claim still standing, and the
# one the language bridge depends on -- was measured at n=1, at depth 2, over three
# instructions, on ``(+,-)``. This file was built specifically to stop claims like
# that, and had never been pointed at it.
#
# The question the pair asks is whether the *differentiable* executor is a capability
# or a convenience: ``answer-only`` removes the gold program entirely, so the only
# thing that can shape a pointer is the answer loss arriving through exact
# arithmetic. Read ``acc`` (answer accuracy); ``canonical_acc`` is conformity to the
# generator's form and is expected near zero for the answer-only arm, because 48
# distinct three-instruction programs compute ``(a+b)+(c+d)``.
REGMACHINE_ARMS = ("regmachine-supervised", "regmachine-answer-only")

# The same two arms with redundant moduli, which turns a silent wrong answer into an
# admitted refusal. They exist because the depth-2 run measured `mis_answered` at
# **0.442** on the answer-only arm: 44% of held-out problems answered with a
# confident wrong number, because CRT has no locality and no arm had the redundancy
# of 3a-ix switched on. That machinery had been measured standalone (100% of single
# errors corrected, 0% mis-corrected) and imported by nothing.
#
# **What this pair does and does not compare.** Adding three moduli changes the
# prediction task -- the model must now get 8 residue blocks right instead of 5 --
# so `answer_acc` is *not* like-for-like against the non-redundant arms, and a drop
# there is the price of the wider code rather than a regression. What is
# like-for-like, and the point, is how the error budget inside each arm splits:
# `answer_acc + refused + mis_answered = 1` over kept rows, and 3a-ix predicts
# `mis_answered` collapses while `refused` absorbs it.
REDUNDANT_ARMS = ("regmachine-supervised-redundant",
                  "regmachine-answer-only-redundant")

# The two regimes between the constant extremes, neither of which had ever been run
# even though ``RegMachineTrainer``'s own docstring describes the first as the design
# ("supervise the program first, lean on the answer after").
#
# ``anneal``   supervised for 200 steps, gold program withdrawn linearly by step 600,
#              answer-only for the last 400. Asks whether the answer loss can *hold* a
#              program it could not *find* -- which separates a cold-start problem from
#              a bad-gradient one.
# ``partial42`` a gold program for a hash-selected 42% of problems and nothing for the
#              rest, which is not a hypothetical: it is exactly what GSM8K's calculator
#              annotations supply (3c). Membership is per *problem*, not per step,
#              because a dataset either annotates a problem or never does.
# Interventions aimed at the *measured* cause of the answer-only failure rather than at
# the failure itself. 3a-xii found the failing seeds are uncommitted, not wrong --
# ptr_sharp 1.0 gives 1.000 and anything below gives ~0.04, with nothing between -- so
# these force commitment and ask whether that is sufficient without a warm start.
#   commit   an entropy penalty on the op and pointer distributions
#   gumbel   straight-through discrete sampling, so the executor sees a real program
#            rather than a blur of several (the mechanism RegisterMachine has carried,
#            unused, since it was written)
COMMIT_ARMS = ("regmachine-answer-only-commit", "regmachine-answer-only-gumbel")
CURRICULUM_ARMS = ("regmachine-anneal", "regmachine-partial42")
ALL_ARMS = (ARMS + EXTRA_ARMS + REGMACHINE_ARMS + REDUNDANT_ARMS
            + CURRICULUM_ARMS + COMMIT_ARMS)

# Ring for the register machine's integer path: the same moduli the depth-2 result
# of 3a-vii was produced with, so that arm stays comparable.
REGMACHINE_MODULI: Tuple[int, ...] = (16, 25, 27, 11, 37)

# Three redundant moduli on top, which is the sizing 3a-ix measured as the point
# where single errors are fully corrected rather than partly refused. Chosen to keep
# the widest modulus at 41 rather than 101 or 271 -- the packed path pads everything
# to the widest, so a large redundant modulus costs the whole batch. Legitimate
# values then occupy 2.198e6 of an 8.200e9 ring (0.027%), which is what makes a
# corrupted residue land outside it, and every digit period stays <= 6.
REDUNDANT_MODULI: Tuple[int, ...] = (7, 13, 41)


# -- one run --------------------------------------------------------------
# Set in every worker by the pool initializer; ``None`` means CPU-only.
_GPU_SEM = None


def _init_worker(sem) -> None:
    global _GPU_SEM
    _GPU_SEM = sem


def _run_one(job: Tuple[str, str, int, int, int, int]) -> Dict[str, object]:
    """Train one (arm, seed) and return its held-out accuracy.

    Runs in its own process with a single torch thread: the arms are embarrassingly
    parallel across seeds, and one thread each beats N threads contending.
    """
    arm, task_key, seed, steps, batch_size, d_model, allow_gpu = job
    # **The device is chosen by the worker, not baked into the job.** Assigning it
    # statically -- round-robin over a device list -- looked fine and load-balanced
    # terribly: with the card ~3.4x faster, a fixed share of jobs pinned to CPU becomes
    # the entire wall clock while the GPU sits idle at the end. Measured: 9 GPU jobs
    # finishing in ~25 min behind 6 CPU jobs taking ~56.
    #
    # A semaphore fixes it without needing to know the speed ratio. Whichever worker is
    # free claims the card if a slot is open and falls back to CPU otherwise, so fast
    # workers churn through jobs and the split settles wherever the hardware puts it.
    device = "cpu"
    claimed = False
    if allow_gpu and _GPU_SEM is not None and _GPU_SEM.acquire(block=False):
        device, claimed = "cuda", True
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    from lamb import ArithmeticTokenizer, CoconutConfig, LotusConfig
    from lamb.config import ModelConfig
    from lamb.coconut import CoconutTrainer
    from lamb.lotus import LotusTrainer

    spec = TASKS[task_key]
    tok = ArithmeticTokenizer()
    mcfg = ModelConfig(d_model=d_model, n_heads=4, d_ff=2 * d_model,
                       n_prelude=1, n_recurrent=1, n_coda=1, recurrent_steps=4)
    t0 = time.time()
    out: Dict[str, object] = {"arm": arm, "task": task_key, "seed": seed,
                              "device": device}
    # **fp32 on the GPU, deliberately.** ``Amp`` would turn on bf16 autocast, and the
    # algebra is distributions over small rings: in reduced precision the tail
    # underflows, the convolutions lose mass, and a renormalised-but-wrong distribution
    # decodes to a *different integer* -- which in a residue system is not a near miss.
    # The tensors are tiny, so the precision costs nothing worth having, and the
    # measured speedup below was obtained without it.

    if arm.startswith("coconut"):
        if arm == "coconut-long":
            steps = int(round(steps * COMPUTE_MATCH))
        cfg = CoconutConfig(steps=steps, batch_size=batch_size, seed=seed,
                            depth=spec.depth, digits=spec.digits, ops_key=spec.ops_key,
                            device=device)
        tr = CoconutTrainer(cfg, tok, mcfg)
        for s in range(steps):
            tr._train_step(s)
        # Coconut's budget is K sequential thoughts; K=n_thoughts is its own best
        # setting and costs K+1 forwards, matching the LOTUS loops+1.
        out["acc"] = tr.greedy_accuracy(cfg.n_thoughts, 512)
        out["acc_k0"] = tr.greedy_accuracy(0, 512)
        out["steps_run"] = steps
    elif arm.startswith("regmachine"):
        from lamb.regmachine import RegMachineTrainer

        # ``ops_key=2`` means division, which plain residues cannot express at all,
        # so that task selects the rational register file rather than being a
        # separate arm. The instruction set follows the ring, not the flag.
        rational = spec.ops_key >= 2
        redundant = REDUNDANT_MODULI if arm.endswith("-redundant") else None
        cfg = LotusConfig(steps=steps, batch_size=batch_size, seed=seed,
                          depth=spec.depth, digits=spec.digits, ops_key=spec.ops_key,
                          n_latent=spec.n_latent, loops=3,
                          trace_coef=0.0, alu_coef=0.0, alu_consistency_coef=0.0,
                          alu_moduli=REGMACHINE_MODULI,
                          use_boundaries=False, switch_coef=0.0, device=device)
        # 200 supervised steps then a linear withdrawal to zero by 600, scaled if the
        # step budget differs so the schedule is a fraction of training rather than an
        # absolute that silently becomes "always supervised" on a longer run.
        anneal = ((int(0.2 * steps), int(0.6 * steps))
                  if arm == "regmachine-anneal" else None)
        frac = 0.42 if arm == "regmachine-partial42" else 1.0
        ent = 0.02 if arm.endswith("-commit") else 0.0
        gum = (1.0, True) if arm.endswith("-gumbel") else (0.0, False)
        tr = RegMachineTrainer(
            cfg, tok, mcfg, rational=rational, redundant_moduli=redundant,
            program_anneal=anneal, program_frac=frac, entropy_coef=ent,
            tau=gum[0], hard=gum[1],
            # ``in``, not ``endswith``: the redundant arms are named
            # ``...-answer-only-redundant``, so an ``endswith("answer-only")`` test
            # silently hands them ``program_coef=1`` and the arm measures the wrong
            # thing while looking like it ran.
            program_coef=0.0 if "answer-only" in arm else 1.0,
            answer_coef=1.0)
        for s in range(steps):
            tr.train_step(s)
        r = tr.evaluate(512)
        out["acc"] = r["answer_acc"]
        for k in ("answer_acc_hard", "canonical_acc", "instr_acc", "op_acc",
                  "ptr_acc", "dropped", "ptr_sharp", "refused", "mis_answered",
                  "max_den_magnitude", "zero_den"):
            if k in r:
                out[k] = r[k]
        out["steps_run"] = steps
    else:
        cfg = LotusConfig(steps=steps, batch_size=batch_size, seed=seed,
                          depth=spec.depth, digits=spec.digits, ops_key=spec.ops_key,
                          n_latent=spec.n_latent, loops=3,
                          trace_coef=(0.0 if arm in ("lotus-answer", "lotus-alu")
                                      else LotusConfig.trace_coef),
                          space_coef=0.3 if arm == "lotus-space" else 0.0,
                          alu_coef=1.0 if arm == "lotus-alu" else 0.0,
                          alu_consistency_coef=0.0,
                          use_boundaries=False, switch_coef=0.0, device=device)
        tr = LotusTrainer(cfg, tok, mcfg)
        for s in range(steps):
            tr._train_step(s)
        if arm == "lotus-alu":
            # The ALU's answer is composed by the algebra, not decoded, so its
            # accuracy is the composed one. The readout is kept beside it because
            # this arm does not train its decoder, and reporting the readout as
            # "the ALU's accuracy" would understate it as badly as reporting the
            # composed number for the other arms would overstate them.
            r = tr.algebraic_accuracy(512)
            out["acc"] = r["algebraic"]
            out["acc_readout"] = r["readout"]
            out["leaf_residue"] = r["leaf_residue"]
            out["root_residue"] = r["root_residue"]
            out["in_range"] = r["in_range"]
        else:
            out["acc"] = tr.accuracy(512)
        out["trace_probe"] = tr.trace_probe(256)
        out["collapse"] = tr.collapse_metric(256)
        # Fraction of gold traces that did not fit the latent block. Non-zero means
        # the trace arm is being handicapped and n_latent is too small.
        out["truncated"] = tr.truncated / float(steps * batch_size)
    out["secs"] = time.time() - t0
    if claimed:
        _GPU_SEM.release()
    return out


# -- statistics -----------------------------------------------------------
def _bootstrap_ci(xs: Sequence[float], reps: int = 20000, alpha: float = 0.05,
                  seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean. No normality assumption."""
    if len(xs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(reps):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * reps)]
    hi = means[min(reps - 1, int((1 - alpha / 2) * reps))]
    return lo, hi


def _sem(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")


def _perm_p_unpaired(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact two-sided permutation test on the difference of means.

    With few seeds this says more than any interval. It assumes only that, under
    the null, the arm labels are exchangeable -- no normality, no variance
    assumption. Complete separation of two arms of 5 gives p = 2/C(10,5) = 0.008,
    which is the strongest statement 5+5 runs can support.
    """
    from itertools import combinations

    pool = list(a) + list(b)
    n, obs = len(a), statistics.mean(a) - statistics.mean(b)
    idx = range(len(pool))
    total = extreme = 0
    for pick in combinations(idx, n):
        ps = set(pick)
        ma = statistics.mean([pool[i] for i in pick])
        mb = statistics.mean([pool[i] for i in idx if i not in ps])
        total += 1
        if abs(ma - mb) >= abs(obs) - 1e-12:
            extreme += 1
    return extreme / total


def _perm_p_paired(d: Sequence[float]) -> float:
    """Exact two-sided sign-flip test on the paired differences.

    The paired counterpart: under the null a difference is as likely to fall
    either way, so flipping signs enumerates the null exactly. With n seeds the
    smallest attainable p is 2/2**n, so 5 seeds can never go below 0.0625 however
    large the effect -- which is itself worth printing, because it puts a floor on
    what this design can claim.
    """
    from itertools import product

    obs = abs(statistics.mean(d))
    total = extreme = 0
    for signs in product((1, -1), repeat=len(d)):
        m = abs(statistics.mean([s * x for s, x in zip(signs, d)]))
        total += 1
        if m >= obs - 1e-12:
            extreme += 1
    return extreme / total


def min_detectable(sd: float, n: int) -> float:
    """Roughly the smallest paired difference n seeds can resolve (2 SEM)."""
    return 2.0 * sd * (2.0 ** 0.5) / max(1, n) ** 0.5


def seeds_needed(sd: float, effect: float) -> int:
    """Seeds for a paired difference of ``effect`` to clear 2 SEM."""
    if effect <= 0:
        return 10 ** 6
    return max(2, int((2.0 * sd * (2.0 ** 0.5) / effect) ** 2 + 0.999))


def summarise(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Per-arm means and, crucially, the *paired* differences between arms."""
    by_arm: Dict[str, Dict[int, float]] = {a: {} for a in ALL_ARMS}
    for r in rows:
        by_arm[str(r["arm"])][int(r["seed"])] = float(r["acc"])   # type: ignore[index]

    summary: Dict[str, object] = {"arms": {}, "pairs": {}}
    for arm, per_seed in by_arm.items():
        if not per_seed:
            continue
        xs = [per_seed[s] for s in sorted(per_seed)]
        lo, hi = _bootstrap_ci(xs)
        summary["arms"][arm] = {                       # type: ignore[index]
            "n": len(xs), "mean": statistics.mean(xs), "sem": _sem(xs),
            "min": min(xs), "max": max(xs), "ci95": [lo, hi], "per_seed": xs,
        }
        # Arm-specific diagnostics, averaged. ``dropped`` is the one that changes how
        # a number should be read: an accuracy measured on 60% of a held-out set is
        # not the same number as one measured on all of it, and a reader should not
        # have to reconstruct which it was.
        extras = {}
        for k in ("answer_acc_hard", "dropped", "canonical_acc", "op_acc",
                  "ptr_acc", "ptr_sharp", "refused", "mis_answered", "zero_den",
                  "max_den_magnitude", "trace_probe", "truncated"):
            vs = [float(r[k]) for r in rows
                  if str(r["arm"]) == arm and r.get(k) is not None]
            if vs:
                extras[k] = statistics.mean(vs)
        if extras:
            summary["arms"][arm]["extras"] = extras     # type: ignore[index]

    for a, b in (("lotus-answer", "coconut"), ("lotus-trace", "lotus-answer"),
                 ("lotus-trace", "coconut"),
                 # the confound controls: same wall clock, not just same step count
                 ("coconut-long", "coconut"), ("lotus-answer", "coconut-long"),
                 ("lotus-trace", "coconut-long"),
                 # the untested second supervision dimension, and the ALU
                 ("lotus-space", "lotus-trace"), ("lotus-alu", "lotus-trace"),
                 # the pre-registered question: does outcome-only program induction
                 # survive past three instructions? The bridge has no gold program
                 # available at all, so this is the claim it rests on.
                 ("regmachine-answer-only", "regmachine-supervised"),
                 # within the redundant ring, the same question
                 ("regmachine-answer-only-redundant",
                  "regmachine-supervised-redundant"),
                 # across rings: this prices the *wider code*, not the model, since
                 # the redundant arm has three more residue blocks to get right
                 ("regmachine-answer-only-redundant", "regmachine-answer-only"),
                 ("regmachine-supervised-redundant", "regmachine-supervised"),
                 # can the answer loss hold a program it could not find?
                 ("regmachine-anneal", "regmachine-supervised"),
                 ("regmachine-anneal", "regmachine-answer-only"),
                 # is the bridge's 42% coverage enough at 7 instructions?
                 ("regmachine-partial42", "regmachine-supervised"),
                 ("regmachine-partial42", "regmachine-answer-only"),
                 # is forcing commitment enough on its own, with no warm start?
                 ("regmachine-answer-only-commit", "regmachine-answer-only"),
                 ("regmachine-answer-only-gumbel", "regmachine-answer-only"),
                 ("regmachine-answer-only-commit", "regmachine-supervised")):
        shared = sorted(set(by_arm[a]) & set(by_arm[b]))
        if len(shared) < 2:
            continue
        d = [by_arm[a][s] - by_arm[b][s] for s in shared]
        lo, hi = _bootstrap_ci(d)
        xa = [by_arm[a][s] for s in shared]
        xb = [by_arm[b][s] for s in shared]
        # The most convincing thing few runs can show: the arms' seed ranges do
        # not overlap at all. No interval is needed to read that.
        separated = min(xa) > max(xb) or min(xb) > max(xa)
        summary["pairs"][f"{a} - {b}"] = {             # type: ignore[index]
            "n": len(d), "mean": statistics.mean(d), "sem": _sem(d),
            "ci95": [lo, hi], "crosses_zero": lo <= 0.0 <= hi, "per_seed": d,
            "separated": separated,
            "p_perm_unpaired": _perm_p_unpaired(xa, xb) if len(d) <= 10 else None,
            "p_signflip_paired": _perm_p_paired(d) if len(d) <= 20 else None,
        }
    return summary


def _fmt(summary: Dict[str, object], spec: TaskSpec, steps: int, batch: int) -> str:
    out = []
    seen = steps * batch
    out.append(f"task {spec.label()}  space ~{spec.space:.2e}  "
               f"train draws {seen:,} ({100.0 * seen / spec.space:.4f}% of the space)  "
               f"n_latent={spec.n_latent}")
    out.append("")
    out.append(f"{'arm':>14}  {'n':>2}  {'mean':>6}  {'sem':>6}  {'min':>6}  {'max':>6}  95% CI")
    for arm in ALL_ARMS:
        a = summary["arms"].get(arm)                    # type: ignore[union-attr]
        if not a:
            continue
        out.append(f"{arm:>14}  {a['n']:>2}  {a['mean']:6.3f}  {a['sem']:6.3f}  "
                   f"{a['min']:6.3f}  {a['max']:6.3f}  [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]")
        ex = a.get("extras")
        if ex:
            out.append(f"{'':>14}  " + "  ".join(f"{k} {v:.3f}" for k, v in ex.items()))
    out.append("")
    out.append("paired differences (same seed => same init and same eval set)")
    for k, p in summary["pairs"].items():               # type: ignore[union-attr]
        verdict = "CROSSES ZERO" if p["crosses_zero"] else "excludes zero"
        sep = "  SEED RANGES DISJOINT" if p["separated"] else ""
        out.append(f"{k:>30}  {p['mean']:+.3f}  sem {p['sem']:.3f}  "
                   f"95% CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  {verdict}{sep}")
        # both exact tests become intractable past a modest seed count; say so
        # rather than crash or silently drop the line.
        pu = p["p_perm_unpaired"]
        pp = p["p_signflip_paired"]
        out.append(f"{'':>30}  exact permutation p "
                   f"{f'{pu:.4f}' if pu is not None else 'n/a (too many seeds)'} (unpaired)"
                   f"   sign-flip p "
                   f"{f'{pp:.4f}' if pp is not None else 'n/a'} (paired, floor "
                   f"{2 / 2 ** p['n']:.4f})")
    out.append("")
    out.append("resolution -- what this many seeds can and cannot show")
    for arm in ALL_ARMS:
        a = summary["arms"].get(arm)                    # type: ignore[union-attr]
        if not a or a["n"] < 2:
            continue
        sd = statistics.stdev(a["per_seed"])
        out.append(f"{arm:>14}  sd {sd:.3f}  min detectable diff at n={a['n']}: "
                   f"{min_detectable(sd, a['n']):.3f}   "
                   f"seeds for a 0.05 diff: {seeds_needed(sd, 0.05)}")
    return "\n".join(out)


# -- driver ---------------------------------------------------------------
def run_study(task_key: str, seeds: int = 5, steps: int = 1000, batch_size: int = 64,
              d_model: int = 96, workers: int = 4, arms: Sequence[str] = ARMS,
              gpu_workers: int = 0,
              out_path: Optional[str] = None, seed_offset: int = 0,
              merge: Optional[str] = None) -> Dict[str, object]:
    """Run ``arms`` at ``seeds`` seeds and summarise.

    ``seed_offset`` starts the seed range past an earlier run's, and ``merge``
    folds in that run's rows, so a study can be *extended* to more seeds without
    recomputing the ones already paid for. That matters because the seed count
    needed is not known until the first few runs reveal the spread.
    """
    spec = TASKS[task_key]
    # There used to be a clamp here: if ``resolve_device("auto")`` reported CUDA,
    # workers dropped to 1, because one CUDA context per worker is ~300-500 MB before
    # a single parameter is allocated and four of them exhaust a small card.
    #
    # It was guarding a situation that could not arise *then*: every arm pinned
    # ``device="cpu"``, so merely *having* a card never put a context in a worker.
    # Arms are device-aware again (see ``gpu_workers`` below), and the VRAM budget
    # that replaces this clamp is sized from measurement. What the clamp did
    # instead was call ``torch.cuda.is_available()``, which initialises CUDA in the
    # parent and poisoned the fork (see the spawn comment below). So it cost a 4x
    # slowdown on any GPU box and caused the failure it was written to prevent.
    #
    # Restore it if an arm ever becomes device-aware. Until then the honest statement
    # is that the card is irrelevant to this file.
    if merge and not os.path.exists(merge):
        # Validate before spending the compute, not after. Discovering a bad path
        # in the summary step throws away every run that preceded it.
        raise FileNotFoundError(f"--merge file does not exist: {merge}")
    # **Hybrid GPU + CPU.** Measured on this box at depth 3: 0.50 s/step on the card
    # against ~1.7 s/step on eight CPU threads at batch 64, and the gap widens with
    # batch (0.73 s/step at batch 256, where CPU is ~9.6). The repo's note that "at
    # 283k parameters every kernel is too small to fill a GPU" does not hold for the
    # register-machine path: the packed algebra moves ``batch x slots x K x P`` tensors,
    # which dwarf the transformer at this width.
    #
    # Jobs are independent, so the card and the cores run different seeds at the same
    # time and throughput is the sum rather than the max. GPU workers are capped by
    # VRAM, not by preference: each carries its own CUDA context (~300 MB) on top of its
    # activations (~490 MB at batch 64, ~2.5 GB at batch 256), and a 4 GB card holds
    # very few of the latter. Overcommitting does not degrade gracefully, it OOMs
    # mid-study -- which is the failure the clamp removed earlier in this file was
    # guarding against, now that an arm is finally device-aware again.
    gpu_slots = 0
    if gpu_workers > 0:
        import torch as _t

        if not _t.cuda.is_available():
            print("[study] --gpu-workers requested but no CUDA device; using CPU only",
                  flush=True)
        else:
            free = _t.cuda.get_device_properties(0).total_memory
            # Measured from ``nvidia-smi``, not from ``max_memory_allocated``: that
            # reported 487 MB at batch 64 while the true per-worker footprint is
            # ~1.27 GB, because it counts neither the CUDA context (~0.4 GB) nor the
            # caching allocator's reserved-but-unallocated pool. Sizing from the
            # optimistic number put three workers on a 4 GB card and OOMed mid-study.
            per = 0.5e9 + 12.0e6 * batch_size
            fits = max(1, int(0.85 * free / per))
            if gpu_workers > fits:
                print(f"[study] {gpu_workers} GPU workers do not fit in "
                      f"{free/1e9:.1f} GB at batch {batch_size}; using {fits}", flush=True)
                gpu_workers = fits
            gpu_slots = gpu_workers
    jobs = [(arm, task_key, seed, steps, batch_size, d_model, gpu_slots > 0)
            for arm in arms for seed in range(seed_offset, seed_offset + seeds)]
    print(f"[study] up to {gpu_slots} concurrent runs on cuda, "
          f"{workers} workers total (claimed dynamically)", flush=True)
    print(f"[study] {len(jobs)} runs = {len(arms)} arms x {seeds} seeds, "
          f"{steps} steps @ batch {batch_size}, {workers} workers", flush=True)
    rows: List[Dict[str, object]] = []
    t0 = time.time()
    # **Spawn, not fork.** The clamp above calls ``resolve_device("auto")``, which
    # calls ``torch.cuda.is_available()``, which *initializes CUDA in the parent* --
    # and a forked child that then touches any accelerator API dies with "Cannot
    # re-initialize CUDA in forked subprocess". Under torch 2.14 `Adam.step` calls
    # `torch.accelerator.current_stream()` in a health check, so *every* worker dies
    # on the first optimiser step -- and at the time every arm was `device="cpu"`,
    # so not one of them even wanted the card.
    #
    # The clamp did not save it and could not: the failure is the fork, not the
    # context count. This went unseen because CI is CPU-only and the GPU work was
    # done from notebooks, so this file -- the tool the project's measurement
    # discipline rests on -- had never once been run on a machine with a card in it.
    # Spawn costs a fresh interpreter per worker, which against a 1000-step run is
    # nothing.
    ctx = multiprocessing.get_context("spawn")
    sem = ctx.Semaphore(gpu_slots) if gpu_slots else None
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_init_worker, initargs=(sem,)) as ex:
        for r in ex.map(_run_one, jobs):
            rows.append(r)
            extra = ""
            if "trace_probe" in r:
                extra = (f"  probe {r['trace_probe']:.3f}  collapse {r['collapse']:.3f}"
                         f"  truncated {r['truncated']:.3f}")
            print(f"  [{len(rows):2d}/{len(jobs)}] {r['arm']:>13} seed {r['seed']} "
                  f"acc {r['acc']:.3f}{extra}  ({r['secs']:.0f}s)", flush=True)
    if merge:
        with open(merge) as f:
            prior = json.load(f)["rows"]
        have = {(r["arm"], r["seed"]) for r in rows}
        merged = [r for r in prior if (r["arm"], r["seed"]) not in have]
        print(f"[study] merged {len(merged)} prior runs from {merge}", flush=True)
        rows = merged + rows
    summary = summarise(rows)
    report = _fmt(summary, spec, steps, batch_size)
    print("\n" + report, flush=True)
    print(f"\n[study] wall clock {(time.time() - t0) / 60:.1f} min", flush=True)
    blob = {"task": task_key, "spec": spec.__dict__, "steps": steps,
            "batch_size": batch_size, "d_model": d_model, "seeds": seeds,
            "rows": rows, "summary": summary}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"[study] wrote {out_path}", flush=True)
    return blob


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Paired multi-seed study of the Stage A arms")
    p.add_argument("--task", choices=sorted(TASKS), default="d2g1")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu-workers", type=int, default=0,
                   help="how many of the workers run on the GPU; the rest run on CPU "
                        "concurrently, so throughput is the sum. Capped by VRAM.")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ALL_ARMS),
                   help="default: the four established arms; lotus-space is opt-in")
    p.add_argument("--out", type=str, default=None, help="write the full result as JSON")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="start the seed range here, to extend an earlier study")
    p.add_argument("--merge", type=str, default=None,
                   help="an earlier study's JSON whose runs to fold into the summary")
    a = p.parse_args(argv)
    run_study(a.task, a.seeds, a.steps, a.batch_size, a.d_model, a.workers, a.arms,
              a.gpu_workers, a.out, a.seed_offset, a.merge)


if __name__ == "__main__":
    main()
