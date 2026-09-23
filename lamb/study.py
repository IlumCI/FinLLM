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
ALL_ARMS = ARMS + EXTRA_ARMS


# -- one run --------------------------------------------------------------
def _run_one(job: Tuple[str, str, int, int, int, int]) -> Dict[str, object]:
    """Train one (arm, seed) and return its held-out accuracy.

    Runs in its own process with a single torch thread: the arms are embarrassingly
    parallel across seeds, and one thread each beats N threads contending.
    """
    arm, task_key, seed, steps, batch_size, d_model = job
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
    out: Dict[str, object] = {"arm": arm, "task": task_key, "seed": seed}

    if arm.startswith("coconut"):
        if arm == "coconut-long":
            steps = int(round(steps * COMPUTE_MATCH))
        cfg = CoconutConfig(steps=steps, batch_size=batch_size, seed=seed,
                            depth=spec.depth, digits=spec.digits, ops_key=spec.ops_key,
                            device="cpu")
        tr = CoconutTrainer(cfg, tok, mcfg)
        for s in range(steps):
            tr._train_step(s)
        # Coconut's budget is K sequential thoughts; K=n_thoughts is its own best
        # setting and costs K+1 forwards, matching the LOTUS loops+1.
        out["acc"] = tr.greedy_accuracy(cfg.n_thoughts, 512)
        out["acc_k0"] = tr.greedy_accuracy(0, 512)
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
                          use_boundaries=False, switch_coef=0.0, device="cpu")
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

    for a, b in (("lotus-answer", "coconut"), ("lotus-trace", "lotus-answer"),
                 ("lotus-trace", "coconut"),
                 # the confound controls: same wall clock, not just same step count
                 ("coconut-long", "coconut"), ("lotus-answer", "coconut-long"),
                 ("lotus-trace", "coconut-long"),
                 # the untested second supervision dimension, and the ALU
                 ("lotus-space", "lotus-trace"), ("lotus-alu", "lotus-trace")):
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
              out_path: Optional[str] = None, seed_offset: int = 0,
              merge: Optional[str] = None) -> Dict[str, object]:
    """Run ``arms`` at ``seeds`` seeds and summarise.

    ``seed_offset`` starts the seed range past an earlier run's, and ``merge``
    folds in that run's rows, so a study can be *extended* to more seeds without
    recomputing the ones already paid for. That matters because the seed count
    needed is not known until the first few runs reveal the spread.
    """
    spec = TASKS[task_key]
    if merge and not os.path.exists(merge):
        # Validate before spending the compute, not after. Discovering a bad path
        # in the summary step throws away every run that preceded it.
        raise FileNotFoundError(f"--merge file does not exist: {merge}")
    jobs = [(arm, task_key, seed, steps, batch_size, d_model)
            for arm in arms for seed in range(seed_offset, seed_offset + seeds)]
    print(f"[study] {len(jobs)} runs = {len(arms)} arms x {seeds} seeds, "
          f"{steps} steps @ batch {batch_size}, {workers} workers", flush=True)
    rows: List[Dict[str, object]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
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
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ALL_ARMS),
                   help="default: the four established arms; lotus-space is opt-in")
    p.add_argument("--out", type=str, default=None, help="write the full result as JSON")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="start the seed range here, to extend an earlier study")
    p.add_argument("--merge", type=str, default=None,
                   help="an earlier study's JSON whose runs to fold into the summary")
    a = p.parse_args(argv)
    run_study(a.task, a.seeds, a.steps, a.batch_size, a.d_model, a.workers, a.arms,
              a.out, a.seed_offset, a.merge)


if __name__ == "__main__":
    main()
