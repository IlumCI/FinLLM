# LAMb roadmap

v0.1 is a faithful, CPU-runnable scaffold of the full architecture. The
extensions below are ordered roughly by leverage. Consult arXiv for the latest
work before implementing each — several of these are active 2025–2026 areas.

## 1. GRPO-trained hypernetwork proposer (implemented)

Implemented in `lamb/selfplay/hyperproposer.py` (`proposer_kind="grpo_hyper"`): a
hypernetwork maps solver competence to task-distribution logits, trained by GRPO
with learnability rewards and a **KL anchor to the bandit** (the anti-collapse
mechanism). The bandit stays the recommended default; the hypernetwork is the
drop-in for large task spaces.

Remaining upgrades:

- **Heavier hypernetwork.** Emit the *weights* of a small task-generator network
  rather than the distribution parameters directly — needed once the task space
  is continuous/compositional rather than a fixed grid.
- **Richer context.** Feed recent success slope and memory-utilisation stats, not
  just per-cell success, so the policy anticipates the frontier.
- **Mixture annealing.** Anneal the coverage floor `eps` down as the policy
  earns trust.

Refs: GRPO (DeepSeekMath, [2402.03300](https://arxiv.org/abs/2402.03300));
HypeRL ([2501.04538](https://arxiv.org/abs/2501.04538));
Absolute Zero ([2505.03335](https://arxiv.org/abs/2505.03335)); R-Zero
([2508.05004](https://arxiv.org/pdf/2508.05004)).

## 2. GRPO / RLVR on the solver (implemented)

Implemented (`solver_algo="grpo"`): sampled-group answers scored by the exact
verifier, group-relative advantages, KL to a refreshed reference, added on top of
expert iteration (a warm start that avoids the RL cold-start). Includes DAPO
dynamic sampling and an optional Dr.GRPO no-std normalization.

Remaining: replace the SFT warm start with a scheduled hand-off; GSPO
sequence-level ratios; clip-higher (DAPO) once multiple epochs per rollout are
used. Refs: DAPO, Dr.GRPO, GSPO.

## 2b. Benchmark harness

Build out the battery in [`BENCHMARKS.md`](BENCHMARKS.md): a length-generalization
eval ships in `lamb.eval.length_generalization`; add long-context recall
(needle / passkey / BABILong-style) and a latent-reasoning task (ProntoQA/ProsQA)
adapter. General-LLM suites (MMLU) are out of scope until a language front-end
exists.

## 2c. Red Queen coevolution (open-endedness) — the active direction

The reactive autocurriculum saturates on a bounded task space (learnability -> 0
once everything is mastered). Unbounded self-improvement needs *relative* fitness:
the solver must keep dominating its own past on an ever-advancing frontier.

- **Step 1 (implemented, `red_queen=True`).** Solver league (historical
  self-play), a novelty/diversity term on task selection, and relative-fitness
  metrics (`dominance`, `forgetting`) in `lamb/selfplay/league.py`. On the bounded
  grid, `dominance` starts positive and decays to 0 as the space saturates — the
  measured signal that motivates Step 2.
- **Step 2 (implemented, `open_ended=True`).** A **generative grammar** of nested
  expressions (`lamb/selfplay/grammar.py`) with an append-only descriptor space
  grown by **minimal-criterion admission** (`openended.py`), and a **factored
  hypernetwork proposer** (`factoredhyper.py`) that generates *in* that space with
  fixed `D+G+O` outputs. Measured: the space grows and the frontier advances with
  zero forgetting; `dominance` is now bounded by solver capacity rather than
  task-space saturation (the ceiling moved from the curriculum to the model), so
  sustained positive dominance is a scale-up result. This subsumes item 5 below.
  Refs: POET ([1901.01753](https://arxiv.org/abs/1901.01753)); Minimal Criterion
  Coevolution (Brant & Stanley).
- **Step 3 (implemented, `python -m lamb.poet`).** A **POET-style population** of
  (environment, agent) pairs (`lamb/poet.py`): per-environment specialist solvers,
  agent **transfer** across environments, minimal-criterion + novelty
  **reproduction** of harder environments, and graduation of easy ones. This is
  the capacity-scaling mechanism -- specialists + transfer raise the ceiling Step 2
  hit -- so it doubles as roadmap item 7. Measured: the population grows, its
  attempted frontier advances, transfers fire, and it conquers base environments;
  the conquered frontier scales with per-agent budget and size. Refs: POET
  ([1901.01753](https://arxiv.org/abs/1901.01753)); Enhanced POET
  ([2003.08536](https://arxiv.org/pdf/2003.08536)); Transfer Dynamics
  ([2203.10941](https://arxiv.org/pdf/2203.10941)).
- **Shared-backbone POET (implemented, `python -m lamb.poet_shared`).** One shared
  backbone plus a tiny low-rank adapter per environment (`lamb/poet_shared.py`): a
  population of N environments costs one backbone + N adapters (~21% of the
  parameters of N full models at capacity 5), and the backbone accumulates
  cross-environment skill for implicit transfer. Measured: it grows, transfers, and
  conquers the base frontier, at lower absolute mastery than independent agents --
  the honest efficiency/capacity tradeoff of a final-layer adapter. **Deeper
  adapters** (`lamb/model/lora.py`) that adapt the core's attention linears are
  the default: **DoRA** (weight-decomposed, arXiv:2402.09353 -- magnitude/direction
  decoupling) and LoRA. At equal budget the best specialist reaches ~0.77 (DoRA)
  vs ~0.53 (LoRA) vs ~0.50 (final-hidden), for one magnitude vector per layer on
  top of LoRA -- DoRA is the clear winner, as in the paper. `--adapter-type
  {dora,lora,hidden}`.
- **Behavioural-novelty admission (implemented, `lamb/selfplay/novelty.py`).** Both
  POET variants gate reproduction on *behaviour*, not descriptor dedup: each
  candidate environment is characterised by its seed agent's competence across
  latent-step budgets, the answer length it demands, and how much extra thinking
  helps (a BC vector), and admitted only if it is far, in behaviour space, from an
  archive of admitted environments (novelty search). Measured: the population still
  grows and conquers while ~12 behaviourally-redundant candidates are rejected --
  real diversity maintenance. Refs: novelty search (Lehman & Stanley); POET.
  Remaining: GPU scale-up of the agents.

Refs: Digital Red Queen ([2601.03335](https://arxiv.org/abs/2601.03335));
PopuLoRA ([2605.16727](https://arxiv.org/pdf/2605.16727)); learnable information
gain ([2603.02218](https://arxiv.org/pdf/2603.02218)).

## 3. Continuous-thought decoding (Coconut) — Stage A implemented

Implemented in `lamb/coconut.py` (`python -m lamb.coconut`) and the Coconut path
on `LAMb` (`_roll_thoughts` / `coconut_logits` / `coconut_solve`). Where the
recurrent core thinks *vertically* (iterate a block at fixed positions), Coconut
adds *horizontal* latent thinking: insert `K` scratchpad positions between the
prompt and the answer whose input embedding is the model's own last hidden state,
fed straight back in continuous space (via `thought_norm` + a learned
`thought_marker`) and never decoded. Those thoughts join the attention context —
a working memory the answer reads. `K=0` reduces *exactly* to ordinary
answer-only teacher forcing (a regression test asserts it; RoPE is relative so the
left-padded prompt is unaffected).

Key result, and a departure from the literature. Coconut's published gains need a
curriculum that distils the thoughts from **language** CoT steps; its own
`w/o curriculum` ablation — our exact setting (feed the state back, supervise only
the answer) — underperforms even no-thoughts *in the language domain*, and
arXiv:2510.12167 traces that to geometrically homogeneous latents plus no reliable
way to **select** a good trajectory. Here, with a number-native substrate and
genuinely multi-step tasks, plain answer-NLL through a randomly sized thought
chain **does** make the scratchpad useful: on depth-2 nested expressions greedy
exact-match rises `K=0: 0.39 → K=3: 0.52` (a tiny 0.28M-param CPU model), and the
latent-collapse diagnostic (`collapse_metric`, the 2510.12167 homogeneity signal)
falls from cos≈0.88 to cos≈0.27 as the thoughts specialise.

The verifier is LAMb's asset the 2510.12167 line lacked. Perturbing the latent
phase with dropout draws diverse trajectories and the exact verifier **selects**
the winner (best-of-N), turning their monotone-but-unusable Pass@N into *realised*
accuracy: `N=1: 0.52 → N=8: 0.66` — a test-time self-improvement loop (search the
latent space, verify, keep the winner) that needs no labels at deploy. Hard
STaR-filtering to the verifier-solved subset was tried and **rejected**: on a tiny
model it starves the gradient and collapses the thoughts to a constant, and since
the grammar already supplies exact labels, teacher forcing already carries the
verifier's information — so the verifier's unique leverage is at inference.

Refs: Coconut ([2412.06769](https://arxiv.org/abs/2412.06769)); inference-time
scaling for continuous-space reasoning ([2510.12167](https://arxiv.org/abs/2510.12167)).

### 3a. Restructured for scale: parallel supervised latents (LOTUS) — implemented

The Coconut path above works at tiny scale, and that is precisely its documented
limit. Measured across backbones ([2606.31779](https://arxiv.org/abs/2606.31779)),
the **sequential** continuous-thought family's gap to explicit chain-of-thought
*widens* with scale (−0.1 pts at 124M, −2.3 at 1B, **−9.2 at 3B** — a performance
cliff), while the **looped, parallel-supervised** family stays flat (−1.5 at 3B).
`lamb/lotus.py` (`python -m lamb.lotus`) is the restructure:

- **Parallel, not autoregressive.** `n_latent` latent positions are appended after
  the prompt at once and refined by `loops` passes through the shared core. Cost is
  `O(loops)` forwards **regardless of the latent count** (asserted in tests: 4 and
  64 latents both cost `loops+1` passes), where Coconut needed one sequential
  forward per thought. The latent budget grows for free.
- **Every latent position is supervised** through the LM head. LOTUS uses gold CoT
  tokens; LAMb has no language, so it generates a gold **numeric** trace — the
  intermediate sub-expression values — exactly and for free from its own evaluator
  (`TaskGrammar.sample_with_trace`). No language, no external data, no annotation.
- **Still a blackbox.** Latent positions are never decoded; only the answer is
  emitted. The trace is a training signal, not an output.

Measured on depth-2 nested expressions, 5 seeds per arm
(`python -m lamb.study --task d2g1`). **The baseline that matters is `coconut-long`**:
the same Coconut given the extra steps that make its *wall clock* equal to LOTUS's,
since LOTUS costs 0.564 s/step against Coconut's 0.335 and equal step counts quietly
hand it 1.68x the compute.

| arm | steps | answer acc (mean, n=5) | sd | range |
| --- | --- | --- | --- | --- |
| Coconut, equal steps | 1000 | 0.504 | 0.138 | 0.316–0.703 |
| **Coconut, equal wall clock** | **1684** | **0.836** | **0.022** | 0.809–0.865 |
| LOTUS, answer-only | 1000 | 0.826 | 0.076 | 0.746–0.910 |
| LOTUS, + per-position trace | 1000 | 0.903 | 0.054 | 0.811–0.951 |

Paired against the wall-clock-matched control:

| comparison | mean | sign-flip *p* | permutation *p* |
| --- | --- | --- | --- |
| LOTUS answer-only − Coconut | **−0.010** | 0.688 | **0.802** |
| LOTUS +trace − Coconut | **+0.067** | 0.0625 (floor) | **0.040** |

- **The structural claim is not reduced, it is absent.** Parallel supervised latents
  buy **nothing** over the sequential loop once compute is equalised: −1.0 points at
  *p* = 0.80, negative on three of five seeds. An earlier version of this section
  claimed **+32.3 pts** and called it the one result a 5-seed study could establish.
  That number was measured against a Coconut arm that had simply not finished
  training, and it is withdrawn.
- **The variance claim is withdrawn too.** This section previously argued that the
  restructure made training *reliable*, from Coconut's 39-point seed swing. That
  swing was **undertraining, not instability**: given equal wall clock the same arm
  has sd **0.022**, the tightest in the study — tighter than either LOTUS arm.
- **The power figure was inflated by the same bug.** "61 seeds to resolve 5 points"
  came from the undertrained arm's spread. Corrected, it is 19 seeds for
  `lotus-answer`, 10 for `lotus-trace`, 2 for `coconut-long`. The constraint is real
  and much milder than stated.
- **What survives is the trace supervision, and only that.** +6.7 points over a
  properly trained, compute-matched baseline, positive on all five seeds,
  permutation *p* = 0.0397. It is also the claim tied to what this project actually
  has that others do not: the gold trace is free here because an exact verifier
  generates it. The honest statement is not "the parallel architecture wins" but
  "the free exact supervision is worth about seven points."

The trace-probe (a diagnostic, never decoded) confirms the latent block carries the
intermediates: 0.97 with supervision against 0.14 at chance without. The `+trace`
arm uses information the control does not, so the comparison is not like-for-like —
it prices the supervision, not the structure.

One methodological note, since this is the second time it has mattered: the single
run figures originally reported here (0.512 / 0.695 / 0.945) were wrong for two
independent reasons, a 53%-contaminated eval set (3a-iii) and a baseline that had
not converged. Each was found only by building the control that could expose it.
A caveat left standing in a document is not a control.

Refs: LOTUS ([2606.31779](https://arxiv.org/abs/2606.31779)); SIM-CoT
([2509.20317](https://arxiv.org/abs/2509.20317)); looped transformers
([2502.17416](https://arxiv.org/abs/2502.17416)).

### 3a-i. SWITCH boundary tokens — implemented (entry only)

RL is the known dead end for latent reasoning: on-policy methods move a latent
model essentially not at all (arXiv:2512.11816 — GRPO takes explicit CoT 62.6 →
72.6 while latent goes 22.6 → 21.8), because a continuous segment has no
well-defined policy ratio to optimise. SWITCH
([2606.13106](https://arxiv.org/abs/2606.13106)) fixes that by making entry into
the latent segment a **predicted token**. Two new ids (`BOT`, `EOT`) are appended
after the digits, so every pre-existing id is unchanged and neither can ever appear
in an answer.

`LotusReasoner` emits `[prompt] [BOT] [latent x L]`. The last prompt position
predicts `BOT`, which gives the latent block a real log-probability and a fixed position for
probes to attach to.

**The accuracy claim for this did not survive.** It was first measured as
0.875 → 0.934 (+5.9) on a contaminated eval set. On the clean held-out partition the
comparison is 0.945 without the boundary vs 0.934 with it — the sign flips. The
multi-seed study in 3a-iv then explained *why* both numbers were meaningless — and
was itself corrected once a compute-matched baseline existed. The sharp version: the
boundary was never compared against a converged baseline, and at the spread of the
arm it *was* compared against, a single run could not have detected a real +5.9 nor
ruled one out. The honest reading is **no measurable effect, and no measurement**. Together with 3a-ii (its RL rationale falsified), the entry
boundary is therefore **off by default**: it costs a sequence position and two
vocabulary ids for nothing demonstrated. It is kept opt-in (`use_boundaries=True`)
because it remains the only well-defined attachment point for a probe or an RL
policy.

**Negative result, and the reason the exit marker is off by default.** Wrapping the
segment on *both* sides is actively harmful: a static `EOT` embedding sits between
the refined latents and the answer readout, so the first answer token gets predicted
from a generic marker instead of the latent state. Controlled at 500 steps:

| variant | answer acc | trace-probe |
| --- | --- | --- |
| no boundaries (control) | 0.371 | 0.697 |
| `BOT`+`EOT`, switch loss off | 0.176 | 0.468 |
| **`BOT` only** | **0.500** | **0.821** |

So the cost is structural, not the switch loss, and entry-only both fixes it and
beats the control. This matches the paper's own finding that the computation
concentrates at the *entry* transition; with a fixed latent budget the exit is
deterministic and needs no marker. `--exit-boundary` re-enables it for ablation.

**Monitorability: not established.** The boundary gives probes an attachment point,
and `boundary_probe` fits a class-balanced linear probe on the entry state to
predict whether the answer will be right — the practical reply to the
CoT-monitorability objection ([2507.11473](https://arxiv.org/abs/2507.11473)). But
it does not yet measure anything: at depth 2 the model is 93% correct, leaving too
few errors to fit against; at depth 3 it only reaches 9%, and the balanced sample
(24 held-out points) returns 0.333 vs a 0.5 baseline — noise. A real number needs a
regime where the model is competent *and* fallible (~50-70%), which this tiny model
on these tasks does not provide. The interface ships; the claim does not.

### 3a-ii. Does on-policy RL move the latent block? No — premise falsified

The boundary tokens in 3a-i were justified by SWITCH's claim that they unblock RL
over latent recurrence. `lamb/latent_rl.py` (`python -m lamb.latent_rl`) tests that
claim instead of assuming it.

Testing it honestly requires being precise about what RL could shape. Sampling
answer tokens has a well-defined log-probability with or without boundaries, but it
only shapes the *answer head* — never the latent computation. So the boundary has
to buy a **sampled action that changes the latent computation itself**: a switch
head on the boundary hidden state picks the latent budget (how many latent
positions stay active, applied by masking). It is sampled, has an exact
log-probability, and demonstrably changes the computation (asserted in tests).
Three arms run from one shared supervised checkpoint, same step budget, same eval:

| arm | acc | Δ from SFT start |
| --- | --- | --- |
| SFT start | 0.266 | — |
| + 300 more **supervised** steps | **0.645** | **+0.379** |
| + GRPO, switch action (boundary) | 0.262 | −0.004 |
| + GRPO, switch + entropy bonus | 0.301 | +0.035 |
| + GRPO, answer-only (the 2512.11816 setting) | 0.254 | −0.012 |

**RL is inert.** At best +0.035 where the same step budget spent supervised gives
+0.379 — an order of magnitude worse. This reproduces arXiv:2512.11816 (GRPO moved
a latent model 22.6 → 21.8) and, more importantly, the boundary tokens did **not**
rescue it: the switch arm is indistinguishable from answer-only. (These numbers are
from the contaminated-eval era; contamination would if anything *favour* the
supervised arm, and the gap is an order of magnitude, so the conclusion stands.)
Combined with the clean re-measurement in 3a-i — where the boundary's apparent
accuracy gain vanished — the entry boundary delivered nothing it was built for.

Observed mechanism, and an honest limit of this test: the switch policy collapsed
to a single budget (max) within ~100 steps, entropy 0.09 → 0.01, and an entropy
bonus did not prevent it. That collapse is partly **by construction** — with
budgets (2, 4, 8) and no cost on compute, "always max" is genuinely optimal, so
there is no allocation policy to discover. For RL to shape latent compute
allocation the reward must *price* compute. Other uncontrolled factors: one seed,
RL lr (2e-4) and KL coefficient not swept. What is solid is the size of the gap:
RL ≈ 0 against SFT +0.379 is far outside any noise band here.

Next, if this is revisited: a reward that charges for latent compute
(`correct − λ·budget`), so allocation is a real tradeoff rather than a dominated
choice. Until then, **do not plan on RL as the self-improvement mechanism for the
latent path** — the verifier-selected best-of-N search in 3 remains the mechanism
that measurably works.

### 3a-iii. Eval contamination — found and fixed

Every accuracy number above was originally measured against a *seed-separated*
held-out set, which turned out not to be held out at all. Disjoint seeds are not
disjoint problems: the depth-2/1-digit space has ~80k expressions and a 1000-step
run at batch 64 draws 64k of them, so **53% of the "held-out" set had been trained
on** — and the contamination grew with training length, biasing precisely the
longer-vs-shorter comparisons the project relies on. The Stage B comm task was
worse: a 200-problem space, trained on in full, so its eval was **100%** seen.

`lamb/holdout.py` fixes this structurally. Membership is decided by a hash of the
problem string, so a problem is permanently either train or eval, independent of
seeds and of how long training runs; training samplers reject the eval partition and
eval sets draw only from it. `tests/test_holdout.py` asserts zero overlap for the
grammar, Coconut, LOTUS and comm streams.

Effect on the results: the LOTUS restructure held up almost unchanged
(0.520→0.512, 0.707→0.695) and the trace arm improved (0.875→0.945) — a 0.28M model
on 44k problems cannot memorise much, so it was largely generalising already. The
boundary claim did **not** hold up (3a-i).

**The first pass at this fix was incomplete**, which is worth recording because the
incompleteness was invisible from the tests that were written for it. Only the
Stage A and Stage B trainers were wired to the partition. A later sweep of every
problem-sampling call site found four more that still drew from the whole space:

- `lamb/comm_pop.py` — the population trainer *and* its eval set. It holds out
  *pairings* `(i, i)`, which is the zero-shot-coordination question, and that was
  mistaken for holding out problems. The partner-randomization result was measured
  on problems the population had trained on.
- `lamb/latent_rl.py` — the GRPO rollouts drew the arms' own evaluation problems.
  Harmless to the conclusion there (it was negative, and contamination biases
  *upward*), but wrong.
- `lamb/selfplay/loop.py` and `lamb/poet.py` / `lamb/poet_shared.py` — the
  self-play and POET training samplers, which back the self-improvement claims.
  POET's `_score` is worse than a misreported number: POET *selects* on it, so
  scoring on trained problems was steering the search toward memorisation. It now
  scores on held-out problems.
- `lamb/eval.py` — the benchmark harness itself, behind the length-generalization
  numbers, sampled from the whole space.

The scope of the lesson is worth stating too, because the obvious over-correction
is to distrust seed separation everywhere. It was checked: the needle-retrieval and
RULER generators draw episodes from a space of ~10<sup>92</sup>, so two
seed-separated streams have an expected collision count of ~10<sup>-85</sup>, and a
measured overlap of exactly zero (`tests/test_holdout.py`). Seed separation is fine
there and those benchmarks needed no change. The problem was never seed separation
as such — it was seed separation over an **80k-expression space that a single run
covers 80% of**.

`TaskGrammar.sample_heldout` is the mirror of `sample(..., exclude_heldout=True)`,
so training rejects a partition that evaluation draws only from, and the two can
never meet. The lesson is about the shape of the fix rather than the fix: a
partition is only as good as its *least* careful call site, and a test that asserts
"these two streams are disjoint" says nothing about the streams nobody thought of.
The sweep is `grep` over every `sample(` call site, and it is cheap to repeat.

One cost is worth being explicit about, because it is a real trade and not a free
win. Partitioning a *small* space buys cleanliness at the price of resolution. The
fixed grid's 1-digit `+` cell holds exactly 100 problems, so its evaluation
partition is **9**: that per-cell accuracy now has a granularity of ~11 points
however many samples are drawn. `evaluate()` therefore returns `per_cell_unique`
alongside `per_cell`, so the resolution is visible in the output rather than
something the reader has to reconstruct. The answer to a coarse cell is a wider
cell, not a dirtier split.

For Stage B the partition is real but small — exactly 22 of 200 problems at 1 digit
(`CommTask.unique_count`), so *any* comm accuracy there has a granularity of ~4.5
points regardless of how many samples are drawn. The comm accuracy is best read as a
*channel* test — the listener never sees `X`, so it cannot answer from memorisation
without the message — rather than a generalisation test. `--a-digits 2` gives 2431
held-out problems if a generalisation claim is wanted.

### 3a-iv. How many seeds a claim needs — the answer is not one

The contamination above was a bug. The deeper problem was that **every number in
this repo was a single run**, and at this scale a single run cannot separate an
effect from seed noise. `lamb/study.py` runs each arm at several seeds and reports
the paired difference with an exact permutation test rather than an interval, since
with 5 seeds a normality assumption does more work than the data can support.

What it found on the original setting (`--task d2g1`, 5 seeds, 1000 steps):

| arm | steps | mean | sd | range |
| --- | --- | --- | --- | --- |
| Coconut, equal steps | 1000 | 0.504 | 0.138 | 0.316–0.703 |
| **Coconut, equal wall clock** | **1684** | **0.836** | **0.022** | 0.809–0.865 |
| LOTUS answer-only | 1000 | 0.826 | 0.076 | 0.746–0.910 |
| LOTUS + trace | 1000 | 0.903 | 0.054 | 0.811–0.951 |

This section originally reported only the first, third and fourth rows and drew
three conclusions from them. The `coconut-long` control was listed at the bottom as
a known confound to be closed later. Running it retracted two of the three:

- **Structure: withdrawn.** Against the compute-matched control, LOTUS answer-only
  is **−0.010** at permutation *p* = **0.802** — no effect at all, negative on three
  of five seeds. It had been reported as **+32.3, established**, with disjoint seed
  ranges and *p* = 0.0079. All of that was an artefact of comparing against a
  Coconut arm that had not finished training.
- **"Variance falls with each change": withdrawn.** The 39-point Coconut swing that
  this section built its methodological argument on was **undertraining**. Given
  equal wall clock, sd drops from 0.138 to **0.022** — the tightest arm in the
  study, tighter than either LOTUS arm. The restructure does not make training more
  reliable; finishing training does.
- **The power figure was inflated by the same bug.** "~61 seeds to resolve 5 points"
  used the undertrained arm's spread. Corrected: 19 seeds for `lotus-answer`, 10 for
  `lotus-trace`, **2** for `coconut-long`. The constraint on small claims is real
  but far milder, and the sharper statement about the SWITCH boundary (3a-i) is not
  that it needed 61 seeds but that it was never compared against a converged
  baseline either.
- **Trace supervision: survives, and is now the only surviving claim.** +0.067 over
  the compute-matched control, positive on all five seeds, permutation *p* =
  **0.0397**. It uses information the control does not — the gold trace — so it
  prices *the supervision*, not the architecture. That is a narrower claim than the
  one this section used to make, and it is the one attached to what this project
  actually has that others do not.

The spread is training variance rather than measurement noise: each arm is scored on
512 held-out problems, so binomial standard error at these accuracies is ~1.8 points.
Arms at the same seed share an eval set, so that component cancels from the paired
differences entirely.

The lesson is not about latent reasoning. **A confound recorded honestly in a
document is not a control**, and the gap between writing "this is a known confound"
and running the 50 minutes of compute that closes it was two wrong headline claims
and a methodological argument built on a measurement artefact.

### 3a-v. Latent budget decoupled from trace length; space supervision added

Two design gaps that the multi-seed work made worth closing, both taken from
mid-2026 results rather than from this repo's own logic:

- **The latent block was a copy of the trace.** One latent position per trace token
  means the block has to grow with the trace, so a long trace is simply out of
  reach — and [2607.16972](https://arxiv.org/abs/2607.16972) finds *both*
  continuous-CoT training regimes collapsing to about a third of explicit-CoT
  accuracy precisely on long traces. `trace_compress = c` supervises `c` consecutive
  trace tokens per latent through multi-token-prediction heads
  ([2404.19737](https://arxiv.org/abs/2404.19737)), so capacity is `n_latent * c`
  and the two are decoupled. The compression is **generative, not geometric**: C-MTP
  compresses by averaging the token embeddings a latent stands for, and
  [2606.20075](https://arxiv.org/abs/2606.20075) finds that rigid geometric
  compression collapses the reasoning space while generative reconstruction
  preserves its capacity — so each head decodes its own token and nothing is
  averaged. `c = 1` builds no extra heads at all, so it is bit-for-bit the previous
  model.
- **There was trajectory supervision but no space supervision.**
  [2606.20075](https://arxiv.org/abs/2606.20075) decomposes process supervision into
  dense stepwise signal (which the trace loss supplies) and preservation of the
  latent manifold's structure (which nothing here supplied), and attributes latent
  drift to the missing second term. `space_coef` adds a supervised-contrastive term
  ([2004.11362](https://arxiv.org/abs/2004.11362)) over latent positions carrying
  the same intermediate value. It is deliberately *relational*: pinning each latent
  to a fixed embedding of its value would be exactly the rigid constraint that
  analysis warns collapses the space.

Both default to **off**. They are arms to be measured on the harness in 3a-iv, not
claims — which is the discipline the boundary saga and the RL arm should have had.

### 3a-vi. The latent ALU: the latents hold values and the algebra does the arithmetic

Every latent-reasoning method supervises latents against **tokens** — LOTUS against
gold CoT tokens, SIM-CoT against a decoder's, C-MTP against averaged embeddings, and
3a above against the digits of LAMb's own trace. A token target teaches a latent to
*name* a number; it says nothing about what numbers *are*. So composition — which is
what reasoning consists of — has to be relearned by the network at every magnitude
it meets. That is the arithmetic length-generalization wall.

Models that *do* extrapolate turn out to have found a **periodic** representation by
themselves: tokens as phases, addition as rotation
([2511.23443](https://arxiv.org/abs/2511.23443) proves the benefit for modular
addition; [2606.17399](https://arxiv.org/abs/2606.17399) finds multiplication reduced
to addition in discrete-log space by the same mechanism). The usual response is to
bake periodicity into the **positional** encoding — Abacus
([2405.17399](https://arxiv.org/abs/2405.17399), already in this model) and position
coupling ([2405.20671](https://arxiv.org/abs/2405.20671)) — which *helps the network
find* the algorithm. The network still has to run it.

LAMb is in a position a language model is not: an exact evaluator hands it the true
value of every intermediate of every problem, free and unannotated. That is enough to
skip the discovery and hand the latent space the algebra outright. `lamb/algebra.py`
and `lamb/alu.py`:

- a latent slot carries one **value**, coded as its residues modulo a set of coprime
  moduli (a residue number system);
- composition is exact and carry-free — `(a+b) mod p` is a cyclic convolution of the
  residue distributions, `(a−b) mod p` a cross-correlation, `(a·b) mod p` one small
  table — and differentiable, so a model *uncertain* about a residue composes its
  uncertainty rather than having to commit first;
- the answer is built **bottom-up from the leaf slots by the algebra**, not read out
  by the network. The network is only ever asked for a leaf: two literals, one
  operator, whatever the expression's depth.

RNS is classical, but in neural networks it has only ever been used for **hardware
efficiency** ([1712.04614](https://arxiv.org/abs/1712.04614),
[2306.09481](https://arxiv.org/abs/2306.09481),
[2408.05639](https://arxiv.org/abs/2408.05639)). Using it as the *reasoning*
representation, supervised from a free exact trace, is the new part.

**The moduli are chosen for the order of 10, not for size.** The network produces
`n mod p` from digits as `Σ dᵢ·(10ⁱ mod p)`, whose coefficients repeat every
`ord_p(10)` positions — and that period is exactly how many digit positions a run
must *see* before the modulus is learnable, with every wider number free afterwards.
Small primes are a trap: `ord(10)` is 16 mod 17, 18 mod 19, 22 mod 23. The default
`(2,5,9,11,7,13,37)` has max period 6 in 84 units, strictly dominating the obvious
`(7,11,13,17,19,23)` (90 units, period 22). Three of them are the schoolbook
divisibility rules — mod 9 the digit sum, mod 11 the alternating sum, mod 2 and 5 the
last digit — which extrapolate at any width immediately.

**What holds without any training** (`tests/test_algebra.py`, `tests/test_alu.py`):
exact round-trip over the whole signed ring; exact composition of `+`, `−`, `×` at
every magnitude in range; and, given correct leaf codes, an exact answer at **depth 6
— a 64-operand expression — with nothing learned about depth**.

**What this does *not* claim, stated plainly:**

- *"Depth is free" applies to the composition, not to end-to-end accuracy.* The leaf
  count grows as `2^(D−1)` and every leaf needs all its residues right, so accuracy
  falls roughly as `leaf_residue_acc ^ (n_moduli · n_leaves)`. CRT has no locality: one
  wrong residue is a wildly wrong answer, not a near one. Redundant moduli are the
  classical fix and would double as a second label-free error signal; not yet built.
- *The slot addressing does not extrapolate the way the algebra does.* Training at
  depth 2 only ever writes slots 0–1, so slot 6's embedding is untrained and a
  depth-4 evaluation asks the model to use slots it has never used. Depth transfer
  therefore needs mixed-depth training; it is not free.
- *The expression tree is read from the input.* That is legitimate here — the tree is
  the question, not the answer — but a natural-language problem does not come with
  one. The bridge needs the model to **emit** the structure, at which point the slots
  become registers and the selectors become instructions: a differentiable register
  machine over the algebra. The gold program is free from the same generator, so it
  can be supervised before being relaxed — which is how to avoid the instability that
  sank earlier neural program induction.
- *The consistency check is not usable yet.* Measured at 400 steps, leaf-residue
  accuracy is **0.985** while the root slot sits at **0.229**, barely above chance:
  the model specialises on leaves exactly as intended, because composition is no
  longer its job, so it never learns to state the answer. A check that compares a
  near-chance direct prediction against a good composed one carries little signal.
  The term is implemented (`alu_consistency_coef`) and **off by default** — training
  on it invites the model to satisfy agreement by routing rather than by being right,
  which makes it worth more as a measurement than as a loss until it has been
  measured.

**Pre-registered test.** Train on operand widths 1–4, evaluate at width 5. The moduli
`(16,25,27,11,37)` have coefficient patterns fully determined by digit position 3, so
widths 1–4 see every pattern and width 5 is pure periodicity. Prediction: the ALU
holds and the token-trace baseline falls. If the ALU does not hold at width 5, the
residue representation is not buying what it claims and this is mostly dead.

**Result: the pre-registered test did not run, and that is the honest verdict.**
1200 steps, batch 64, 0.28M params, trained on widths 1-4:

| operand width | in training? | ALU (algebraic) | ALU leaf-residue | token-trace baseline |
| --- | --- | --- | --- | --- |
| 1 | yes | **0.992** | 0.998 | 0.133 |
| 2 | yes | 0.000 | 0.054 (chance) | 0.012 |
| 3 | yes | 0.000 | 0.053 | 0.000 |
| 4 | yes | 0.000 | 0.059 | 0.000 |
| 5 | **no** | 0.000 | 0.053 | 0.000 |

Widths 2-4 were *in the training distribution* and sit at chance, for **both** arms
— the baseline scored 0.133 at width 1 here against 0.903 in the single-width study
of 3a-iv, so mixed-width training at this budget broke it too. Extrapolation cannot
be measured from a model that failed in-distribution, so the pre-registration's
"then it is mostly dead" does not apply: its precondition failed. What does survive
is narrower and real — at the one width both arms could learn, on the same budget
and the same information, the ALU reached **0.992 against the baseline's 0.133**.

**The failure is a cliff, not a slope, and it exposes a design error.** At width 1
the leaf values are at most 18, so `n mod p` is nearly the identity for moduli
16/25/27/37 and almost no arithmetic is required; from width 2 the network must
actually compute `n mod p` from digits. But **that is a known fixed linear function**
— `Σ dᵢ·(10ⁱ mod p)` — and the digits are in the prompt. Asking a network to induce
an exact function it could simply be given is a waste of capacity, not a test of the
idea. Measured: a *fixed* digit→residue encoder plus algebraic composition is exact
at depth 2 width 8, depth 3 width 4, depth 4 width 3 and depth 5 width 2, with
nothing learned anywhere.

Confirmed by a follow-up: training on width 2 **alone**, with the whole budget on
one width, leaves leaf-residue accuracy flat at chance for 1200 steps (0.050, 0.055,
0.047, 0.047 at steps 300/600/900/1200, every modulus at chance). It does not learn
slowly; it never starts. So the fixed encoder is not an optimisation, it is required.

That reframes what the ALU is for, and the reframing is not a consolation. With a
fixed encoder and a known structure, LAMb+ALU on synthetic arithmetic **degenerates
to an exact calculator** — correct at any width and depth, and evidence of nothing,
because no learning is involved. The corollary has to be stated plainly: *synthetic
arithmetic is the wrong benchmark for this component.* Its value is that arithmetic
stops consuming capacity and stops being an error source, which only shows up where
arithmetic is incidental and comprehension is the bottleneck. Validating it
therefore has to wait for the bridge, which is a real deferral and not a result.

**One thing did land, and it replaces the broken part.** The root-slot consistency
check of 3a-vi is weak because the model never learns the root. Redundant residues
give the same signal without it: size the moduli so legitimate values occupy a
sub-range of the ring, and a single corrupted residue throws the CRT reconstruction
outside that range. Measured over 20,000 trials with core `(16,25,27,11)` and
redundant `(37,7)`: **100% of single-residue corruptions detected, 0 false alarms**
(theoretical rate 99.61%). No labels, no root slot, no training — a pure range check
on the reconstruction. That is the label-free correctness signal the project needs in
order to survive leaving the distribution its verifier was written for.


### 3a-vii. The program can be induced from the answer alone

The register machine (`lamb/regmachine.py`) turns the latents into registers and has
the model emit a *program* over them — an operation and two pointers per step — which
the residue algebra executes. Registers are append-only, so the dataflow is a DAG and
a pointer mask makes reading an unwritten register unrepresentable rather than merely
penalised.

The question that mattered, pre-registered before running: **is the differentiable
executor a capability or a convenience?** PAL and Program-of-Thought call an external
Python interpreter, so a wrong answer cannot tell a pointer which way to move and the
program can only be imitated or reinforced — and RL on this latent block was measured
inert (3a-ii). Here the executor is exact algebra and differentiable, so the answer
loss reaches the pointer heads *through the arithmetic*. Two arms, depth 2:

| arm | `program_coef` | answer acc | canonical acc |
| --- | --- | --- | --- |
| supervised | 1.0 | 1.000 | 1.000 |
| **answer-only** | **0.0** | **1.000** | **0.000** |

With the gold program removed entirely, the model reaches **perfect held-out answer
accuracy** while matching the generator's program on *zero* instructions. That is not
a contradiction, and the diagnostic that looked like a failure is the one that
explains it: **the program computing a function is not unique.** Exhaustively, 48
distinct three-instruction programs compute `(a+b)+(c+d)` exactly — commutativity and
re-association — of which the grammar emits one. A model inducing any of the other 47
scores 0.000 against the canonical form while being completely correct.

So `canonical_acc` measures *conformity to the generator's form*, and `answer_acc` is
the correctness measure. Held-out answer accuracy of 1.000 with a non-canonical
program is the strong version of the result: the program generalises, so it is
computing the right function rather than memorising a mapping.

**Gradients through exact arithmetic are sufficient to induce a correct program from
outcomes alone.** That is the one thing in this design that a non-differentiable
executor cannot do, and it is the mechanism the language bridge depends on — a word
problem does not come with a gold program either.

Caveats, since the arms are n=1 at the smallest depth: three instructions over four
operands is a small program space, `op_acc` at 0.603 says the answer-only arm's
operator choices are partly canonical and partly not, and depth 3+ has not been run.
The claim established is that outcome-only induction *works here*, not that it scales.

### 3a-viii. Division and decimals: exact rationals in the ring

Checking what the algebra could actually express for a grade-school word problem
turned up two gaps, both fatal rather than inconvenient, and one of them silent.

**Division was absent and unavailable.** In a residue system you divide by
multiplying with a modular inverse, which exists only for divisors coprime to every
modulus and gives the true quotient only when the division is exact. Worse, the
moduli had been chosen for short digit-periods (3a-vi) — powers of 2 and 5, and
divisors of `10^k − 1` — and that is *precisely* the set that makes small divisors
non-invertible. Under `(2,5,9,11,7,13,37)`, **not one divisor from 2 to 12 has an
inverse**, and "half as many" is the most common operation in GSM8K. The
optimisation that made the encoder extrapolate is the one that made division
impossible.

**Mixed scales were silently wrong.** `extract_quantities` returns `3.25` as
`(325, scale=2)` and `7` as `(7, scale=0)`; composing those gives `332`, i.e. 3.32.
The parser tracked scale and the algebra did not, so any problem mixing decimals
with integers produced a confident wrong number — the worst failure mode available.

Both dissolve under one change (`lamb/rational.py`): carry a value as a **pair**
`(numerator, denominator)`, each an ordinary residue vector.

| | |
| --- | --- |
| `a/b + c/d` | `(ad + cb) / bd` |
| `a/b − c/d` | `(ad − cb) / bd` |
| `a/b × c/d` | `ac / bd` |
| `a/b ÷ c/d` | **`ad / bc`** |

Division becomes multiplication with the operands swapped: exact, closed, requiring
no modular inverse, and still differentiable because every step is `+`, `−` or `×`
on distributions. Decimals stop needing scale bookkeeping entirely — `3.25` *is*
`325/100`, and a scale is just a denominator.

Verified against Python's own `Fraction`: **0 errors over 1200 random operations**
across all four operations, plus chains, `3.25 + 7 = 41/4`, and `7 ÷ 2 = 7/2`.

The cost is that a fraction cannot be reduced in residue form, so denominators grow
multiplicatively and that, not the design, sets the ring size. `RATIONAL_MODULI`
`(64,125,27,11,7,13,37,101,41,271)` gives ±4.5e15 with every digit-period still ≤ 6;
the worst chain a grade-school problem produces (five two-decimal values, denominator
1e10) has five orders of magnitude of headroom. `denominator_magnitude` exists so a
long chain approaching the ring is *observed* rather than discovered from a wrong
answer.

Two things this does not solve, recorded rather than left implicit:

- **Division by zero is undetectable in the ring.** A zero denominator is a legal
  residue vector; decoding raises rather than returning a number, so guarding it is
  the program's job. That is the right place for it — the emitted program is what
  knows whether a divisor could be zero — but it is now a thing the program must do.
- **The batched-over-moduli path wastes more here.** Padding to the widest modulus
  costs ~7× the arithmetic at `P=271`, against ~2× at `P=37`. On a GPU that is still
  the right trade, since the path is launch-bound by four orders of magnitude; on CPU
  it is not, and the per-modulus loop remains available.

### 3a-ix. Redundant residues: correcting errors, not just detecting them

The harshest constraint on the residue representation is that **CRT has no
locality**. One wrong residue does not give a nearby number, it gives an essentially
uniform one, so an answer's accuracy is roughly the per-residue accuracy raised to
the number of moduli. At the 0.985 per-residue accuracy measured in 3a-vi, seven
moduli give **0.900** — a 10% error rate produced entirely by the encoding, on top
of whatever the model gets wrong.

The classical fix costs width and nothing else. Size the **core** moduli so
legitimate values occupy only part of the ring and carry extra **redundant** ones.
A corrupted residue throws the reconstruction outside the legitimate range, so it is
detected; and because the survivors still over-determine the value, dropping each
modulus in turn identifies *which* residue was wrong and recovers the value exactly.

    q**K                      -> detection only
    q**K + K q**(K-1) (1-q)   -> single error corrected

At the measured `q = 0.985`, `K = 7`: **0.900 → 0.996**, a 25× reduction in error
rate. Nothing is learned and no label is required, which is what makes it usable at
inference on a benchmark with no verifier — the case an exact checker cannot reach.

Measured over 4,000 single-residue corruptions per row, core `(16,25,27,11)`:

| redundant moduli | units | corrected | refused | **mis-corrected** | 2-error silent |
| --- | --- | --- | --- | --- | --- |
| `(37,7)` | 123 | 84.3% | 15.7% | **0.00%** | 33.5% |
| `(37,7,41)` | 164 | **100.0%** | 0.0% | **0.00%** | 0.1% |
| `(37,7,41,101)` | 265 | 100.0% | 0.0% | **0.00%** | 0.0% |

Three redundant moduli is the sizing: full correction of single errors, and two
simultaneous errors go silent only 0.1% of the time.

**The column that matters is mis-correction, and it is zero throughout.** Under-
provisioned redundancy loses corrections — it never invents one. When the evidence
does not single out a culprit, `correct` returns `None` rather than picking the most
plausible candidate. That is deliberate: this representation is being used precisely
where nothing downstream can catch a wrong answer, so the failure mode has to be an
admitted failure rather than a confident number. It is the same reasoning that made
out-of-range values masked rather than clipped in 3a-vi.

## 3b. Latent inter-agent communication (Coconut) — Stage B implemented

Implemented in `lamb/comm.py` (`python -m lamb.comm`). The message passed between
two agents *is* a Coconut thought vector: the speaker rolls latent thoughts from
its half of a split problem, a differentiable `Channel` carries those continuous
vectors, and the listener consumes them as latent input positions (receiving =
thinking another agent's thought) and answers — reusing Stage A's exact
continuous-input path. No token is ever exchanged; the channel is a pure-latent
blackbox.

**Split task**: the speaker sees operand `X`, the listener sees `op` and `Y` and
must emit `X op Y` — impossible without the message. Training is one joint
objective (the listener's answer cross-entropy) back-propagated *through the
message* into the speaker: differentiable inter-agent learning (DIAL,
arXiv:1605.06676) with Coconut thoughts as the message — a combination the recent
latent-agent-communication papers (Interlat arXiv:2511.09149, DiffMAS
arXiv:2604.21794) approach but none train as the Coconut recurrence driven
end-to-end across the agent boundary.

Measured (tiny CPU pair, 1-digit `X,Y`, `+/−`, noiseless channel): exact-match
`comm 1.000` vs a zeroed-message ablation `0.098` (the guess-`X` prior) — a
`+0.90` causal channel gain (the positive-*listening* test of arXiv:1903.05168).
Two agents solve, purely in latent space, a task neither can solve alone. A
`channel_noise>0` (DIAL DRU) bandwidth sweep shows a sharp capacity threshold
(width `1/2/4` at the `~0.10` prior, `8/96 → 1.000`: a 1-digit operand needs `≥8`
noisy channel dimensions); message diagnostics (`signal_std`, `msg_cos`) guard
against the representational collapse of arXiv:2604.03809. `python -m lamb.comm --sweep`.

**Held-out-partner test** (`python -m lamb.comm_transfer`). Is the emergent code a
private co-adaptation or a shareable protocol (zero-shot coordination; Other-Play,
arXiv:2003.02979)? Two independent pairs each reach `comm 1.000`, but a zero-shot
speaker swap collapses to `0.004/0.041` — *below* the `0.076` blank prior (a
foreign code actively misleads); yet a freshly trained receiver learns a frozen
speaker's code to `1.000`. So the protocol is **private and strongly co-adapted
but learnable** — idiosyncratic, not canonical, exactly as Other-Play predicts.

**Partner randomization** (`python -m lamb.comm_pop`) is the fix, and it works. A
population of `P` speakers and `Q` listeners is trained with random pairing, the
diagonal `(i, i)` pairings held out of training and evaluated zero-shot. On a 3×3
CPU population the held-out (never-co-trained) pairings reach zero-shot `1.000`,
matching trained pairings and up from the single pair's `0.004` swap — the private
code becomes canonical. Other-Play realized in pure latent space.

**Re-measured on the clean split, and it holds.** `comm_pop.py` held out *pairings*
but not *problems* until the sweep in 3a-iii, so the original `1.000` was measured on
problems the population had trained on. Re-run with both held out (1600 steps, 3x3
population):

| | trained pairings | held-out zero-shot | blank |
| --- | --- | --- | --- |
| clean split | 0.941 | **0.935** (min 0.918, max 0.953) | 0.000–0.043 |

Zero-shot pairings perform **indistinguishably from trained ones** against a blank
prior at ~0. The headline number comes down from `1.000` to `0.935`, but the claim
it was making — partner randomization makes the latent code *canonical* rather than
private, so a never-co-trained pair coordinates as well as a trained one — is
exactly what survives, and this is now the only Stage A/B claim measured on an
uncontaminated split that did not shrink under scrutiny.

The 1-digit evaluation partition holds 22 problems, so this resolves to ~4.5 points;
`--a-digits 2` gives 2431 if a finer number is wanted.

Remaining: larger populations, heterogeneous agent sizes, and cross-architecture
transfer.

Refs: DIAL ([1605.06676](https://arxiv.org/abs/1605.06676)); pitfalls of measuring
emergent communication ([1903.05168](https://arxiv.org/abs/1903.05168));
zero-shot coordination / Other-Play ([2003.02979](https://arxiv.org/abs/2003.02979));
Interlat ([2511.09149](https://arxiv.org/abs/2511.09149)); DiffMAS
([2604.21794](https://arxiv.org/abs/2604.21794)).

## 4. Deeper test-time memory (ATLAS)

Upgrade the linear delta-rule memory to a small non-linear MLP memory with a
momentum ("surprise") term and Muon/second-order-style updates, targeting the
10M-token regime. Wire the Rust `TopKStore` as an exact retrieval tier alongside
the compressive neural memory.

Refs: Titans; ATLAS ([2505.23735](https://arxiv.org/abs/2505.23735)).

## 5. Richer task space

Multi-term nested expressions, division/modulo, mixed radices, rationals, and
eventually symbolic/algebraic goals — the combinatorial space where the
hypernetwork proposer (item 1) earns its keep. The Rust verifier already parses
general `+ − * ( )`; extend its grammar and the tokenizer in lockstep.

## 6. Number representation experiments

A/B the value channel against BitTokens (IEEE-754 as one token) and Adelic
operation-preserving embeddings; sweep digit base and reversal.

Refs: *Numbers Already Carry Their Own Embeddings* ([2606.14108](https://arxiv.org/html/2606.14108v1)); BitTokens.

## 7. Scale-up path (GPU + CPU + RAM hybrid) — implemented

`lamb/device.py` makes every entry point device-agnostic and mixed-precision-ready
with no architecture change. `resolve_device` auto-selects `cuda` > `mps` > `cpu`
(`--device`, `LAMB_DEVICE`); `Amp` turns on bf16/fp16 autocast (+ gradient scaler)
on CUDA automatically and stays fp32 on CPU so the CPU path never regresses; the
exact Rust kernels keep running on CPU alongside the accelerator and RAM holds the
buffers/stores; `--threads` gives the CPU legs every core. `--scale
{tiny,small,base,large}` grows width/depth/batch together (0.28M → 1.98M → 7.90M →
31.5M params). Wired into `train`, `coconut`, `comm`, `comm_pop`; covered by
`tests/test_device.py`. The GPU path is unit-tested on CPU (forced bf16 autocast);
it has not been run on a physical GPU in this repo (CI is CPU-only), but
`--device auto` picks CUDA up with no code change.

Remaining: multi-GPU (FSDP / tensor-parallel) for the `large`+ regime, a serving
mode, and `torch.compile` once shapes are static.
