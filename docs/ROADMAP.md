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
  sank earlier neural program induction. (The *tree* is indeed absent from prose, but
  GSM8K's calculator annotations supply the evaluation *chain*, which is what the
  register machine actually needs — see 3c. The gap was narrower than this bullet
  assumed.)
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
executor cannot do.

**Corrected in place:** this paragraph used to end *"and it is the mechanism the
language bridge depends on — a word problem does not come with a gold program
either."* The first clause stands; the second is **false for GSM8K**, whose train
solutions carry inline calculator annotations (`<<48/2=24>>`) that recover to a
program directly (3c). So the bridge does not have to rest on this result — it gets to
*test* it, with the supervised arm as a control. Which is better news for the bridge
and worse news for this section, because it removes the argument that an `n=1` result
had to be relied on rather than measured. The depth-3 criterion is pre-registered in
3a-xii.

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

### 3a-x. Division in the register machine

The rational layer (3a-viii) made division exact; this puts it in the machine's
instruction set. `RegisterMachine(rational=True)` holds each register as a
`(numerator, denominator)` pair and widens `OPS` from `(+,−,*)` to `(+,−,*,/)`,
which costs one extra output on the operation head and nothing else — division is
multiplication with the operands swapped, so no new primitive is introduced.

Verified end to end: `(48 ÷ 2) + 3.25 → 109/4`, exactly, through emitted
instructions — a division *and* a decimal, neither expressible on plain residues.
Gradients still reach the pointer and operation heads through the rational
arithmetic, so the outcome-only induction of 3a-vii is not given up to gain
division.

The integer path is left byte-for-byte as it was, since it carries the depth-2
result; `rational=True` is opt-in.

**One caveat, and the first version of this paragraph got it wrong.** I wrote that a
soft pointer read produces a *blend* of the registers it mixes. A test written to
demonstrate that falsified it: a 50/50 read over `1/2` and `3/4` decoded to exactly
`1/2`. Investigating found something worse. Decoding takes an argmax **per modulus,
independently**, so the winning residues need not come from the same register; when
they disagree the CRT lands somewhere unrelated to either input. Measured over random
pairs, **~30% of soft reads decode to a value in neither register**, and the failure
is *incoherence*, not averaging.

It is also not specific to rationals — the integer register file has it too, for the
same reason, which the original framing obscured.

Two things contain it, one of which was already built:

- **Sharp pointers are exact**, and training drives pointers sharp: both depth-2 arms
  reached fully determined pointers and scored 1.000 answer accuracy.
- **Redundant moduli catch the rest.** An incoherent residue vector is not a
  legitimate value, so the range check of 3a-ix flags it without caring what made it
  inconsistent. Measured: **100% of incoherent reads detected, none silently wrong.**
  The redundancy added for the model's own residue errors covers this too.

Training is unaffected either way, since the losses read distributions rather than
the argmax.

**Corrected in place.** This paragraph used to read *"the grammar generates `+`,
`−` and `*`, so there is no division training data yet"*. That was true when it was
written and stopped being true one commit later: `ops_key=2` generates `+ − * /`
with divisions made exact by construction. The accurate statement is that the data
exists and **the consumer did not** — see 3a-xi, which is the more interesting
failure.

**Still open:** a zero divisor remains undetectable in the ring (3a-viii), so
guarding it belongs to the emitted program — and 3a-xi turns that from a note into
a cost, because the natural answer-side loss for rationals has a degenerate zero at
exactly that point.

### 3a-xi. Four seams, and an invariant broken where nobody was looking

Wiring the division work of 3a-x to the trainer of 3a-vii turned up four gaps
between components that were each correct on their own. Recorded because the shape
repeats: every one of them sat in the *join*, and the tests all passed because each
test constructed its own inputs rather than taking the previous stage's output.

**1. Out-of-range values were not masked in the register-machine trainer.** The
project's own invariant is that a wrapped value is a *different* number, not a large
one, so an out-of-ring value is masked and never clipped; `lamb/lotus.py` has done
that since the ALU landed. `RegMachineTrainer._losses` did not, and
`ResidueSystem.targets` is an unguarded `int(v) % p`, so an out-of-ring answer became
a perfectly legal cross-entropy target *for the wrong number*. It never bit because
the only configuration ever run — depth 2, one digit, `(+,−)` — cannot leave the ring.
Depth 3 and multiplication leave it immediately, which is to say the bug was waiting
precisely in the direction the next measurement goes. The drop rate is now reported
by `train_step` and `evaluate` rather than applied silently: an accuracy measured on
60% of a held-out set is not the same number as one measured on all of it.

**2. The grammar's division output was unparseable.** `alu.parse_expr` scanned
`"+-*"`, so the leaf `48/2` raised; and `gold_program` defaulted to the integer
`OPS`, so a `/` node raised from `ops.index`. Both sit inside
`RegMachineTrainer._prepare`. So `ops_key=2` — data that generates perfectly — crashed
the only program trainer in the repo, while the ROADMAP recorded the *opposite*
problem. Every rational test hand-fed `Fraction` values through hand-built pointers,
and a test that constructs its own inputs cannot fail on a parser.

**3. `execute` read the module-level `OPS` rather than the machine's.** Harmless
while every machine had three operations; silently wrong the moment the operation
head is four wide, since the fourth logit would be trainable, selectable, and never
composed.

**4. The bridge threw the scale away one line after parsing it.**
`registers_from_quantities` returned `[q.value for q in qs]`, so `3.25` entered the
register file as `325` and `7` as `7`. That is the mixed-scale failure 3a-viii was
written to kill, reintroduced at the one seam where nothing downstream can catch it,
with the correct join (`RationalAlgebra.encode_scaled`) sitting unused two modules
away. It now returns `(values, scales, counts)`.

**5. `lamb/study.py` was broken on any machine with a GPU, and had never been run on
one.** The first attempt to run the arms above died instantly: every worker raised
*"Cannot re-initialize CUDA in forked subprocess"* on its first optimiser step. The
chain is worth spelling out because each link looked harmless.

The file opened with a clamp — if `resolve_device("auto")` reported CUDA, drop to one
worker, since one CUDA context per worker is ~300–500 MB and four exhaust a small
card. That clamp calls `torch.cuda.is_available()`, which **initialises CUDA in the
parent**, and `ProcessPoolExecutor` then *forks*. Under the pinned torch 2.14,
`Adam.step` runs a health check through `torch.accelerator.current_stream()`, so every
child dies — even though every arm in `_run_one` pins `device="cpu"` and no worker
ever wanted a card.

So the clamp caused the failure it was written to prevent, and clamping to one worker
did not help, because the problem was the fork and not the context count. Fixed by
using a **spawn** context; the clamp is removed outright, because the arms are
CPU-pinned by construction and the guard was protecting against a situation that
cannot arise. On this box it also cost a 4× slowdown for nothing.

It went unseen because CI is CPU-only and the GPU work was done from notebooks — so
**the tool this project's measurement discipline rests on had never once executed on
the hardware its own scale-up section points at.** That is a worse gap than any of
the four above: a broken trainer produces a wrong number, and a broken harness
produces none, but it also means none of the discipline was available on the hardware
where the interesting runs happen.

**A design cost that came out of the fix, stated rather than buried.** The integer
answer loss is cross-entropy of the executed result against the answer's residues.
That does not transfer to rationals: a fraction cannot be reduced in residue form, so
the pair a program builds for `109/4` is some unreduced `(N, D)`, and supervising `N`
against `109` supervises the wrong number. The criterion that *is* correct is the
cross-product residual `N·q − D·p = 0` — one target, zero, in every modulus, built
from the `+ − *` the ring already has.

It has a degenerate zero, and **the first version of this paragraph got its shape
wrong** — in the direction that overstates the problem. I wrote that a program reaches
`(N, D) = (0, 0)` by dividing through a register holding `0`. The test written to
demonstrate that falsified it: `5 / 0` gives `(5, 0)`, whose residual is
`5·1 − 0·7 = 5`, so the criterion *rejects* it at full cost. A zero divisor is
detected, not hidden.

Since `a/b ÷ c/d → (ad, bc)`, reaching `(0, 0)` needs a zero **numerator** as well as
a zero divisor — `0 ÷ 0`. Still reachable: one-digit operands include `0`, and a
pointer pair may address the same register twice. So `den_zero_coef` is load-bearing
for that one case rather than a general necessity, and the residual does more of the
work than the first draft credited it with. 3a-viii's *"division by zero is
undetectable in the ring"* met from the loss side — where the ring turns out to be
better at it than expected.

### 3a-xiv. Pre-registered: can the answer loss *hold* a program it cannot *find*?

3a-xii establishes that outcome-only induction fails at 7 instructions (0.015 against a
supervised 1.000) while the supervised arm is perfect at both depths. Both arms are
constant extremes, and `RegMachineTrainer`'s own docstring has described the intended
curriculum since it was written -- *"supervise the program first, lean on the answer
after"* -- which **no arm had ever run**. The gap between those extremes is where the
answer lies, and it is also where the language bridge actually sits.

Two arms, both at **depth 3**, since that is where outcome-only failed:

`regmachine-anneal` -- supervised for the first 20% of steps, gold program withdrawn
linearly to zero by 60%, answer-only for the last 40%.

> **Criterion.** Precondition: the arm reaches ~1.000 during the supervised phase (the
> supervised arm is 1.000 on 5/5, so a failure here means the schedule broke something
> and the test did not run). Then:
> - **`answer_acc` >= 0.9 after withdrawal** => the answer loss can *hold* a committed
>   program it could not *find*. The depth-3 failure is then a cold-start / exploration
>   problem, not a gradient-quality one -- and that is an actionable difference, because
>   cold starts have known fixes and bad gradients do not.
> - **collapse toward 0.015** => the outcome gradient cannot even maintain a correct
>   program at 7 instructions. That is a *stronger* negative than 3a-xii: it would mean
>   the answer loss is actively destructive at this width rather than merely
>   insufficient, and the differentiable executor's practical value would be confined
>   to very short programs.

`regmachine-partial42` -- a gold program for a hash-selected **42%** of problems and
nothing for the other 58%. Membership is decided per *problem* rather than sampled per
step, because that is how a dataset behaves -- per-step dropout would hand every problem
supervision eventually and measure something else.

> **Criterion.** **`answer_acc` >= 0.9** => 42% supervision suffices at 7 instructions.
> Below that, partial supervision does not carry the unsupervised remainder at this
> width, and the bridge needs either more coverage or a different mechanism for it.

**Corrected before the arm reported, not after.** The 42% was chosen because it was
GSM8K's measured program coverage when this was pre-registered. It is no longer: 3c's
compound decomposition and unit constants moved coverage to **0.680 train / 0.691 test**.
So the arm no longer *mirrors* the bridge -- it is a **lower bound** on it, which is the
more useful direction to be wrong in. If 42% suffices then 68% does with margin; if 42%
fails, a 68% arm is the follow-up rather than the conclusion.

The criterion itself is untouched, and the arm was already running when the mismatch was
noticed. Rewriting a threshold after seeing an outcome is the failure this project logs
at 3a-i and 3a-vi; rewriting a *rationale* before any number exists is just bookkeeping,
and the distinction is only meaningful if the correction is timestamped by the run it
precedes.

Both criteria are fixed before running. Note what is deliberately *not* claimed: the
anneal schedule (20%/60%) and the 42% ratio are single points, not swept, so a negative
result at these settings does not establish that no schedule works -- only that the
obvious one does not. That distinction is the one 3a-ii failed to make about RL.

### 3a-xvi. Forcing commitment: the pre-registered fork resolves against the hypothesis

3a-xii found the answer-only failure is *indecision*, not error -- every seed at
`ptr_sharp` 1.0 scored 1.000, every seed below it ~0.04, nothing between. 3a-xiv
pre-registered the fork: if commitment is the **cause**, forcing it should fix accuracy;
if pointers go sharp while accuracy stays near zero, commitment is a **symptom** and the
outcome gradient genuinely points the wrong way.

Depth 3, 5 seeds, `program_coef=0` throughout, GPU+CPU hybrid:

| arm | mean | sd | `ptr_sharp` | verdict |
| --- | --- | --- | --- | --- |
| `answer-only` (control) | 0.011 | 0.009 | 0.862 | — |
| `answer-only-gumbel` | 0.023 | 0.010 | **0.996** | commitment supplied |
| `answer-only-commit` | ~~0.024~~ | — | **nan** | **void, see below** |

**Straight-through Gumbel made the pointers sharp -- 0.996 against the control's 0.862 --
and accuracy did not move: 0.023 against 0.011, permutation *p* = 0.087, CI crossing
zero.** The fork resolves to *symptom*. Commitment is not the missing ingredient, and
the outcome gradient does not locate a correct program at seven instructions even when
it is handed a decision.

That reinterprets 3a-xv's positive result. The warm start does not work by supplying
*commitment*; it works by supplying the **right program**. Which means the exact
generator has to stay in the loop permanently -- outcome-only learning will not bootstrap
structure, at any sharpness.

#### The entropy arm is void, and how it announced itself

`answer-only-commit` reported **+0.013 at permutation *p* = 0.0317, CI excluding zero**,
and that number is **withdrawn**: its weights were `nan` from **step 0**, so the accuracy
was decoded from a destroyed model.

The mechanism is worth writing down because every layer of it looked fine:

- A masked pointer slot has `lp = -inf`, so `p * lp` computes `0 * -inf` = `nan`.
- `nan_to_num(0.0)` repaired the **forward** value and does nothing to the **backward**,
  so a `nan` gradient reached `clip_grad_norm_`, which turned the whole gradient `nan`,
  which turned every weight `nan` on the first optimiser step.
- The printed loss stayed **finite** (3.22) because the forward was clean.
- The entropy term then reported **0.0000** -- because `nan_to_num` was also swallowing
  the evidence that anything was wrong.
- `evaluate` still returned an accuracy, and the permutation test still called it
  significant.

So a destroyed model produced a statistically significant result, and the only symptom
anywhere in the output was `ptr_sharp` reading `nan` -- a metric that exists because
3a-xii needed it, added for an unrelated reason a few hours earlier.

Two fixes. The entropy is computed on `log_softmax(...).clamp_min(-30)` rather than
`nan_to_num`, so both passes stay finite and the masked terms contribute ~1e-13 x -30.
And `train_step` now **raises** on a non-finite loss or gradient, because a trainer that
silently continues on wreckage keeps printing numbers and the numbers pass significance
tests. A crash is cheaper than a void result that looks like a finding.

The arm is re-run with the fix; the entropy penalty is also floored at a target entropy,
since minimising entropy is unbounded below in logit space and would otherwise drive the
logits apart forever.

### 3a-xv. The curriculum arms: both criteria pass, and the task is easier than it looked

Depth 3, 5 seeds, everything else as 3a-xii:

| arm | mean | sd | `canonical_acc` | `mis_answered` |
| --- | --- | --- | --- | --- |
| `regmachine-supervised` | 1.000 | 0.000 | 1.000 | 0.000 |
| **`regmachine-anneal`** | **1.000** | 0.000 | 1.000 | 0.000 |
| **`regmachine-partial42`** | **1.000** | 0.000 | 1.000 | 0.000 |
| `regmachine-answer-only` | 0.015 | 0.013 | 0.000 | 0.985 |

Both pre-registered criteria (3a-xiv) pass: the anneal arm holds 1.000 through 400 steps
with the gold program fully withdrawn, and 42% problem-level supervision is as good as
100%. `answer_acc_hard` is 1.000 for both, so this is not a soft-readout artefact.

**And then the check that should have come first.** At fixed depth the grammar builds a
*balanced* tree, so the post-order traversal is identical for every problem — measured,
**exactly 1 distinct pointer pattern across 300 depth-3 problems**, with only the
operators varying (118 patterns of a possible 128). The program the model must emit is a
**constant** 7-instruction address structure plus seven binary operator reads.

That changes what these numbers mean, and mostly downward:

- **`partial42` says much less than it appears to.** Supervision on 42% of problems
  transfers to the other 58% — in a setting where the program is *the same program*.
  It shows 42% is enough to learn a constant. GSM8K's programs genuinely differ per
  problem in length, structure and operand addresses, so this arm does **not** license
  the inference its criterion was written to support. The criterion is satisfied
  literally; the conclusion I wanted from it does not follow. Recorded rather than
  reinterpreted.
- **`anneal` survives, but narrower.** What the answer loss holds without labels is a
  constant pattern, not a per-problem program. The gap it bridges is nonetheless real:
  the answer-only arm *fails to find this same constant*, so 0.015 → 1.000 is genuinely
  attributable to the warm start.
- **It also explains `sd` = 0.000 everywhere supervision is present.** A constant is
  memorisable, which is why every supervised variant is perfect on every seed at both
  depths. That is not evidence of a robust learner.

**The one result it strengthens is the negative one.** Outcome-only induction cannot
discover a pointer pattern that is *identical for every problem in the task* — 675
choices per instruction over seven instructions, with the same answer every time, and
it lands at 0.015. That is a worse verdict than 3a-xii stated, not a better one.

**What this makes necessary:** a task whose program structure actually varies. The
balanced-tree grammar cannot provide one, and `n_instr` was derived from depth until
3a-xvi, so nothing here could express a short program. Variable-length or unbalanced
expressions are the prerequisite for any claim about program *induction* as opposed to
constant-pattern recall — including everything 3a-vii ever said.

### 3a-xiii. Redundant residues on a trained model: the prediction fails

3a-ix measured `RedundantResidueSystem` standalone — 100% of single-residue
corruptions corrected, 0% mis-corrected — and nothing imported it. The depth-2 run in
3a-xii gave the reason to: the answer-only arm's `mis_answered` was **0.442**, i.e.
44% of held-out problems answered with a confident wrong number, with no checker in the
path at all.

So the arms were re-run with three redundant moduli (`(7,13,41)` on top of the core
`(16,25,27,11,37)`; legitimate values then occupy 2.198e6 of an 8.200e9 ring, 0.027%).
**Prediction, written into the code before the run: `mis_answered` collapses while
`refused` absorbs it.**

| arm | `answer_acc` | `refused` | `mis_answered` |
| --- | --- | --- | --- |
| `regmachine-supervised-redundant` | 1.000 (5/5) | 0.000 | 0.000 |
| `regmachine-answer-only-redundant` | 0.540 (sd 0.423) | **0.013** | **0.446** |

**The prediction is wrong.** Redundancy converted 1.3% of the error budget into
refusal and left 44.6% as silent wrong answers — statistically indistinguishable from
the 0.442 it was supposed to fix.

**Why, and this is the part worth keeping.** Redundant residues detect a *corrupted
codeword*: flip one residue and the reconstruction leaves the legitimate range. A
random incoherent vector should be caught ~99.97% of the time in this ring, and the
1.3% refusal rate says incoherent vectors are rare here. The failing seeds are not
emitting corrupted codes. At `ptr_sharp` 0.86–0.89 their pointers are sharp enough that
the program executes *coherently* — it is simply a **different program**, and its result
is a perfectly legal integer that happens to be wrong.

**Redundancy cannot detect a wrong answer, only an ill-formed one.** That distinction
was implicit in 3a-ix and 3a-x and is now explicit, because it is the difference between
the machinery being useful here and not. Both earlier measurements stand exactly as
stated — single-residue corruptions and incoherent soft reads *are* caught — they just
do not describe the error this model actually makes. The README's "the failure mode is
refusal, never a wrong answer" is true of the **code** and is not a guarantee about the
**computation**; it has been corrected to say so.

What the run does establish, and it is not nothing: **the wider code is free.** The
supervised arm is 1.000 on all 5 seeds with three extra residue blocks to predict (8
instead of 5) and `mis_answered` exactly 0. So redundancy costs no accuracy and can be
carried wherever an ill-formed code is the risk — a corrupted store, a soft read, a
transmission — which is a narrower and more honest claim than the one this project was
heading toward.

Open, and the obvious next question: a checker for a *coherent but wrong* answer cannot
come from the code space. It has to come from recomputation — the root-slot consistency
idea of 3a-vi, which failed for a different reason (the model never learned the root), or
from executing a second, independently-emitted program and comparing. Neither is built.

### 3a-xii. Pre-registered: does outcome-only induction survive depth 3?

3a-vii is the one learned claim in this project still standing, and the one the
language bridge depends on, and it is `n=1`: depth 2, four operands, three
instructions, `(+,−)`, one seed, run from a notebook cell. `lamb/study.py` exists
precisely to stop claims of that shape and had never been pointed at it. It is now:
`--arms regmachine-supervised regmachine-answer-only`.

**The criterion, fixed before running.** Outcome-only induction (`program_coef=0`)
reaches held-out `answer_acc` within noise of the supervised arm at **depth 3**, over
≥5 seeds, by the exact permutation test in `lamb/study.py`. Read `answer_acc`;
`canonical_acc` is conformity to the generator's form and is *expected* near zero for
the answer-only arm, because 48 distinct three-instruction programs compute
`(a+b)+(c+d)` and the grammar emits one of them. Low canonical agreement does not
imply a wrong program — that inference has never held here.

Two things that must be read alongside the number, or it means less than it looks:

- **`dropped`.** Depth 3 and multiplication leave the ring, and until 3a-xi this
  trainer trained on the wrapped value. Rows outside the ring are now masked and the
  share is reported; an accuracy measured on 60% of a held-out set is not the same
  number as one measured on all of it.
- **The search space.** Pointer choices per instruction grow as `|ops| · n_slots²` —
  3·7² at depth 2, 4·15² at depth 3 with division. This is not a small extrapolation
  and a negative result is a real one, to be reported with the same energy as a
  positive one.

Division (`ops_key=2`, task `d2g1d`) is a second arm rather than part of this one:
it needs rational registers, so the ring changes with it, and two changes in one
comparison is the confound this project keeps retracting claims over.

#### Depth 2 first, as a replication. It does not replicate.

Before the pre-registered depth-3 run, the same arms at the depth 3a-vii actually
measured. 5 seeds, 1000 steps, batch 64, `d_model=96`, moduli `(16,25,27,11,37)`:

| arm | n | mean | sd | per seed |
| --- | --- | --- | --- | --- |
| `regmachine-supervised` | 5 | **1.000** | 0.000 | 1.000 ×5 |
| `regmachine-answer-only` | 5 | **0.558** | **0.405** | 1.000, 1.000, 0.303, 0.283, 0.203 |

*(Read `answer_acc_hard` below for the stricter and more honest version of this row:
the arm mean is **0.422**, and the three failing seeds are at ~0.04 rather than ~0.25.)*

**The `n=1` answer-only result in 3a-vii was a lucky seed.** It reported 1.000 and
concluded that *"gradients through exact arithmetic are sufficient to induce a correct
program from outcomes alone."* At 5 seeds that outcome occurs **twice**; the other
three land at 0.203–0.303. The supervised arm, by contrast, is 1.000 on every seed
with sd exactly 0.

The honest claim is therefore **"outcome-only induction can find a correct program but
does so unreliably"**, not "is sufficient". Two of five is a real capability and a
poor foundation.

Diagnostics, which say what the failing seeds are doing:

| | supervised | answer-only |
| --- | --- | --- |
| `canonical_acc` | 1.000 | 0.000 |
| `op_acc` | 1.000 | 0.547 |
| `ptr_acc` | 1.000 | **0.037** |
| `ptr_sharp` | 1.000 | 0.886 |
| `mis_answered` | 0.000 | **0.442** |

`canonical_acc 0.000` is expected and not a failure (48 programs compute
`(a+b)+(c+d)`). With no redundant moduli configured, the 0.442 `mis_answered` are
silent wrong answers, which is precisely the case 3a-ix exists for and which no arm
here had switched on.

**Per seed, `ptr_sharp` separates the two basins exactly, and that is the whole
finding.** No extra compute; it was already in the run's own JSON:

| arm | seed | `answer_acc` | `ptr_sharp` | `op_acc` | `canonical_acc` |
| --- | --- | --- | --- | --- | --- |
| answer-only | 0 | **1.000** | **1.0000** | 0.510 | 0.000 |
| answer-only | 3 | **1.000** | **1.0000** | 0.655 | 0.000 |
| answer-only | 1 | 0.303 | 0.7663 | 0.520 | 0.000 |
| answer-only | 2 | 0.283 | 0.7744 | 0.456 | 0.000 |
| answer-only | 4 | 0.203 | 0.8913 | 0.596 | 0.000 |
| supervised | all 5 | 1.000 | 1.0000 | 1.000 | 1.000 |

Every seed that reached `ptr_sharp = 1.0000` scored exactly 1.000. Every seed that did
not scored below 0.31. There is no middle.

**So the failure mode is not a wrong program, it is an *uncommitted* one.** That is a
much sharper statement than "high variance", and it is mechanically consistent with
3a-x: a soft pointer read is not a blend, because decoding argmaxes each modulus
independently, so ~30% of blurred reads decode to a value in neither register. An
undecided program cannot be right even when its intent is right.

Note also that the two successful seeds have `canonical_acc 0.000`. They did not
recover the grammar's program; they found a *different* program that is exactly
correct — which is the 3a-vii non-uniqueness point holding up, and the reason
`answer_acc` is the only correctness measure here.

#### Pre-registered: is commitment the cause, or a symptom?

`RegisterMachine` has carried straight-through Gumbel-Softmax on the discrete choices
(`tau`, `hard`) since it was written, off by default, with the note that *"whether the
discreteness is needed is a question to measure, not to assume."* The table above is
the motivation for measuring it: straight-through keeps the forward pass discrete, so
the executor sees a real program rather than a blur of several, which is exactly what
the failing seeds lack.

The confound to be honest about first: `ptr_sharp = 1` may be a *consequence* of having
found a correct program rather than a cause of it — confidence following correctness.
Forcing discreteness is the intervention that separates the two, because it supplies
commitment without supplying any information about *which* program is right.

**Criterion, fixed before running.** Arm `regmachine-answer-only-gumbel`
(`program_coef=0`, `tau>0`, `hard=True`), 5 seeds, `d2g1`, everything else identical:

- **Commitment is the cause** if the arm reaches `answer_acc` ≥ 0.9 on at least 4 of 5
  seeds. The mechanism would then be that outcome-only induction finds correct
  programs routinely and loses them to indecision.
- **Commitment is a symptom** if pointers go sharp (`ptr_sharp` ≈ 1) while accuracy
  stays near the 0.2–0.3 band. The gradient would then genuinely be pointing the wrong
  way on those seeds, and outcome-only induction is weaker than even the corrected
  reading above suggests.
- **Neither, and the test did not run**, if `hard=True` destabilises training so that
  `ptr_sharp` does not reach ~1. A precondition failure is not a licence to
  reinterpret the outcome — that error is logged at 3a-vi and again two sections up.

This is a question about the *executor*, which is the exact part of the design that
distinguishes it from PAL/Program-of-Thought, so it is worth a clean answer either
way.

**On the statistics, including where my own criterion was badly written.** The paired
difference is −0.442 with a 95% CI of [−0.745, −0.139] that excludes zero, but the
exact permutation test gives *p* = **0.167** and the sign-flip test *p* = **0.250**.
By this project's own rule — permutation tests, not intervals — that is *not*
significant. The reason is the arm's own variance: at sd 0.405 the smallest paired
difference 5 seeds can resolve is **0.513**, and the study prints that it would take
**526 seeds** to resolve 0.05. Two of the five paired differences are exactly zero,
which weakens the paired test further.

So the criterion I pre-registered above — "within noise of the supervised arm ... by
the exact permutation test" — is **wrong as worded**, and I am not going to read a
pass out of it. A non-significant test is not evidence of equivalence, and a design
that could not have detected a 50-point difference cannot certify a 44-point one as
noise. That is the same error as 3a-i, where the boundary was never compared against a
converged baseline: a test whose precondition fails did not run.

What survives without any test is a description: three of five answer-only seeds
scored 0.203–0.303, and no supervised seed came within 0.7 of that. The claim to make
is about **reliability**, and reliability is what the n=1 result could not have seen.

**This makes the GSM8K annotation finding (3c) more valuable, not less.** The reliable
arm is the supervised one, and GSM8K supplies exactly that supervision for 42% of its
train split — so the bridge does not have to depend on the mechanism that just failed
to replicate.

#### The pre-registered depth-3 verdict: outcome-only induction does not scale.

5 seeds, 1000 steps, same settings, depth 3 (8 operands, **7 instructions**, 15
registers):

| arm | n | mean | sd | per seed |
| --- | --- | --- | --- | --- |
| `regmachine-supervised` | 5 | **1.000** | 0.000 | 1.000 ×5 |
| `regmachine-answer-only` | 5 | **0.015** | 0.013 | 0.029, 0.021, 0.021, 0.002, 0.000 |

Paired difference **−0.985**, 95% CI [−0.995, −0.975], **seed ranges disjoint**, exact
permutation *p* = **0.0079**, sign-flip *p* = 0.0625 (its floor at 5 seeds).

**The criterion fails, and unlike depth 2 this test was powered to say so.** The
smallest paired difference 5 seeds could resolve here is **0.017**; the observed one is
0.985, fifty-eight times that. Both arms need only 2 seeds for a 0.05 difference. There
is no underpowering to hide behind and no reinterpretation to make.

So: **gradients through exact arithmetic are not sufficient to induce a correct program
from outcomes alone.** They were sufficient on two of five seeds over *three*
instructions and on none over *seven*. The mechanism that ROADMAP 3a-vii called "the
one thing in this design that a non-differentiable executor cannot do" does work — it
just does not survive the step from a 3-instruction program to a 7-instruction one.

That is not surprising in hindsight and the pre-registration said so before the run:
pointer choices per instruction grow as `|ops| · n_slots²`, which is 3·7² = 147 at
depth 2 and 3·15² = 675 at depth 3, over seven instructions instead of three. What the
run establishes is that the answer signal alone does not navigate it.

Meanwhile **the supervised arm is 1.000 on every seed at both depths**, with `sd`
exactly 0 and `canonical_acc` 1.000. The executor, the algebra, the pointer masking and
the register file all scale to depth 3 without a wobble. It is specifically the
*outcome-only* learning signal that does not.

**The soft-eval caveat is closed at this depth too, and the verdict does not move.**
Executing the argmax program rather than the soft mixture gives `answer_acc_hard`
**1.000** for the supervised arm (identical — `ptr_sharp` is exactly 1, so soft *is*
argmax) and **0.025** for answer-only against its soft 0.015. At depth 2 hardening
*hurt* the failing seeds (0.558 → 0.422); here it helps them negligibly (0.015 →
0.025). Either way both numbers round to "nothing", so the readout is not what produced
this result — which is the only thing the check needed to establish.

The answer-only arm's `mis_answered` is **0.985** — essentially every held-out problem
answered with a confident wrong number, since no redundancy was configured. `op_acc`
sits at ~0.50 against a 0.33 chance baseline, so the operators are partly learned while
the program as a whole is not, and `ptr_sharp` is 0.87–0.95: as at depth 2, the failing
runs never commit.

**What this costs the project.** The language bridge was described in 3a-vii as
depending on this mechanism, because a word problem supplies no gold program. On this
evidence that dependency is not safe, and the bridge should not be built on it. Two
things carry the weight instead, both established tonight: GSM8K's calculator
annotations supply a real program for 42% of the train split (3c), and the *supervised*
arm is the one that is perfectly reliable. The honest plan is supervision where it
exists and an answer-only arm reported as a control, not as the mechanism.

#### The soft-eval caveat, tested: it went the other way, and the finding was understated

`answer_acc` executes the *soft* program, so a blurred-but-recoverable program would be
scored on a mixture it never computes — which would make the whole result an artefact of
the readout rather than a fact about the learning. The argmax program is now executed as
a real program and reported as `answer_acc_hard`. Depth 2, per seed:

| seed | `ptr_sharp` | `answer_acc` (soft) | `answer_acc_hard` |
| --- | --- | --- | --- |
| 0 | 1.0000 | 1.000 | **1.000** |
| 3 | 1.0000 | 1.000 | **1.000** |
| 1 | 0.7663 | 0.303 | **0.043** |
| 2 | 0.7744 | 0.283 | **0.041** |
| 4 | 0.8913 | 0.203 | **0.027** |

**Hardening does not rescue the failing seeds; it exposes them.** The committed seeds
are identical either way (`ptr_sharp` is exactly 1, so soft *is* argmax). The
uncommitted ones drop from 0.20–0.30 to **0.027–0.043** — the soft score was flattering
them by roughly sevenfold, because a mixture can land on the right answer by spreading
mass where the single program it would actually emit does not.

So the caveat resolves against itself and **the earlier reading was too generous, not
too harsh**. Two corrections to what is written above:

- The failing seeds do not "collapse to ~0.25". They collapse to **~0.04**. The 0.25
  was an artefact of scoring a mixture, and the arm mean is **0.422** on the stricter
  reading rather than 0.558.
- Depth 2 and depth 3 are therefore the *same* picture, not two different ones. At both
  depths the answer-only arm either finds the program (1.000) or essentially fails
  (~0.04 at depth 2, 0.015 at depth 3). **There is no partial credit in outcome-only
  induction** — the intermediate scores were measurement, not learning.

That also completes the case for `ptr_sharp` as the diagnostic: sharp gives 1.000, blurred
gives ~0.04, and nothing sits between. And it is a reminder in the other direction from
this project's usual one — the check was run expecting to weaken a negative result, and
it strengthened it.

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

## 3c. The language bridge (GSM8K) — implemented, unmeasured

`lamb/bridge.py` held a quantity parser, a resampler and a register loader, and no
path between them; `encode_dataset` was referenced twice in its own docstring and
defined nowhere. Nothing in the repo turned a sentence into a number.
`lamb/bridge_train.py` (`python -m lamb.bridge_train`, `lamb-bridge`) is that path.

    text -> frozen encoder -> cached embeddings     no
    text -> quantities -> registers                 no   (regex + fixed rational map)
    embeddings -> K latents (resampler)             yes
    latents -> program over registers               yes
    program -> answer                               no   (exact rational algebra)

The encoder runs **once** over the dataset and is cached to disk, so it never
participates in training and its size stops being a constraint. `transformers` is an
optional extra (`uv sync --extra bridge`) rather than a dependency: every arithmetic
number in this repo was produced under the pinned environment, torch moves accuracy
on a model this small between minor versions, and a text front-end should not be able
to shift the ground under a measurement it has nothing to do with.

**GSM8K ships gold programs, and that changes the shape of this.** The premise
recorded in 3a-vi and 3a-vii is that a natural-language problem does not come with a
program, so the bridge rested entirely on the `n=1` outcome-only result. That is true
of prose in general and **false of this dataset**: GSM8K's train solutions carry
inline calculator annotations of the form `<<48/2=24>>` — an operation, its operands
and its result, in evaluation order. So the bridge is a measurement with a control,
not a leap: the supervised arm recovers a program from the dataset's own annotations,
and `--program-coef 0` removes it entirely and asks 3a-vii's question on real text.

Three things are built into it because each one caps what any model on top could
reach, and all three are reported before training rather than inferred from a bad
number afterwards:

- **Coverage.** Only annotations that are a single binary operation over two
  literals are usable; compound ones (`<<48*3+5=149>>`) need a parser and more than
  one register write. Rejects are counted. So is `operand_miss` — the share of rows
  whose annotation names a number the extractor never found, which is the ceiling on
  the supervised arm and has nothing to do with the network.
- **Execution check.** A program recovered from someone else's annotations is a
  hypothesis. `verify_alignment` executes it in this ring and asks whether it
  reproduces the dataset's own answer; if it does not, training on it would teach the
  wrong program perfectly.
- **Constant registers.** "Half as many", "twice", "a third", "20% off" each name an
  operation whose other operand is never written down. Preloading `(1, 2, 3, 100)`
  keeps that decision in the program — `x / 2`, not a parser deciding that "half"
  means 0.5 — which is the same argument the percent flag already makes. By this
  repo's own note, "half as many" is the most common operation in GSM8K, so without
  the constants the alignment fails on problems the parser read perfectly.

**Measured: coverage, and one defect it caught.** Run before any training
(`python -m lamb.bridge_train --coverage`), on the official splits:

| | n | no_answer | answer_not_in_chain | operand_miss | program_coverage |
| --- | --- | --- | --- | --- | --- |
| train | 7473 | 0.000 | 0.054 | 0.316 | **0.614** |
| test | 1319 | 0.000 | 0.073 | 0.287 | **0.625** |

Recovered programs that execute to the dataset's own stated answer: **1.000**.

That last number was **0.824** on the first pass, and finding out why is the reason
the check exists. 16.6% of usable annotation chains **end somewhere other than the
answer** — the step that produced it was compound (`<<48*3+5=149>>`, which this
parser rejects) or was never annotated at all. Keeping those meant one recovered
program in six executed to the wrong number, which is worse than having no program:
it is supervision toward a wrong answer, and the model would learn it perfectly. The
chain is now truncated at the last step whose result *is* the answer, and a chain
that never reaches it is dropped. The execution rate went 0.824 → 1.000, and coverage
fell 0.533 → 0.420 to pay for it. **Less supervision that is correct beats more that is
not.** (Coverage later recovered to 0.614 for an unrelated reason — see the compound
decomposition below — and `answer_not_in_chain` fell 0.170 → 0.054 with it, because the
step that produced the answer was so often the compound one.)

**Ablated, because both of those components were added from reasoning rather than
evidence.** `--no-lexical` existed as the ablation and nobody had run it:

| constants | written numerals | `program_coverage` | Δ | what it buys |
| --- | --- | --- | --- | --- |
| `()` | no | 0.004 | — | |
| `(1,)` | no | 0.239 | **+23.5** | the identity register, not a fact about GSM8K |
| `(1,)` | yes | 0.318 | **+7.9** | "three", "a dozen" |
| `(1,2)` | yes | 0.397 | **+7.9** | "half", "twice", "double" |
| `(1,2,3)` | yes | 0.416 | +1.9 | "a third" |
| `(1,2,3,100)` | yes | **0.420** | **+0.4** | percent |

*(Measured before compound annotations were decomposed, so every row is ~19 points
below its current value. The ablation is a comparison between rows and the ranking is
unaffected; the absolute figures are not the headline coverage.)*

The first row is a confound in my own measurement and is called out rather than
reported: with no constants there is no register holding `1`, so the `x * 1` padding
instruction is unavailable and rows are rejected for a reason that has nothing to do
with the dataset. Isolating it costs one extra run and moves +23.5 points out of the
"constants help" column, where they did not belong.

With that removed, the ranking is: **the constant `2` and the written-numeral table are
worth about 8 points each**, and they are the two components that were guesses. `2` is
"half as many", which this repo's own note predicted would be the most common operation
in GSM8K, and the measurement agrees. `3` is worth 1.9.

**`100` is worth 0.4 and is nearly useless**, which is a negative result on a component
added by reasoning about percent: GSM8K's annotations write a percentage as a decimal
(`<<0.2*50=10>>`) rather than as `20/100`, so the constant is rarely what the chain
asks for. It is kept only because it still nets positive against the register slot it
occupies, and that is a thin margin rather than a justification.

**The largest single fix, found by asking what was actually in the failures.** The 39%
of usable chains that failed alignment break down as:

| cause | share |
| --- | --- |
| an operand produced by a **rejected compound step** | **52.1%** |
| integer operand absent from the text | 33.0% |
| non-integer operand absent from the text | 13.9% |
| register file full, a quantity pushed out | 0.9% |
| needed more than 8 instructions | 0.1% |

Half the gap was self-inflicted. A compound annotation (`<<48*3+5=149>>`) was rejected
for not being a single binary operation — but it is simply *two* instructions, and
rejecting it cascades: its result never reaches a register, so every later step that
reads it fails to align too. `_lhs_steps` now decomposes it with a recursive-descent
parser (precedence, left-associativity, parentheses, unary minus), and **coverage moved
0.420 → 0.614** with `exec_ok` still exactly 1.000.

Two things that breakdown settles, both of which were guesses in `BridgeConfig`: the
register file is essentially never the constraint (0.9%, measured at `n_operands=12`
with the four operation constants only -- it becomes 97.3% once unit constants are added,
which is the coupling below) and the instruction budget almost never is (0.1% at
`n_instr=8`). Chain length is median 3, p90
6, p99 9, max 15, so `n_instr=12` would add +1.0 point of coverage for 50% more
execution per step — measured and declined.

**Then the same question again, and the answer changed.** Re-categorising what *still*
failed after decomposition:

| cause | share |
| --- | --- |
| **integer operand absent from the text** | **59.6%** |
| non-integer operand absent from the text | 27.7% |
| operand from a still-rejected step | 5.7% |
| register file full | 3.8% |
| chain longer than 8 instructions | 3.2% |

The leader is not an extraction failure at all. *"Weng earns $12 an hour ... 50
minutes"* needs **60**, and no parser can extract a number the problem never writes.
That is unit knowledge, and it is the same shape as "half as many" needing a `2` — so
it gets the same answer: `DEFAULT_CONSTANTS` gains `60, 24, 7, 12, 52, 1000`, and
coverage moves **0.614 → 0.680** train, **0.691** test, `exec_ok` still 1.000.

**And the naive version of that change is actively harmful, which is the finding worth
keeping.** Every constant occupies a register slot. At `n_operands=12` those ten
constants fill **97.3%** of register files and push the problems' own quantities out:
coverage does not improve, it *halves*, 0.614 → **0.249**. Constants and file width are
one decision:

| constants | `n_operands` | coverage | file full |
| --- | --- | --- | --- |
| `(1,2,3,100)` | 12 | 0.614 | 0.044 |
| `+ 6 unit constants` | 12 | **0.249** | **0.973** |
| `+ 6 unit constants` | 16 | 0.657 | 0.162 |
| `+ 6 unit constants` | **20** | **0.680** | 0.012 |

`n_operands=20` is the default for that reason, and a test pins the pair together
because adding a constant without widening the file is a regression that reads as a
feature. The ring is unaffected (margin still 76.7×) since chain length is capped by
`n_instr`, not by operand count — checked rather than assumed, after the last time a
coverage change quietly ate the ring margin.

The dominant remaining loss is now `operand_miss` at ~0.25: an operand the annotation
names is not in the register file. Part of that is genuine extraction misses
(`1/2`, values the text never writes), and part is a cascade — when a compound
annotation is rejected, its result is never written to a register, so every later
step that reads it fails to align. **That 0.42 is the ceiling on the supervised arm
and it has nothing to do with the network**, which is exactly why it is measured
first. The answer-only arm is not bounded by it: it trains on all 7473.

**Measured: the ring GSM8K actually needs, which is far smaller than 3a-viii sized.**
Also computable with no model in the loop. What matters in residue form is the
*unreduced* numerator and denominator a program builds, since a fraction cannot be
reduced — so simulate that growth in plain integers over the 4590 recovered programs:

| percentile of worst \|num\|,\|den\| per program | value |
| --- | --- |
| p50 | **120** |
| p90 | 5.5e3 |
| p99 | 3.8e6 |
| p99.9 | 1.1e9 |
| p100 (worst of 4590) | **2.16e11** |

`RATIONAL_MODULI` gives ±4.49e15 — **six orders of magnitude more than the worst case
needs**. 3a-viii sized it from an argument ("five two-decimal values, denominator
1e10"); the argument was not wrong, it was answering a worst case the data does not
contain.

`GSM8K_MODULI = (64, 125, 27, 11, 7, 13, 37, 101, 41)` is the measured sizing:
±1.66e13, a **77× margin** on the observed worst case, every digit period still ≤ 6 —
and much cheaper, because the packed path pads every modulus to the widest. 9 moduli at
P=125 is `9·125² = 141k` against `10·271² = 734k`, so **5.2× less arithmetic per
composition**, and the head width drops 697 → 426 units. It is the bridge's default;
`RATIONAL_MODULI` is one constructor argument away.

**It was 8 moduli and a 139× margin until the chains got longer, and that is the part
worth remembering.** Decomposing compound annotations — a change about *coverage*,
with no apparent connection to the ring — roughly tripled the median chain length, and
since every operation multiplies denominators the worst case moved 2.91e9 → **2.16e11**
and the margin collapsed **139× → 1.9×**. No program exceeded the ring even then, so
nothing failed and **nothing would have failed visibly**: the next slightly longer chain
would simply have wrapped to a different number. It was caught only by re-measuring the
ring after a change that had nothing obviously to do with it.

The alternatives with a larger margin were rejected on principle rather than cost:
adding 73 or 137 instead of 41 buys a wider ring at similar width but pushes
`max_period` to 8, and a modulus whose digit-coefficient pattern needs 8 positions to
repeat is useless at the operand widths training actually sees. That constraint, which
exists for extrapolation, did real work here.

Two limits on that sizing, since it is a decision and not just an observation. It is
measured over the programs the annotations *describe*, so a model emitting a different
one — dividing repeatedly, say — is not bounded by it. And a soft, undecided program is
bounded by nothing at all, because its decoded denominator is an argmax over
incoherent residues rather than a value (see `ptr_sharp`). `denominator_magnitude`
remains the monitor.

**Measured: it runs end to end, and what it costs.** 7473 train and 1319 test
problems encoded once by `all-MiniLM-L6-v2` at `max_len=192`, with **1 problem
truncated** in 8792 — so the cache is not quietly cutting questions off. The trainer
then runs text -> cached embeddings -> resampler -> shared latent core -> program ->
exact rational execution, with gradients reaching the program heads through the
arithmetic.

Where the parameters are, which is the point of the design:

| | params | share |
| --- | --- | --- |
| resampler (learned) | 0.318M | 38.4% |
| shared latent core (learned) | 0.502M | 60.6% |
| **program heads (learned)** | **7.7k** | **0.93%** |
| frozen encoder | 22M, cached, never in the training graph | — |
| the arithmetic | 0 | — |

**The learned surface that decides *which computation to perform* is under eight
thousand parameters.** Everything else compresses the text or performs arithmetic
exactly. (It was 5k before `n_operands` went 12 → 20: the pointer heads are
`Linear(d_model, n_slots)`, so they scale with the register file.)

**Cost, and why no single number is given.** The same configuration measured **1.68
s/step** and **2.48 s/step** at batch 8 within an hour of each other, differing only in
what else the box was running. Earlier in this session a contended host turned a
19-minute study arm into a reported 36,915 s (3a-xi, item 5), so a wall-clock figure
from a shared machine is not a cost measurement and will not be quoted as one. The
honest statement is that this is a CPU-bound prototype at ~2 s/step at batch 8, that a
real run wants a bigger batch and a GPU, and that the profile in `CLAUDE.md` -- 92% of a
register-machine step is torch forward+backward, ~6.5% the exact execution -- is the
part that transfers between machines.

**No accuracy is claimed.** Nothing here has been trained to convergence or run on a
GPU, and this section will carry accuracy when there is accuracy. One number from the
smoke run is worth repeating as a warning rather than a result: `max_den_magnitude`
read **3.96e11** against a 4.04e11 ring after 10 steps, which looks like the ring is
exhausted and is not. It is an argmax over each modulus independently, so an
uncommitted program decodes to an essentially uniform ring value whatever the
arithmetic did. Read it with `ptr_sharp` or do not read it. And the contamination
claim stays withdrawn: the encoder is pretrained, it has seen these benchmarks, and
"frozen" means its weights do not move rather than that the information is absent.
What replaces the claim is an arm — `--encoder` swaps the tower, embeddings are
cached, so running the same core on a pretrained encoder and on one that never saw
the benchmark prices the encoder's prior directly. That arm is the only part of this
work that would be a *new* result rather than a port, and it has not been run either.

Known limits, recorded rather than left implicit:

- `extract_quantities` is a regex plus a table of written cardinals. It does not see
  `1/2`, and it emits spurious operands for dates, ordinals and item numbers, which
  can push a real quantity out of a fixed-width register file. Quantity *selection*
  is a missing component, not a tuning detail.
- The register file is a fixed shape, so a chain longer than `n_instr` is rejected
  rather than truncated.
- The answer-side loss for rationals is the cross-product residual. Its only blind
  spot is `0 ÷ 0`; a zero divisor with a live numerator is detected by the residual
  itself. See 3a-xi, including where I had that wrong.

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
