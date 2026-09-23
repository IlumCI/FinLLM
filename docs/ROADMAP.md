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

Measured on depth-2 nested expressions at matched forward-pass budget (1000 steps,
batch 64, 0.28M params), **over 5 seeds per arm** (`python -m lamb.study --task d2g1`),
decomposing the two changes:

| arm | structure | supervision | answer acc (mean ± sem, n=5) | sd | range | trace-probe |
| --- | --- | --- | --- | --- | --- | --- |
| Coconut (Stage A original) | sequential | answer-only | 0.504 ± 0.062 | 0.138 | 0.316–0.703 | — |
| LOTUS | **parallel** | answer-only | **0.826 ± 0.034** | 0.076 | 0.746–0.910 | 0.14 (chance) |
| LOTUS | parallel | **+ per-position trace** | **0.903 ± 0.024** | 0.054 | 0.811–0.951 | 0.97 |

- **Structure is worth +32.3 pts** at identical supervision, and it is the one
  result here that a 5-seed study can actually establish: the two arms' seed ranges
  are **disjoint** — the worst LOTUS run (0.746) beats the best Coconut run (0.703) —
  which is an exact permutation *p* = 0.0079. It also costs nothing extra: no gold
  trace, no new information, just the parallel block in place of the sequential one.
- **The per-position trace adds +7.7 pts, and that is *not* established at n=5.**
  Exact permutation *p* = 0.064; the paired sign-flip test cannot go below 0.0625
  with 5 seeds however large the effect, and it sits exactly there (all 5 seeds
  positive). Suggestive, consistent in sign, unproven. This is a large correction:
  the single-run numbers said +25.0.
- **Variance falls with each change** — sd 0.138 → 0.076 → 0.054. Given that the
  Coconut baseline swings 39 points on seed alone, making training *reliable* is
  arguably worth as much as the mean gain, and it is the less obvious result.

Earlier reported figures for these arms were 0.512 / 0.695 / 0.945, each a **single
run**. Two things were wrong with that. The eval set was 53% contaminated (fixed in
3a-iii below), and — the larger error — one run cannot separate an effect from seed
noise at this scale. At Coconut's spread, **61 seeds** would be needed to resolve a
5-point difference; nothing this repo claimed below ~18 points was ever measurable.
That is the direct explanation for the SWITCH boundary saga in 3a-i: its claimed
+5.9 was noise from the start, and no amount of care in *running* that single
experiment could have revealed it.

The trace-probe (a diagnostic, never decoded) confirms the latent block really does
carry the intermediates: 0.97 with supervision, chance (0.14) without. Note the
third arm uses information the first two do not — the gold trace — free here only
because an exact verifier exists; the second arm is the matched-supervision control,
and it is the one that carries the result.

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
multi-seed study in 3a-iv then explained *why* both numbers were meaningless: at
this model's seed-to-seed spread, a 5-point difference needs ~61 seeds to resolve,
so a single run could never have detected a real +5.9 nor ruled one out. The honest
reading is **no measurable effect, and no measurement**. Together with 3a-ii (its RL rationale falsified), the entry
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

| arm | mean ± sem | sd | range |
| --- | --- | --- | --- |
| Coconut | 0.504 ± 0.062 | 0.138 | 0.316–0.703 |
| LOTUS answer-only | 0.826 ± 0.034 | 0.076 | 0.746–0.910 |
| LOTUS + trace | 0.903 ± 0.024 | 0.054 | 0.811–0.951 |

The **Coconut baseline swings 39 points on seed alone** (0.316 to 0.703). That is
the number that reframes everything earlier: at that spread, resolving a 5-point
difference needs ~61 seeds, so no single-run claim below roughly 18 points was ever
measurable. The SWITCH boundary's +5.9 (3a-i) never had a chance of being real; it
took two experiments to kill something that a power calculation would have
predicted was unmeasurable.

One claim grows, one shrinks, and one was not being looked for:

- **Structure: +32.3 pts, established.** The seed ranges are disjoint — the worst
  LOTUS run beats the best Coconut run — an exact permutation *p* = 0.0079. The
  single-run estimate had been +18.3, so this was *under*-claimed.
- **Trace supervision: +7.7 pts, not established.** Positive on all 5 seeds, but
  *p* = 0.064, and the paired sign-flip test has a floor of 0.0625 at n=5 so it
  could not have shown significance whatever the effect size. The single-run
  estimate had been +25.0 — over-claimed by more than 3×.
- **Variance falls with each change** (0.138 → 0.076 → 0.054). Unplanned, and on a
  baseline this unstable, arguably the more useful property: the restructure makes
  training *reliable*, not merely better on average.

The spread above is training variance, not measurement noise: each arm is scored on
512 held-out problems, so the binomial standard error at these accuracies is ~1.8
points — an order of magnitude below the 13.8-point seed spread it would have to
explain. Arms at the same seed also share an eval set, so that component cancels
from the paired differences entirely.

Known confound, stated rather than buried: the arms are matched on **core forward
passes** (Coconut's K=3 thoughts + answer = 4; LOTUS's loops=3 + answer = 4) but not
on wall clock — LOTUS costs 0.564 s/step against Coconut's 0.335 s/step, because its
sequences carry the latent block. The `coconut-long` arm in `lamb/study.py` is the
control that removes it, giving Coconut the extra ~1.68× steps instead.

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

Results: pending (`scratchpad/width_test.py`).

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

**Caveat, pending re-measurement.** `comm_pop.py` held out *pairings* but not
*problems* until the sweep in 3a-iii; it trained and evaluated on the whole space,
so that `1.000` was measured on problems the population had seen. The zero-shot
*coordination* claim is about a never-co-trained speaker/listener pair and does not
obviously depend on problem novelty — but "does not obviously depend on" is not a
measurement, and this is exactly the kind of reasoning that produced the numbers
3a-iv had to correct. The number stands until re-run, and is marked as unconfirmed
until then. Note also that the 1-digit evaluation partition holds 22 problems, so a
re-run there resolves to ~4.5 points; `--a-digits 2` gives 2431.

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
