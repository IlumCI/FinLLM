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

Measured on depth-2 nested expressions at matched budget (1000 steps, batch 64,
0.28M params), decomposing the two changes:

| arm | structure | supervision | answer acc | trace-probe |
| --- | --- | --- | --- | --- |
| Coconut (Stage A original) | sequential | answer-only | 0.512 | — |
| LOTUS | **parallel** | answer-only | **0.695** | 0.102 (chance) |
| LOTUS | parallel | **+ per-position trace** | **0.945** | 0.976 |

Structure alone is worth **+18.3 pts** at identical supervision; the per-position
trace adds **+25.0** more (0.512 → 0.945). These are re-measured on a *clean*
held-out partition (see below); the first reported numbers (0.520/0.707/0.875) came
from a seed-separated eval set that was 53% contaminated. The trace-probe (a
diagnostic, never decoded) confirms the latent block really does carry the
intermediates: 0.95 with supervision vs chance without. Note the third arm uses
information the first two do not — the gold trace — which is free here only because
an exact verifier exists; the second arm is the matched-supervision control, and it
already beats Coconut.

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
comparison is 0.945 without the boundary vs 0.934 with it — the sign flips, and both
differences sit inside a run-to-run spread of several points. The honest reading is
**no measurable effect**. Together with 3a-ii (its RL rationale falsified), the entry
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
boundary claim did **not** hold up (3a-i). One further lesson: the same nominal
config moved ~7 points across two runs differing only in the eval partition, which
puts a floor under how large an effect has to be before it means anything here.

For Stage B the partition is real but small (~22 of 200 problems at 1 digit), so
the comm accuracy is best read as a *channel* test — the listener never sees `X`, so
it cannot answer from memorisation without the message — rather than a
generalisation test. `--a-digits 2` gives a 20k-problem space if a generalisation
claim is wanted.

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
code becomes canonical. Other-Play realized in pure latent space. Remaining:
larger populations, heterogeneous agent sizes, and cross-architecture transfer.

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
