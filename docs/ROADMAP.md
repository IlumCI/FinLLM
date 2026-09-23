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

## 7. Scale-up path

GPU training, mixed precision, larger `d_model`/depth, and a serving mode. The
architecture is unchanged; only config and device move.
