# LAMb roadmap

v0.1 is a faithful, CPU-runnable scaffold of the full architecture. The
extensions below are ordered roughly by leverage. Consult arXiv for the latest
work before implementing each — several of these are active 2025–2026 areas.

## 1. GRPO-trained hypernetwork proposer (headline)

The default proposer is a learning-progress **bandit** over a discrete grid:
robust, non-collapsing, cheap — but tabular, so it cannot condition on solver
state, cannot represent a combinatorial/continuous task space, and cannot
generalise to difficulty cells it has never sampled.

Plan: keep the bandit as the **stability backbone** and add an expressive policy
on top, trained by **GRPO** (Group Relative Policy Optimization — critic-free,
group-relative advantages; the RLVR workhorse):

- **Hypernetwork** `H_φ(context) → θ` emits the parameters `θ` of the task
  distribution, where `context` summarises solver competence (per-region
  learnability, recent success slope, memory-utilisation stats). Keep `θ`
  low-dimensional (a few difficulty logits/means) to keep the hypernetwork
  stable.
- **GRPO update.** Sample a group of `G` tasks, have the solver attempt each,
  reward = verifiable learnability (group correctness variance / Δ-competence),
  advantage `A_i = (r_i − mean)/std`, with a KL penalty to a reference policy.
- **Bandit as anchor.** Use the bandit two ways: (a) coverage mixture
  `(1−λ)·π_hyper + λ·π_bandit`, annealing `λ↓`; (b) the KL reference the GRPO
  policy is regularised toward. This is exactly what prevents the collapse that
  vanilla REINFORCE suffered here.
- **Guards.** Exact verifier + a validity filter block reward-hacked/ill-posed
  proposals; an EMA reference handles the non-stationarity of a moving solver.

The `SelfPlayTrainer` already accepts an injected `proposer`; this lands as a
`GRPOHyperProposer` implementing the same `sample / probs / entropy` surface plus
a group-rollout `update` hook.

Refs: GRPO (DeepSeekMath, [2402.03300](https://arxiv.org/abs/2402.03300));
hypernetworks (Ha et al.); Absolute Zero ([2505.03335](https://arxiv.org/abs/2505.03335)); R-Zero.

## 2. GRPO / RLVR on the solver

Today the solver learns by expert iteration (teacher forcing on verified traces).
Add GRPO on the solver itself so proposer and solver co-evolve under the same
verifiable-reward machinery — the natural "exploding self-improvement" loop.

## 3. Continuous-thought decoding (full Coconut)

The core already reasons in latent space via recurrent depth. Add the Coconut
inference mode: feed the last hidden state back as the next *input* embedding for
several "thought" steps between emitted tokens, enabling breadth-first latent
search on backtracking-heavy problems.

Ref: Coconut ([2412.06769](https://arxiv.org/abs/2412.06769)).

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
