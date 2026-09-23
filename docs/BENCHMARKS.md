# Benchmarking LAMb

**MMLU (and general-LLM suites) do not apply.** LAMb's vocabulary is digits and
arithmetic operators; it has no natural-language tokens and no world knowledge,
so it cannot read an MMLU / HellaSwag / ARC question. Benchmarking a number-native
math specialist on broad NL multiple-choice is a category error. The right
battery targets LAMb's four actual claims.

## 1. Arithmetic competence and length generalization (primary)

The single most diagnostic benchmark. Train on operands of width `<= k`, test on
widths `> k`. This directly measures whether the digit + Abacus representation
*extrapolates*, which is the whole point of the number-native design.

Already in the repo:

```python
from lamb.eval import evaluate, length_generalization
length_generalization(model, tok, ops=("+", "-"), max_test_digits=6, train_max_digits=3)
# -> {1: .., 2: .., 3: .., 4: .., 5: .., 6: ..}   widths > 3 are extrapolation
```

Report per-width exact-match accuracy and the extrapolation cliff. Compare
reversed vs. forward digits and Abacus on/off to attribute the gains.

## 2. Latent reasoning and test-time compute

- **Test-time scaling curve** — accuracy vs. latent step budget `T` at fixed
  weights (`lamb.eval.test_time_scaling`). A well-trained LAMb is monotone in `T`.
  This is *vertical* latent compute (deepen fixed positions).
- **Continuous-thought scaling (Coconut, shipped)** — `python -m lamb.coconut`
  (`lamb/coconut.py`) inserts `K` continuous-thought scratchpad positions between
  prompt and answer (*horizontal* latent compute) and reports two curves on
  depth-2 nested expressions:
  - **Accuracy vs. #thoughts** — greedy exact-match rises `K=0: 0.39 → K=3: 0.52`
    on a ~0.28M-param CPU model, and the latent-collapse diagnostic (mean pairwise
    cosine of the thought vectors, the arXiv:2510.12167 homogeneity signal) falls
    cos `0.88 → 0.27` as thoughts specialise. Coconut's language-domain
    `w/o curriculum` ablation underperforms no-thoughts; the number-native
    substrate makes the scratchpad useful here without a language curriculum.
  - **Verifier-selected best-of-N** — dropout-diverse latent trajectories selected
    by the exact verifier turn arXiv:2510.12167's monotone-but-unusable Pass@N
    into *realised* accuracy `N=1: 0.52 → N=8: 0.66` (their trained reward models
    could not select; LAMb's verifier can). A test-time self-improvement loop.
  Covered by `tests/test_coconut.py` (incl. the `K=0` reduction to ordinary
  teacher forcing).
- **Parallel supervised latents (LOTUS restructure, shipped)** — `python -m
  lamb.lotus` (`lamb/lotus.py`). The sequential continuous-thought family's gap to
  explicit CoT widens with scale (arXiv:2606.31779); the looped parallel-supervised
  family stays flat, so Stage A was restructured onto it. Latents are refined
  together (cost `loops+1` forwards independent of latent count) and every latent
  position is supervised through the LM head against a gold **numeric** trace the
  evaluator generates for free. Nothing latent is ever decoded. Measured at matched
  budget on depth-2 nested expressions:
  `Coconut 0.512 → LOTUS answer-only 0.695 → LOTUS +trace 0.945`, trace-probe
  `0.98` vs `0.10` at chance, on a hash-partitioned held-out set (`lamb/holdout.py`;
  the earlier seed-separated split was 53% contaminated). The SWITCH **entry**
  boundary (arXiv:2606.13106) showed **no measurable effect** once the eval was clean
  (0.945 without vs 0.934 with) and its RL rationale was falsified, so it is off by
  default. The *exit* marker is measurably harmful and off by
  default (500-step control: no-boundary `0.371`, `BOT`+`EOT` `0.176`, `BOT`-only
  `0.500`): a static marker between the latents and the answer readout blocks the
  readout from the latent state. A class-balanced `boundary_probe` ships as the
  monitorability handle but is **not yet a measurement** — the model is either too
  accurate (too few errors) or too weak for the number to mean anything.
  Covered by `tests/test_lotus.py`.
- **Latent inter-agent communication (Stage B, shipped)** — `python -m lamb.comm`
  (`lamb/comm.py`) measures whether two agents can communicate a value in *pure
  latent space*. A speaker sees operand `X`, a listener sees `op` and `Y` and must
  emit `X op Y`; the message is a Coconut thought vector through a differentiable
  channel, trained end to end (DIAL). Reported:
  - **Causal channel gain** — comm accuracy minus a zeroed-message ablation. On
    1-digit `X,Y` (`+/−`), `comm 1.000` vs `blank 0.098` (the guess-`X` prior): a
    `+0.90` gain that is the positive-*listening* test of arXiv:1903.05168 (high
    reward with a receiver that ignores the channel is the classic false positive;
    the ablation rules it out).
  - **Bandwidth (capacity) sweep** — `--sweep` with DRU channel noise
    (arXiv:1605.06676) trades channel width against accuracy (a noiseless
    continuous channel has near-infinite capacity, so the sweep needs noise to be
    meaningful). At noise `0.5` the threshold is sharp: channel width `1/2/4` stay
    at the `~0.10` prior, then `8/96 → 1.000` — a 1-digit operand needs `≥ 8` noisy
    channel dimensions to transmit.
  - **Message diagnostics** — `signal_std` (does the message depend on `X`?) and
    `msg_cos` (representational-collapse detector, arXiv:2604.03809).
  - **Held-out-partner** (`python -m lamb.comm_transfer`) — is the code private or
    shareable (zero-shot coordination, arXiv:2003.02979)? Two pairs each reach
    `comm 1.000`, a zero-shot speaker swap collapses to `0.004/0.041` (below the
    `0.076` prior — a foreign code misleads), but a freshly trained receiver learns
    a frozen speaker's code to `1.000`: private and co-adapted, yet learnable.
  - **Partner randomization** (`python -m lamb.comm_pop`) — the fix. A population
    (3 speakers × 3 listeners) trained with random pairing and the diagonal `(i,i)`
    pairings held out reaches held-out **zero-shot `1.000`** (up from the single
    pair's `0.004` swap): partner randomization makes the code canonical.
  Covered by `tests/test_comm.py`.
- **ProntoQA / ProsQA** — the synthetic logical-reasoning sets Coconut used to
  show latent breadth-first reasoning beats token chain-of-thought. These need a
  task encoder but no general NL, so they fit LAMb's paradigm.

## 3. infContext / test-time memory

Shipped: **`python -m lamb.memory_bench`** (`lamb/memory_bench.py`) — a
needle/passkey retrieval benchmark that plants a target key->value binding at the
front of a sequence, buries it under distractor bindings and a long run of filler
tokens, and queries it at the end. It reports two curves that are the signature of
a fixed-size test-time memory, plus a memoryless ablation that isolates the
memory:

- **Unbounded context length (extrapolation).** Trained at length 48, the memory
  retrieves at 1.00 through length 96, ~0.96 at 192 (4x), and ~0.73 at 384 (8x) --
  well above chance (0.03) -- because the O(1) state is length-agnostic. The
  memoryless model stays at chance at every length, confirming the neural memory
  performs the retrieval.
- **Bounded capacity (honest tradeoff).** At fixed length, accuracy falls from
  ~1.00 (4 bindings) to ~0.37 (64 bindings) as the load approaches `d_mem=64` --
  unbounded *length*, finite *capacity*, exactly the Titans/ATLAS property.

Shipped: **`python -m lamb.ruler_bench`** (`lamb/ruler_bench.py`) — a
RULER/BABILong-*style* battery. BABILong and RULER are natural-language suites, so
the real datasets need the language-bridge fork; what ships here is RULER's
*design* (a synthetic, model-agnostic set of long-context task families with
configurable length) reconstructed over LAMb's symbols, on a model that does
**memory + latent multi-hop together** (iterative dereferencing of the built
memory state — no quadratic attention):

- **NIAH** (retrieve a bound value): ~0.7 across lengths, extrapolating from
  training length 64 to 256 (4x).
- **NIAH multi-key** (40 distractors): degrades with length — the retrieval-under-
  load stress case.
- **Variable tracking** (resolve `v1 := v2 := ... := literal`): `k` reads resolve a
  `k`-hop chain (chain=2 ~0.54, chain=4 ~0.41 vs chance 0.06), while a
  single-read ablation collapses to ~0.19 for any chain ≥ 2 — iterative memory
  reads are what perform the multi-hop reasoning. As in the real RULER, variable
  tracking is the hardest category and degrades with hop count.

Covered by `tests/test_memory.py`, `tests/test_memory_bench.py`, and
`tests/test_ruler.py`.

To add next: the real **BABILong** and **RULER** datasets, once a language front-end
exists (see the bridge below).

## 4. Self-improvement

- **Frontier-expansion curve** — mastered operand-digit sum vs. training step
  (the trainer already logs `frontier`).
- **Relative fitness (Red Queen)** — with `--red-queen`, `dominance` (current vs.
  best-past accuracy on the current frontier) and `forgetting` (regression on a
  fixed easy set). Positive, sustained `dominance` is the signature of genuine
  coevolution; its decay to 0 flags saturation of a bounded task space.
- **Sample efficiency** — accuracy vs. number of self-generated problems.
- **OOD transfer** — evaluate on operations / widths the proposer never
  emphasized, to test whether self-play generalizes beyond its own curriculum.

## The natural-language bridge (GSM8K, MATH, MMLU-STEM)

`GSM8K` and `MATH` are the eventual targets, but they are *natural-language* math
(word problems, LaTeX). Reaching them — and any MMLU-STEM subset — requires adding
a language front-end (a text tokenizer feeding LAMb's latent core, or LAMb as an
arithmetic module a language model calls). That is a deliberate roadmap fork, not
a drop-in evaluation. Until then, the synthetic + length-generalization +
long-context suite above is the honest way to measure this model.
