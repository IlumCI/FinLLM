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

Also covered by `tests/test_memory.py` (in-context associative recall) and
`tests/test_memory_bench.py` (retrieval + extrapolation + ablation).

Larger, external suites to add next:

- **BABILong** — the long-context QA benchmark Titans/ATLAS report up to 10M
  tokens; the canonical infContext test.
- **RULER** — a battery of long-context retrieval/aggregation probes.

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
