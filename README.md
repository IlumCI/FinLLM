# LAMb — Latent Arithmetic Machine

A number-native reasoning model. LAMb has no language tokens: its vocabulary is
digits and arithmetic operators, it reasons in continuous latent space, and it emits
**programs** rather than prose. Arithmetic is not approximated by the network — it is
performed exactly by a residue algebra the network writes instructions for.

The design question behind it: **what if the parts of reasoning that can be exact
were made exact, so the learned part only has to do the part that genuinely
requires understanding?**

---

## Status

This project keeps an explicit ledger of what survives scrutiny and what does not,
because several headline claims have not. Full detail in `docs/ROADMAP.md`.

**Established.**

| result | evidence |
| --- | --- |
| Residue algebra is exact | `+ − × ÷` exact at every in-range magnitude; composition exact at **depth 6** (a 64-operand expression) with nothing learned |
| Exact rationals | 0 errors over 1200 random operations against Python's `Fraction`, including division |
| Single-error correction | **100%** of single-residue errors corrected, **0%** mis-corrected, over 4,000 trials |
| Program induction **from a supervised program** | **1.000** held-out answer accuracy, `sd` 0.000, on **all 5 seeds at both depth 2 and depth 3** — 7 instructions over 8 operands, executed exactly, `canonical_acc` 1.000 |
| Latent code becomes canonical | Stage B partner randomization: zero-shot **0.935** vs trained-pair 0.941, blank ~0.00, on a clean split |
| Trace supervision | **+6.7** over a converged, compute-matched baseline, *p* = 0.040, positive on all 5 seeds |

**Retracted.** *Program induction from outcomes alone* — the headline of this table
until a multi-seed study was pointed at it. It was `n=1`: at 5 seeds, depth 2 gives
**0.558 (sd 0.405)**, succeeding on 2 seeds and collapsing to ~0.25 on 3; at depth 3 it
gives **0.015**, against the supervised arm's 1.000, seed ranges disjoint,
permutation *p* = 0.0079. The claim that "gradients through exact arithmetic are
sufficient to induce a correct program from outcomes alone" is withdrawn: they suffice
sometimes over three instructions and never over seven. Details and the pre-registered
criterion in `docs/ROADMAP.md` 3a-xii.

Also retracted earlier: the parallel latent block's "+32.3 structural advantage" (it is
**−0.010 at *p* = 0.80** against a wall-clock-matched baseline); the claim that the
restructure reduces variance (the baseline was simply undertrained); a "61 seeds"
power figure computed from that same undertrained arm; SWITCH boundary tokens; the
premise that on-policy RL moves the latent block; and "zero GSM8K→GSM1K contamination
gap by construction", which does not survive a pretrained encoder.

**The pattern worth noting: everything exact survived, and nothing learned has survived
a multi-seed study unchanged.** Six retractions now, and every one of them came from
building the control that could expose it. The exact machinery has instead gone the
other way: the register machine, the algebra, the pointer masking and the rational
registers all scale from depth 2 to depth 3 at 1.000 with `sd` 0.000 — it is the
*learning signal*, not the execution, that keeps failing to generalise.

---

## Install

```bash
uv sync                              # exact pinned environment
uv run python -m lamb.lotus
```

`uv.lock` pins the dependency graph, torch included, and is platform-portable
(version pinned, no local build tag). That matters here: torch changes kernel
selection and reduction order between minor versions, and on a model this small those
move *accuracy*, not just low-order bits — a difference that would read as a finding.

Plain pip works too:

```bash
pip install -e . && pip install pytest
```

On **Windows with a CUDA card**, PyPI's `torch` wheel is CPU-only; add
`--index-url https://download.pytorch.org/whl/cu124`. On Linux the default wheel
already includes CUDA.

The Rust kernels are **optional** — there is an exact Python fallback, and
`USING_RUST=False` changes speed, not results. They profile at ~1.5% of a training
step, so they matter for self-play verifier throughput rather than for training:

```bash
pip install maturin && (cd rust && maturin develop --release)
```

**Device.** Every entry point honours `LAMB_DEVICE`, which overrides the config
without editing code:

```bash
LAMB_DEVICE=cuda python -m lamb.lotus --amp
```

For GPUs, `examples/LAMb_Colab.ipynb` sets up, **verifies the CUDA path**, and runs the
open experiments. Run its verification cell first: two bugs fixed here are unreachable
on CPU, and the cell exists so a bad result is never mistaken for a bad idea.

---

## The exact machinery

This is the part that has held up, and it needs no training at all.

### Residue arithmetic (`lamb/algebra.py`)

A value is carried as its residues modulo coprime moduli. Addition is cyclic
convolution of residue distributions, subtraction cross-correlation, multiplication
one small table per modulus — all carry-free, exact, and **differentiable on
distributions**, so a model unsure of a residue composes that uncertainty instead of
committing first.

**Moduli are chosen for the multiplicative order of 10, not for size.** The network
computes `n mod p` from digits as `Σ dᵢ·(10ⁱ mod p)`, whose coefficients repeat every
`ord_p(10)` positions — and that period is how many digit positions must be *seen*
before the modulus is learnable. Small primes are a trap: `ord(10)` is 16 mod 17, 18
mod 19, 22 mod 23. The default `(2,5,9,11,7,13,37)` has max period 6 in 84 units, and
three of them are the schoolbook divisibility rules.

### Exact rationals (`lamb/rational.py`)

Division is *not available* on plain residues: an inverse needs a divisor coprime to
every modulus, and the short-period moduli are exactly the set that denies that — not
one divisor from 2 to 12 is invertible. Carrying `(numerator, denominator)` makes
division multiplication with the operands swapped: exact, closed, still
differentiable. Decimals stop needing scale bookkeeping, because `3.25` **is**
`325/100`.

### Redundant residues (`RedundantResidueSystem`)

CRT has no locality — one wrong residue gives a wildly wrong number, so answer
accuracy is roughly per-residue accuracy raised to the modulus count. Carrying extra
moduli detects that, and dropping each in turn identifies and repairs it. At the
measured 0.985 per-residue accuracy this takes **0.900 → 0.996**, with no label and no
training. Where the evidence does not single out a culprit it returns `None` — an
admitted failure rather than a guess.

**What it does not do, measured on a trained model.** Redundancy detects an *ill-formed
code*, not a wrong answer. Asked to catch the register machine's real errors it
converted **1.3%** of them into refusals and left **44.6%** as confident wrong numbers,
because the failing runs emit coherent codes for a *different program* rather than
corrupted codes for the right one. It costs nothing to carry — the supervised arm is
1.000 on 5/5 seeds with three extra moduli — but it is a guard against corruption, not
against being wrong. Detail in `docs/ROADMAP.md` 3a-xiii.

### The register machine (`lamb/regmachine.py`)

Latents become **registers**; the model emits a program over them and the algebra
executes it.

```
registers 0..N-1   the operands, loaded exactly
instruction t      (op, ptr_a, ptr_b) -> writes register N+t
answer             the last register written
```

Registers are append-only and preallocated, so the dataflow is a DAG by construction,
a pointer mask makes reading an unwritten register *unrepresentable*, and the shape is
static (which is what compilers can fuse across). Execution is differentiable: a read
is a mixture over registers, an operation a mixture over composed results.

**What a non-differentiable executor cannot do, and what that turned out to be worth.**
PAL and Program-of-Thought call an external Python interpreter, so a wrong answer cannot
tell a pointer which way to move — the program can only be imitated or reinforced, and
RL on this latent block was measured inert. Here the answer loss does reach the pointer
heads *through exact arithmetic*, and the gradient is real: with the gold program
removed entirely, 2 of 5 seeds still reach 1.000 held-out accuracy at depth 2, on
programs that match the generator's on *zero* instructions.

**It does not scale, and the claim that it does is withdrawn.** At depth 2 the
answer-only arm averages 0.558 with `sd` 0.405 — **0.422** when the program it would
actually emit is executed rather than a soft mixture; at depth 3 — 7 instructions
instead of 3 — it averages **0.015** against the supervised arm's 1.000, with disjoint
seed ranges and permutation *p* = 0.0079. There is no partial credit: per seed it either
reaches 1.000 or falls to ~0.04, and `ptr_sharp` predicts which. Pointer choices per instruction grow as `|ops| · n_slots²`
(147 → 675), and the answer signal alone does not navigate that. The differentiable
executor is a real capability that a Python interpreter lacks; it is not a substitute
for knowing the program.

---

## The language bridge (`lamb/bridge.py`)

Language as a **peripheral**, not as the model. The core never learns a token
distribution.

| stage | learned? |
| --- | --- |
| text → frozen encoder → embeddings | no — cached once |
| text → quantities → registers | **no** — regex plus the fixed digit→residue map |
| embeddings → K latents (resampler) | yes |
| latents → program over registers | yes |
| program → answer | **no** — exact algebra |

The encoder is run **once** over a dataset and cached, so it never participates in
training and its size stops being a constraint. Quantities come out of the text by
rule, with decimals carried as scaled integers — a rounded operand is a wrong operand
in a ring. The learned surface is therefore the one thing that genuinely requires
understanding: **which computation to perform.**

```bash
uv sync --extra bridge                       # transformers is an optional extra
python -m lamb.bridge_train --coverage       # extraction recall, before any training
python -m lamb.bridge_train --encode         # run the frozen encoder once, cache it
python -m lamb.bridge_train --train          # supervised arm
python -m lamb.bridge_train --train --program-coef 0   # answer-only: no gold program
```

`transformers` is an **optional extra** on purpose. Every arithmetic number here was
produced under the pinned environment, torch moves accuracy on a model this small
between minor versions, and a text front-end should not be able to shift the ground
under a measurement it has nothing to do with.

**GSM8K ships gold programs, for 68% of the train split.** The premise that made the
bridge look like a leap — a word problem does not come with a program, so everything
rests on outcome-only induction — is true of prose in general and false of this dataset:
GSM8K's train solutions carry inline calculator annotations (`<<48/2=24>>`), an
operation with its operands and result in evaluation order. Compound ones
(`<<48*3+5=149>>`) decompose into two instructions rather than being discarded, which is
worth 19 points of coverage on its own. So the bridge is a measurement with a control:
the supervised arm recovers a program from the dataset's own annotations,
`--program-coef 0` removes it entirely.

That matters more than it did, because outcome-only induction **does not scale** (see
Retracted above). Supervision is the mechanism that works, and 61% of GSM8K supplies
it.

Three numbers are printed before training, because each caps what any model on top
could reach: annotation coverage, `operand_miss` (the share of rows naming a number
the extractor never found — extraction recall, nothing to do with the network), and
whether a recovered program actually *executes* to the dataset's stated answer.

**No accuracy is claimed.** It is built and unmeasured. `extract_quantities` cannot
see `1/2`, and emits spurious operands for dates and ordinals; quantity selection is
a missing component rather than a tuning detail. Constants `(1, 2, 3, 100)` are
preloaded as registers so that "half as many" — by this repo's own count the most
common operation in GSM8K — is expressible as `x / 2` rather than as a parser
deciding what "half" means.

---

## Running things

```bash
python -m lamb.train                   # self-play arithmetic trainer
python -m lamb.lotus                   # parallel supervised latent block
python -m lamb.coconut                 # sequential continuous thought (Stage A original)
python -m lamb.comm                    # latent inter-agent communication (Stage B)
python -m lamb.comm_pop                # partner randomization -> canonical code
python -m lamb.study --task d2g1       # paired multi-seed comparison with error bars
python -m lamb.study --task d3g1 --arms regmachine-supervised regmachine-answer-only
python -m lamb.bridge_train --coverage # GSM8K: extraction recall before any training
python -m lamb.memory_bench            # needle/passkey retrieval (infContext)
python -m lamb.ruler_bench             # RULER-style long-context battery
python -m lamb.poet                    # POET population of (environment, agent) pairs
```

---

## Measurement discipline

Several claims here died under scrutiny, each from something unpinned in a
comparison. The tooling that resulted is part of the repo:

- **`lamb/holdout.py`** — train/eval partition by a **hash of the problem**, not by
  seed. Disjoint seeds are not disjoint problems: the depth-2/1-digit space holds
  80k expressions and a 1000-step run draws 64k, so a seed-separated "held-out" set
  was 53% contaminated.
- **`lamb/study.py`** — paired multi-seed arms with **exact permutation tests** rather
  than intervals, and a `coconut-long` arm that matches *wall clock* rather than step
  count. It prints what the design can and cannot resolve, so a null is never mistaken
  for an absence.
- **`uv.lock`** — the environment is part of the comparison.

The rule the project now runs on: **a confound recorded honestly in a document is not
a control.** The gap between writing "this is a known confound" and spending the 50
minutes that closes it was two wrong headline claims.

---

## Tests

```bash
python -m pytest -q          # 237 tests
```

Exactness properties are tested **without a model in the loop** — if composition is
not exact by inspection, no amount of training rescues it.

---

## Repo map

```
lamb/
  algebra.py        residue arithmetic; redundant residues for detect + correct
  rational.py       exact rationals: division, and decimals
  alu.py            latent ALU: one value per slot, composed by the algebra
  regmachine.py     registers + emitted programs + differentiable execution
  bridge.py         frozen-encoder peripheral, exact quantity extraction
  bridge_train.py   GSM8K: loader, annotation->program alignment, trainer, eval
  holdout.py        train/eval partition of the problem space
  study.py          paired multi-seed arms, permutation tests, power
  lotus.py          parallel supervised latent block
  coconut.py        sequential continuous thought (Stage A original)
  comm*.py          latent inter-agent communication, transfer, populations
  memory_bench.py   needle/passkey retrieval;  ruler_bench.py  RULER-style battery
  eval.py           held-out accuracy, length generalization, test-time scaling
  model/            transformer, latent core, test-time memory, embeddings
  selfplay/         grammar, verifier, proposers, league, POET, GRPO
docs/               ARCHITECTURE.md  BENCHMARKS.md  ROADMAP.md
examples/           LAMb_Colab.ipynb
rust/               optional exact kernels (Python fallback is equivalent)
```

`docs/ROADMAP.md` is the working record, including the retractions and why each
happened. `docs/BENCHMARKS.md` explains why MMLU-style suites do not apply to a model
with 22 tokens and what does.

---

## License

Apache-2.0. See `LICENSE`.
