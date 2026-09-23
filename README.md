# LAMb — Latent Arithmetic Machine

LAMb is a **number-native, depth-recurrent, latent-reasoning autoregressive
model** that teaches itself arithmetic from zero external data. It is a research
scaffold that combines four current frontier lines into one small, CPU-runnable
system:

| Design goal | Mechanism in LAMb | Grounding |
| --- | --- | --- |
| **Reason in latent space** | A depth-recurrent core iterates a shared block `T` times (*vertical* latent thinking), feeding the hidden state back as its own next input. A **Coconut** scratchpad (`lamb/coconut.py`, Stage A) adds *horizontal* latent thinking: `K` continuous-thought positions between prompt and answer, each fed the model's own last hidden state. All reasoning is continuous; no intermediate tokens are decoded, and both `T` and `K` are knobs you turn up at inference. | Coconut (continuous thought, [2412.06769](https://arxiv.org/abs/2412.06769)); inference-time scaling for continuous reasoning ([2510.12167](https://arxiv.org/abs/2510.12167)); *Survey on Latent Reasoning* ([2507.06203](https://arxiv.org/html/2507.06203)) |
| **Communicate in pure arithmetic** | No BPE. Numbers are digit tokens with **Abacus** intra-number positions (reset per number, LSB-first) plus a **value channel** so a digit also knows its number's magnitude. | Abacus embeddings; *Numbers Already Carry Their Own Embeddings* ([2606.14108](https://arxiv.org/html/2606.14108v1)); BitTokens |
| **infContext** | A fixed-size **test-time neural memory** written during the forward pass by a surprise-gated delta rule (data-dependent forget/write gates). O(1) state, unbounded effective context. | Titans ([NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf)); ATLAS ([2505.23735](https://arxiv.org/abs/2505.23735)); DeltaNet |
| **Exploding self-improvement** | **Self-play**: a learning-progress bandit proposes tasks at the solver's frontier, an exact **Rust verifier** gives the reward, and the solver improves by expert iteration. Zero human data; the mastered difficulty frontier expands on its own. | Absolute Zero ([2505.03335](https://arxiv.org/pdf/2505.03335)); R-Zero; automatic curriculum learning |

The Python/PyTorch model is paired with a small **Rust extension** (`lamb_core`,
via PyO3/maturin) that owns the three hottest / correctness-critical paths — the
exact arithmetic verifier, the curriculum sampler, and a top-k memory store —
with pure-Python fallbacks so everything runs even if the extension is not built.

> Status: v0.1 research scaffold. The point is a faithful, end-to-end, runnable
> realization of the architecture on CPU, not a state-of-the-art solver.

## Install

```bash
# 1. Python deps (CPU-only torch keeps it light)
python -m pip install numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .

# 2. (optional but recommended) build the native kernels
python -m pip install maturin
cd rust && maturin build --release && \
  python -m pip install --force-reinstall target/wheels/lamb_core-*.whl && cd ..
```

If step 2 is skipped, LAMb transparently uses the Python fallbacks
(`lamb.backend()` reports `"python"` instead of `"rust"`).

## Run the self-play demo

```bash
python -m lamb.train                     # ~4 min on 4 CPU cores (bandit + expert iteration)
python -m lamb.train --steps 4000 --recurrent-steps 6   # pushes the frontier further
python -m lamb.train --use-memory        # enable the test-time memory in the core
python -m lamb.train --proposer grpo_hyper   # GRPO-trained hypernetwork proposer
python -m lamb.train --solver grpo           # add a GRPO/RLVR term on the solver
python -m lamb.train --red-queen             # Red Queen coevolution (league + novelty + relative fitness)
python -m lamb.train --red-queen --open-ended --proposer factored_hyper   # open-ended grammar (Step 2)
python -m lamb.poet                          # POET population of (env, agent) pairs (Step 3)
python -m lamb.poet_shared                    # shared-backbone POET (DoRA adapters by default; --adapter-type lora|hidden)
python -m lamb.memory_bench                   # long-context needle/passkey retrieval (infContext)
python -m lamb.ruler_bench                     # RULER/BABILong-style suite (NIAH, multi-key, variable tracking)
python -m lamb.coconut                         # Coconut continuous-thought reasoning + verifier best-of-N (Stage A)
python -m lamb.comm                            # latent inter-agent communication: message = a thought vector (Stage B)
python -m lamb.comm --sweep                     # channel-bandwidth (capacity) sweep with DRU noise
python -m lamb.comm_transfer                     # held-out-partner test: is the latent code private or shareable?
```

You will watch, from zero data:

- `frontier` — the mastered operand-digit sum — expand as the solver improves;
- `H(prop)` — proposer entropy — stay healthy (the bandit never collapses) while
  `top` cell shifts from easy to hard;
- held-out exact-match accuracy climb (1-digit is mastered quickly; 2-digit
  climbs steadily and mastered with more steps/latent-depth);
- **test-time latent-step scaling**: accuracy is robust as `T` grows, because
  the latent depth is randomised during training.

## What's here

```
lamb/                     Python package (torch)
  tokenizer.py            number-native tokenizer (digits, Abacus, value channel)
  model/
    embeddings.py         token + Abacus + value embeddings
    transformer.py        RMSNorm, RoPE attention, SwiGLU, pre-norm block
    latent_core.py        depth-recurrent latent reasoning (+ optional ACT halting)
    memory.py             Titans/ATLAS-style test-time neural memory
    lora.py               LoRA + DoRA adapters for shared-backbone POET (deeper adaptation)
    lamb.py               the LAMb model: forward / loss / batched solve
  selfplay/
    proposer.py           learning-progress bandit (default; pluggable interface)
    hyperproposer.py      GRPO-trained hypernetwork proposer, bandit-anchored
    factoredhyper.py      factored GRPO proposer for the open-ended space
    grammar.py            generative grammar of nested expressions
    openended.py          curricula: fixed grid + open-ended (MCC admission)
    grpo.py               GRPO utilities + solver RLVR objective (DAPO/Dr.GRPO options)
    league.py             Red Queen: solver league + relative-fitness metrics
    verifier.py           exact reward oracle (wraps the Rust kernels)
    loop.py               Absolute-Zero-style self-play trainer
  eval.py                 held-out accuracy, length generalization, test-time scaling
  coconut.py              Coconut continuous-thought reasoning + verifier-selected best-of-N (Stage A)
  comm.py                 latent inter-agent communication: speaker/listener + differentiable channel (Stage B)
  comm_transfer.py        held-out-partner test: cross-pair swap + fresh-partner learnability (Stage B analysis)
  train.py                CPU-first end-to-end entry point (single-agent self-play)
  poet.py                 POET population of (environment, agent) pairs (Step 3)
  poet_shared.py          shared-backbone POET: one backbone + per-environment adapters
  memory_bench.py         long-context needle/passkey retrieval benchmark (infContext)
  ruler_bench.py          RULER/BABILong-style suite: NIAH, multi-key, variable tracking
rust/                     lamb_core native kernels (PyO3/maturin)
  src/arith.rs            exact recursive-descent integer evaluator + verifier
  src/curriculum.rs       deterministic problem sampler
  src/store.rs            top-k associative store
tests/                    pytest suite (native, tokenizer, model, memory, self-play)
docs/                     ARCHITECTURE.md, ROADMAP.md
```

## Test

```bash
python -m pytest -q
```

The suite includes an **in-context associative-recall** test that binds a fresh
random key→label mapping every episode: passing it above chance is direct
evidence the test-time memory works, since the mapping cannot live in the weights.

## Proposers and solver optimisation

The proposer is a pluggable interface (`lamb.selfplay.proposer.BaseProposer`):

- **`bandit`** (default) — a learning-progress bandit, `softmax(beta * 4 s (1-s))`.
  Robust, stateless, cannot collapse. Recommended for the small demo grid.
- **`grpo_hyper`** — a **hypernetwork** mapping solver competence to the task
  distribution, trained by **GRPO** with a **KL anchor to the bandit** (the anchor
  is what averts the documented "proposer drifts to trivial/unsolvable tasks"
  collapse). It concentrates on the highest-learnability cell and tracks the
  frontier; its payoff over the tabular bandit is generalization on large/
  continuous task spaces.

Solver: **expert iteration** (default; teacher forcing on verified traces) or a
**GRPO/RLVR** term (`--solver grpo`, added on top after a warm start, with DAPO
dynamic sampling and an optional Dr.GRPO no-std normalization).

**Red Queen coevolution.** Step 1 (`--red-queen`): a solver **league** (historical
self-play), a **novelty** term (diversity maintenance), and relative-fitness
metrics — `dominance` (does the solver keep beating its past on the current
frontier?) and `forgetting`. On the bounded grid `dominance` decays to 0 as the
space saturates. Step 2 (`--open-ended --proposer factored_hyper`): a generative
grammar of nested expressions grown by **minimal-criterion admission**, with a
**factored** hypernetwork proposer (fixed `D+G+O` outputs over an unbounded
space). Measured: the space grows and the frontier advances with zero forgetting;
`dominance` becomes bounded by *solver capacity* rather than task-space
saturation — the ceiling moves from the curriculum to the model. Step 3
(`python -m lamb.poet`) raises that ceiling with a **POET population**: per-
environment specialist solvers, **transfer** across environments, and minimal-
criterion **reproduction** of harder environments. Specialists + transfer are the
capacity-scaling mechanism (rather than one larger model). Grounded in Digital Red
Queen and POET/MCC.

## Benchmarks

MMLU and general-LLM suites do not apply to a number-native math specialist. The
right battery — length generalization, latent-reasoning tasks, long-context
recall (BABILong / needle), and self-improvement curves — is described in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). A length-generalization eval ships in
`lamb.eval.length_generalization`.

## Extending

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the deeper-memory (ATLAS), full
Coconut continuous-thought, richer task-space, and language-bridge directions.

## License

Apache-2.0.
