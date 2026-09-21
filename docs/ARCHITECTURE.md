# LAMb architecture

This document specifies the mechanisms precisely. Notation: `B` batch, `T`
sequence length, `d` = `d_model`, `H` heads, `d_mem` memory width, `S` = number
of latent recurrent steps.

## 1. Number-native representation

A sequence is `[BOS] <problem> = <answer> [EOS]`. There is no BPE; the vocabulary
is `{PAD, BOS, EOS, =, +, -, *, (, )}` plus base-B digits (B=10 → vocab 19).

Each number is written **least-significant-digit-first** (reversed), which aligns
carry propagation with the left-to-right autoregressive scan and is the setting
under which learned positional schemes generalise to unseen operand lengths.

Three embedding channels are summed then RMS-normalised (`lamb/model/embeddings.py`):

1. **Token** — `nn.Embedding(vocab, d)`.
2. **Abacus** — `nn.Embedding(max_number_len+1, d)` indexed by the digit's
   position *within its own number* (index 0 = "not a digit"). Reset at every
   number boundary.
3. **Value** — for a digit that belongs to a **problem operand**, features of the
   operand's magnitude `v`:
   `[ sign(v)·log(1+|v|),  sin(v·ω_k),  cos(v·ω_k) ]` for log-spaced `ω_k`,
   projected to `d`. Gated by a mask that is **zero on the answer side**, so
   magnitude information can never leak the label being predicted.

## 2. Depth-recurrent latent core (`lamb/model/latent_core.py`)

```
h = prelude(x)                         # n_prelude blocks
s = h
repeat S times:                        # the latent "thinking" loop
    s = recurrent_block( s + x )       # input re-injected each step (Huginn-style)
    s = s + memory(s)                  # optional test-time memory branch
out = coda(s)                          # n_coda blocks
```

- The recurrent block(s) share weights across the `S` iterations, so "thinking
  longer" costs compute, not parameters. This is the continuous-latent-reasoning
  mechanism: intermediate states are never decoded to tokens.
- `S` is provided at call time (`n_steps`), so **test-time compute scales** by
  spending more latent steps. Training randomises `S ∈ [train_min_steps,
  train_max_steps]` so the model is robust across the inference budget.
- Blocks are standard pre-norm transformer blocks: RMSNorm → RoPE causal
  self-attention → RMSNorm → SwiGLU, with residuals. Causal + key-padding masks
  are combined; padding is always a suffix, so no query row is ever fully masked.

### Adaptive halting (optional)

With `adaptive_halting=True` the core runs Graves-style ACT: each position
accumulates a halting probability, stops early once it crosses `1-ε`, and a
ponder cost (`ponder_cost`) discourages over-thinking. This makes the number of
latent steps per position learned and input-dependent. Off by default.

## 3. Test-time neural memory (`lamb/model/memory.py`)

A fixed-size associative memory `M ∈ R^{d_mem×d_mem}` updated during the forward
pass. Per token `t` (causally — read before write):

```
read_t   = q_t · M
surprise = v_t − (k_t · M)              # prediction error under current M
M        = α_t · M + η_t · (k_t ⊗ surprise)
```

`q_t, k_t, v_t` are projections of the hidden state (`q,k` ℓ2-normalised). The
gates `α_t` (retention/forget, sigmoid) and `η_t` (write rate, softplus) are
**data-dependent** — the network decides what to keep. This is the DeltaNet
linearisation of the Titans/ATLAS "learn to memorise at test time" update:
state size is independent of sequence length, so effective context is unbounded
and per-token cost is O(1). Gradients flow through the whole recurrence, so the
projections learn *how* to use the memory.

Evidence it works: `tests/test_memory.py` trains a memory-only model on
in-context associative recall with a **fresh random key→label map per episode**;
it exceeds chance, which is only possible by writing/reading the bindings at test
time.

## 4. The model (`lamb/model/lamb.py`)

`embeddings → core → RMSNorm → tied LM head`. Loss is next-token cross-entropy
**masked to answer positions only** (plus EOS), so the model is scored on
producing the answer, not on copying the problem. `solve()` does batched greedy
decoding via left-padding (all prompts in a difficulty cell share a length),
tracking Abacus indices incrementally and decoding LSB-first digits back to an
integer string.

## 5. Self-play self-improvement (`lamb/selfplay/`)

The reward is verifiable and the data is self-generated — no human labels.

Each training step:

1. **Propose.** The bandit samples `B` difficulty cells `(op, a_digits,
   b_digits)` from `p = (1−ε)·softmax(β·L) + ε·uniform`, where the learnability
   `L_i = 4·s_i·(1−s_i)` peaks at intermediate solver success `s_i` (smoothed).
2. **Generate & supervise.** The Rust sampler draws a concrete `(problem, exact
   answer)` per cell; one teacher-forcing SGD step is taken on the solver, mixed
   with a fraction of replayed **verified-solved** traces (expert iteration).
3. **Roll out.** The solver greedy-decodes the fresh problems; the exact verifier
   scores them — the only reward signal.
4. **Co-evolve.** Per-cell success EMA is updated; solved traces enter the replay
   buffer; the bandit's distribution reacts to the new learnability.

Because a mastered cell's `L → 0`, probability mass leaves it automatically and
the frontier marches to harder cells — the observable signature of self-
improvement. Unlike a REINFORCE policy, the bandit cannot collapse (its
distribution is a fixed function of learnability plus a uniform floor).

Optimisation: AdamW, warmup + cosine decay to 10% of peak, gradient clipping.

## Defaults

Tiny CPU-first model: `d_model=128`, 1 prelude / 1 recurrent / 1 coda block,
`recurrent_steps=4`, ~0.5M parameters. Grid: `{+,-} × {1,2 digits}²`. These reach
full 1-digit mastery in minutes and climb on 2-digit; larger `d_model`, more
`recurrent_steps`, and more `steps` extend the mastered frontier.
