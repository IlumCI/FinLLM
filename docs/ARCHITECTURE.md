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

The gates carry a **long-term-memory inductive bias**: at initialisation
retention starts high (`alpha ~ 0.95`) and the write rate low, so a binding
survives across long spans of uninformative tokens until the model learns what to
write. This is what lets retrieval extrapolate to lengths far beyond training.

Evidence it works: `tests/test_memory.py` trains a memory-only model on
in-context associative recall with a **fresh random key→label map per episode**;
it exceeds chance, which is only possible by writing/reading the bindings at test
time. The **`python -m lamb.memory_bench`** needle/passkey benchmark
(`lamb/memory_bench.py`) then stresses it at length: trained at length 48,
retrieval stays near-perfect to 4x and degrades gracefully to 8x, while a
memoryless ablation is at chance -- and a capacity curve shows the honest
fixed-state tradeoff (accuracy falls as bindings approach `d_mem`).

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

## 6. GRPO co-evolution (optional)

Both the proposer and the solver can be trained by **GRPO** (Group Relative
Policy Optimization): critic-free, with a sample's advantage computed relative to
a *group* drawn from the same state, `A_i = (r_i - mean) / std`. Utilities live in
`lamb/selfplay/grpo.py`, with **DAPO** dynamic sampling (drop zero-variance
groups) and an optional **Dr.GRPO** no-std normalization (avoid difficulty bias).

**Solver GRPO/RLVR** (`solver_algo="grpo"`). For each problem, sample `G` answers,
score them with the exact verifier, form group-relative advantages, and add
`-A_i * sum_t log pi(o_{i,t}) + beta * KL(pi || pi_ref)` (k3 KL to a periodically
refreshed reference) to the expert-iteration cross-entropy. The expert term is a
warm start that avoids the RL cold-start where nothing is ever solved.

**Hypernetwork proposer** (`proposer_kind="grpo_hyper"`,
`lamb/selfplay/hyperproposer.py`). A small hypernetwork maps the solver-competence
state `s` to task-distribution logits; the step's proposals form one GRPO group
with learnability rewards; the update is anchored by `KL(pi || bandit)`. The KL
anchor to the non-collapsing bandit is what prevents the drift-to-trivial/
unsolvable failure that plagues bare self-play proposers. On the small demo grid
the policy concentrates on the highest-learnability cell (the correct frontier
target) with the `eps` floor preserving coverage; its advantage over the tabular
bandit is generalization across large/continuous task spaces, where a per-cell
table does not fit. The bandit remains the recommended default.

## 7. Red Queen coevolution (open-endedness, Step 1)

Enabled with `red_queen=True`. The reactive autocurriculum (a fixed function of
solver competence) saturates once a bounded task space is mastered. Red Queen
dynamics make improvement *relative*: the solver must keep dominating its own
past on an ever-advancing frontier. Step 1 adds the minimal, measurable machinery
(`lamb/selfplay/league.py`):

* **Solver league** -- periodic frozen snapshots of the solver (historical
  self-play; keeps the oldest baseline plus the most recent). Cheap for a tiny
  model.
* **Novelty / diversity term** -- per-cell EMA visitation drives a count-based
  novelty bonus that up-weights rarely-proposed cells, both at sampling time and
  in the (learned) proposer's reward. This is the "diversity maintenance"
  ingredient Digital Red Queen (arXiv:2601.03335) pairs with historical
  self-play.
* **Relative-fitness metrics** -- `dominance` (current minus best-past accuracy on
  the *current* frontier; > 0 means the solver is still pulling ahead where the
  proposer now pushes) and `forgetting` (oldest-snapshot minus current accuracy on
  a fixed easy set; > 0 means regression). On a bounded grid `dominance` decays to
  0 as the space saturates -- the signal that an open-ended task space is needed.

## 8. Open-ended task grammar (Red Queen Step 2)

Enabled with `open_ended=True`. The fixed grid is replaced by a generative
grammar (`lamb/selfplay/grammar.py`) of nested arithmetic expressions whose
complexity grows without bound along depth, operand width, and operator set. The
descriptor space (`lamb/selfplay/openended.py`) is **append-only** and grown by
**minimal-criterion admission**: a descriptor's harder neighbours (deeper, wider,
richer ops) are admitted only once the descriptor itself is mastered, with the
exact verifier guaranteeing every admitted task is well-posed. The reachable
space therefore expands purely as a function of the solver's own competence.

Because the descriptor set grows, the grid hypernetwork (fixed output per cell)
no longer fits. The **factored hypernetwork proposer** (`factoredhyper.py`) emits
independent softmaxes over depth, digits, and operator set from a fixed-width
competence context, so a combinatorial space of `D*G*O` descriptors costs only
`D+G+O` outputs -- the concrete sense in which the hypernetwork beats a table
once the space is open-ended. It is GRPO-trained with a KL anchor to a
uniform-over-admitted factored prior.

Measured behaviour on the tiny CPU model: the space grows (tasks 1 -> 6), the
mastered frontier advances (0 -> 4), and `forgetting` stays 0. Unlike Step 1's
fixed grid -- where `dominance` decayed to 0 from *task-space saturation* --
`dominance` here is bounded instead by *solver capacity* (the tiny model caps out
around depth-2 nested expressions). The ceiling has moved from the curriculum to
the model, which is the intended effect; sustained positive `dominance` is a
scale-up (larger solver) result. Step 3 (a POET-style population of (task,
solver) pairs with transfer) is the next roadmap item.

## 9. POET population (Red Queen Step 3)

`lamb/poet.py` (`python -m lamb.poet`). A population of `(environment, agent)`
pairs, where an environment is a grammar descriptor and each agent is its own
specialist LAMb solver. Per iteration: **optimize** every agent on its own
environment (expert iteration on exact answers); periodically **transfer** (if
another agent beats an environment's incumbent by a margin, it replaces the
incumbent -- innovations cross between environments); **reproduce** (competent
environments spawn harder, novel grammar-neighbour children, each seeded by its
parent's agent and admitted only if the seed does not already solve it -- the
minimal criterion); and **graduate** the easiest environments when over capacity.

Reproduction is gated by **behavioural novelty** (`lamb/selfplay/novelty.py`), not
descriptor dedup: a candidate environment is characterised by its seed agent's
competence across latent-step budgets, the answer length it demands, and the gain
from extra thinking (a behaviour-characterisation vector), and admitted only if it
is far, in behaviour space, from an archive of admitted environments (novelty
search / diversity maintenance) -- so structurally-new but behaviourally-redundant
environments are rejected.

This is the capacity-scaling answer to Step 2's finding: rather than one larger
model, a *population of specialists plus transfer* raises the ceiling. Reported
metrics are the attempted frontier (hardest environment present), the peak
conquered frontier (hardest environment mastered), and the transfer count.
Measured on the tiny CPU population: the population grows and its attempted
frontier advances while transfers fire, and it conquers base environments (e.g.
`d1g1o1` to ~0.95); how high the *conquered* frontier climbs is set by per-agent
optimization budget and agent size -- i.e. it scales. Following POET / Enhanced
POET (arXiv:1901.01753, 2003.08536).

**Shared-backbone variant** (`lamb/poet_shared.py`, `python -m lamb.poet_shared`):
one shared backbone plus a tiny low-rank adapter per environment (via the
`hidden_adapter` seam in `LAMb.forward`, which modulates the hidden state before
the LM head). A population of N environments then costs one backbone + N adapters
(~21% of N full models at capacity 5), and the shared backbone -- updated by every
environment -- accumulates cross-environment skill, so a reproduced environment
inherits a competent backbone (implicit transfer); explicit transfer just copies
the small adapter. The tradeoff is honest: parameter-efficient, but a final-layer
adapter specialises less than a full model. **Deeper adapters** (`lamb/model/lora.py`,
default) fix this by adapting the core's attention linears via forward hooks (one
adapter set per environment, selected per forward): **LoRA** (additive low-rank
delta) and **DoRA** (weight-decomposed, arXiv:2402.09353 -- the low-rank update
adapts the *direction* while a separate magnitude vector is trained, `W' = m . (W
+ s.BA)/||W + s.BA||`, with the norm detached in backward). Both are exactly the
identity at initialisation. At equal budget the best specialist reaches ~0.77
(DoRA) vs ~0.53 (LoRA) vs ~0.50 (final-hidden) -- DoRA wins for one magnitude
vector per layer on top of LoRA. `--adapter-type {dora,lora,hidden}`.

## 10. Coconut continuous-thought reasoning (Stage A, `lamb/coconut.py`)

The recurrent core (§2) thinks *vertically* — iterate a block `S` times at fixed
positions. Coconut adds *horizontal* latent thinking: insert `K` scratchpad
positions between the prompt and the answer whose input embedding is the model's
own last hidden state, fed back in continuous space and never decoded. They enter
the attention context, so the answer can attend to them as a working memory.

Mechanism (methods on `LAMb`):

```
x = embed([BOS] expr =)                       # left-padded prompt embeddings
repeat K times:                               # _roll_thoughts
    h = core(x)                               # residual stream
    t = thought_norm(h[:, -1]) + thought_marker   # bounded feedback + latent tag
    x = concat(x, t)                          # append a latent position (no token)
logits = readout(core(concat(x, answer)))     # coconut_logits: BPTT through thoughts
```

- **Feedback hygiene.** `thought_norm` (an RMSNorm) maps a residual-stream hidden
  state into input-embedding statistics, so the fed-back state has bounded
  magnitude — the one stability trick the Coconut paper relies on (they reuse the
  final norm). `thought_marker` is a learned additive tag (zero-init, so the
  interface is inert at start) letting the model tell a thought from a token — a
  latent analogue of Coconut's `<bot>/<eot>` boundaries. The thought is the
  **pre-readout** residual stream: `norm_f`+`lm_head` is the verbalisation head,
  which the thought deliberately bypasses (nothing is decoded to a token).
- **`K=0` is exactly ordinary teacher forcing.** RoPE is relative, so left-padding
  the prompt does not change any real-position output; `tests/test_coconut.py`
  asserts the answer cross-entropy matches `LAMb.forward` to `1e-5`.
- **Test-time budget.** `K` is chosen at call time; training randomises
  `K ∈ [train_min_thoughts, train_max_thoughts]` (the §2 recurrent-depth recipe,
  applied to thoughts), so greedy accuracy rises as more thoughts are spent.
- **Cost.** `K` sequential core forwards to build the thoughts, plus one for the
  answer (`n+1` forward passes); gradients flow back through the whole chain.

Training is plain next-token cross-entropy on the answer, back-propagated through
the thought chain. Coconut's gains normally need a curriculum distilling the
thoughts from **language** CoT; LAMb has none, and Coconut's own `w/o curriculum`
ablation is exactly this setting (language-domain: underperforms no-thoughts,
diagnosed by arXiv:2510.12167 as homogeneous latents). Measured here on the tiny
CPU model (depth-2 nested expressions, ~0.28M params): the scratchpad becomes
useful anyway — greedy exact-match `K=0: 0.39 → K=3: 0.52` — because the substrate
is number-native and the tasks are genuinely multi-step. `collapse_metric` (mean
pairwise cosine of the thought vectors, the 2510.12167 homogeneity signal) falls
cos `0.88 → 0.27` as thoughts specialise. Hard STaR-filtering to the
verifier-solved subset was tried and rejected (it starves the tiny model and
collapses the thoughts; exact labels already carry the verifier's information at
train time).

**Verifier-selected best-of-N (test-time self-improvement).** arXiv:2510.12167
showed dropout-diverse latent trajectories raise Pass@N monotonically but could
not *select* the winner (trained reward models barely beat chance). LAMb's exact
verifier is that missing selector: perturb the latent phase with dropout, decode
each trajectory greedily (dropout off during answer emission — perturb the
thoughts, not the tokens), keep any the verifier accepts. Realised accuracy
`N=1: 0.52 → N=8: 0.66` — searching the latent space and verifying, needing no
labels at deploy. Nothing is verbalised; the verifier checks only the emitted
number, so the thoughts stay a blackbox. Stage B reuses this continuous-input path
for latent inter-agent messages (a message *is* a thought vector).

## 11. Latent inter-agent communication (Stage B, `lamb/comm.py`)

Stage A gave one agent a continuous-thought scratchpad. Stage B makes the thought
a **message between two agents**: the vector one agent produces is consumed by
another as an input embedding -- receiving is literally thinking someone else's
thought, through the same `_roll_thoughts` interface.

**Split task.** A problem neither agent can solve alone. The **speaker** sees
operand `X`; the **listener** sees the operator and operand `Y` and must emit
`X op Y`. The listener's half never contains `X`, so it cannot answer without the
message -- a clean, verifiable information-asymmetry game over the model's native
domain.

**Channel.** The speaker rolls `n_msg` Coconut thoughts from its view of `X`;
those continuous vectors pass through a differentiable `Channel`
(`down: d→c`, `up: c→d`, RMSNorm, learned `recv_marker`) into the listener, which
injects them as latent positions, optionally rolls its own thoughts, and decodes
the answer. `c` is the **bandwidth** (bottleneck width). With `channel_noise>0`
the code is tanh-bounded and Gaussian noise is added in training -- the DIAL
**DRU** (arXiv:1605.06676), which gives the channel finite capacity and pushes the
sender toward a robust code; noiseless (`c=d`) is a clean high-capacity channel.

**Training is differentiable inter-agent learning (DIAL).** One joint objective --
the listener's answer cross-entropy -- is back-propagated *through the message*
into the speaker, so the speaker learns to encode `X` in latent form purely from
the listener's downstream loss. No token is ever exchanged; the channel is pure
latent, a blackbox (the whole point of a latent-space model -- nothing is
verbalised).

**Diagnostics, following the emergent-communication literature.** Positive
*signalling* (the message depends on `X`) and positive *listening* (the message
changes the listener's output) are not the same, and high reward can hide a
receiver that ignores the channel (arXiv:1903.05168). So two measures are
reported:

- **Zeroed-message ablation** (the causal listening test): blank the channel at
  eval; the gap `comm − blank` is the channel's causal value.
- **Message diagnostics**: `signal_std` (variation of the message across inputs;
  ~0 => the speaker emits a constant) and `msg_cos` (mean pairwise cosine of the
  messages; ~1 => collapsed to one direction -- the representational-collapse
  detector of arXiv:2604.03809).

Measured on the tiny CPU pair (`X,Y` 1-digit, `+/−`, `n_msg=2`, noiseless): the
speaker learns a perfect latent code -- exact-match `comm 1.000` vs blanked
`0.098` (the ~0.10 guess-`X` prior), a `+0.90` causal channel gain, with
`signal_std` rising and `msg_cos` falling as the code forms. Two agents solve, in
pure latent, a task neither can solve alone. Under DRU noise `0.5` the bandwidth
sweep shows a sharp capacity threshold -- `width 1/2/4 -> 0.11/0.11/0.10` (stuck
at the prior) then `width 8/96 -> 1.000/1.000`: a 1-digit operand needs `>= 8`
noisy channel dimensions to transmit, the capacity--accuracy tradeoff a noiseless
channel hides. This is the
substrate the roadmap's language-bridge deliberately is not: agents that reason
*and* communicate without ever leaving latent space. Grounded in DIAL
(arXiv:1605.06676), Coconut (arXiv:2412.06769), and the latent-agent-communication
line (Interlat arXiv:2511.09149, DiffMAS arXiv:2604.21794).

**Held-out-partner test** (`lamb/comm_transfer.py`, `python -m lamb.comm_transfer`).
Because the pair trains together, the emergent code could be a *private*
co-adaptation rather than a shareable protocol (the zero-shot-coordination
problem; Other-Play, arXiv:2003.02979). Two probes settle it. *Cross-pair swap*:
train two independent pairs to `comm 1.000` each, then give each listener the
other's speaker -- accuracy collapses to `0.004 / 0.041`, *below* the `0.076`
blank prior (a foreign code actively misleads, worse than silence).
*Fresh-partner learnability*: freeze one speaker and train a new receiver against
it -- it reaches `1.000`. So the protocol is private and strongly co-adapted, yet
a well-formed language a new partner can *learn* -- idiosyncratic, not canonical.

**Partner randomization** (`lamb/comm_pop.py`, `python -m lamb.comm_pop`) is that
fix, and it works. Train a *population* -- `P` speakers and `Q` listeners (each
listener its own channel) -- pairing them at random, with the diagonal `(i, i)`
pairings **held out** of training; because every speaker must be understood by
many listeners and every listener must decode many speakers, the code is pressured
to be canonical rather than private. The held-out pairings are then evaluated
zero-shot (agents that never co-trained). On the tiny 3x3 CPU population the
held-out zero-shot accuracy climbs `0.33 -> 0.81 -> 0.89 -> 1.000`, matching the
trained pairings (`1.000`) and *far* above both the `~0.11` blank prior and the
single co-adapted pair's `0.004` swap. Partner randomization converts the private
code into a shared one -- the Other-Play prescription (arXiv:2003.02979) realized:
a population of agents that reason and coordinate in a common latent language,
none of them ever leaving latent space.

## Defaults and scaling

Tiny CPU-first model: `d_model=128`, 1 prelude / 1 recurrent / 1 coda block,
`recurrent_steps=4`, ~0.5M parameters. Grid: `{+,-} × {1,2 digits}²`. These reach
full 1-digit mastery in minutes and climb on 2-digit; larger `d_model`, more
`recurrent_steps`, and more `steps` extend the mastered frontier.

**Hardware (GPU + CPU + RAM hybrid, `lamb/device.py`).** The architecture is
device-agnostic: `resolve_device` auto-selects `cuda` > `mps` > `cpu`, `Amp` adds
bf16/fp16 autocast (+ gradient scaler) automatically on CUDA and stays fp32 on CPU
(no regression), while the exact Rust kernels run on CPU alongside the accelerator
and RAM holds the buffers/stores. Every entry point takes `--device/--amp/--threads`;
`--scale {tiny,small,base,large}` grows width/depth/batch together (0.28M → 1.98M →
7.90M → 31.5M params). The GPU path is unit-tested on CPU (forced bf16 autocast) and
picked up automatically by `--device auto` on a CUDA box.
