"""Configuration dataclasses for LAMb.

Two configs: :class:`ModelConfig` (architecture) and :class:`TrainConfig`
(self-play optimisation). Defaults are tuned for a *tiny CPU-first* model that
trains end-to-end in a couple of minutes while exercising every component:
number-native embeddings, the depth-recurrent latent core, the test-time neural
memory, and the self-play loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    # Vocabulary is owned by the tokenizer; filled in from it at build time.
    vocab_size: int = 19
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256

    # Depth-recurrent latent core: `n_prelude` fixed blocks -> the recurrent
    # block is applied `recurrent_steps` times (this is the latent "thinking"
    # budget, tunable at test time) -> `n_coda` fixed blocks read it out.
    n_prelude: int = 1
    n_recurrent: int = 1
    n_coda: int = 1
    recurrent_steps: int = 4  # latent "thinking" budget; more steps help multi-digit carry
    inject_input: bool = True  # re-inject the embedded input each latent step

    # Number-native embedding.
    max_number_len: int = 24  # cap on Abacus intra-number position index
    value_freqs: int = 6      # multi-scale sinusoidal features of operand magnitude

    # Test-time neural memory (Titans/ATLAS-style delta-rule fast weights).
    use_memory: bool = False  # off for the short-sequence arithmetic demo; see eval
    d_mem: int = 64

    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_embeddings: bool = True

    # Adaptive computation (ACT-style ponder). Off by default; when on, the core
    # learns how many latent steps to spend per position.
    adaptive_halting: bool = False
    ponder_cost: float = 1e-2
    max_recurrent_steps: int = 8


@dataclass
class TrainConfig:
    steps: int = 1500
    batch_size: int = 96
    lr: float = 2e-3
    weight_decay: float = 0.01
    warmup: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    amp: Optional[bool] = None  # mixed precision; None => auto (on for CUDA, off for CPU)

    # Randomised latent depth during training: sampling the number of recurrent
    # "thinking" steps each step makes the model robust to the test-time budget,
    # so accuracy rises monotonically as you spend more latent steps at inference
    # (the recurrent-depth training recipe).
    sample_train_depth: bool = True
    train_min_steps: int = 1
    train_max_steps: int = 6

    # Self-play / curriculum.
    ops: Tuple[str, ...] = ("+", "-")
    max_digits: int = 2          # ceiling on operand width the proposer may reach
    start_digits: int = 1        # difficulty frontier begins here
    mastery_threshold: float = 0.9  # a cell counts as mastered above this solve rate
    proposer_kind: str = "bandit"   # "bandit" | "grpo_hyper"
    proposer_temp: float = 6.0   # softmax sharpness over per-cell learnability
    proposer_warmup: int = 60    # uniform-coverage steps before the proposer engages
    proposer_eps: float = 0.35   # exploration floor: fraction of cells drawn uniformly

    # GRPO hypernetwork proposer (proposer_kind == "grpo_hyper").
    hyper_hidden: int = 64
    hyper_lr: float = 5e-3
    hyper_kl_coef: float = 0.3    # KL anchor to the (non-collapsing) bandit
    hyper_entropy_coef: float = 0.05
    # Note: on this small grid the GRPO policy concentrates sharply on the single
    # highest-learnability cell (the correct frontier target); the proposer_eps
    # floor preserves coverage and it tracks the frontier as it moves. Its payoff
    # over the tabular bandit is generalization on large/continuous task spaces.

    # Solver optimisation: expert iteration (default) or add a GRPO/RLVR term.
    solver_algo: str = "expert"  # "expert" | "grpo"
    grpo_warmup: int = 120       # expert-only steps before GRPO engages (warm start)
    grpo_problems: int = 16      # distinct problems per GRPO step (subset of the batch)
    grpo_group_size: int = 4     # sampled answers per problem
    grpo_temperature: float = 1.0
    grpo_kl_coef: float = 0.02
    grpo_coef: float = 1.0       # weight of the GRPO term added to the expert CE loss
    grpo_normalize_std: bool = True   # False = Dr.GRPO-style (no difficulty bias)
    grpo_dynamic_sampling: bool = True  # DAPO: drop zero-variance groups
    grpo_ref_update_every: int = 100  # steps between reference-policy refreshes

    # Expert iteration replay buffer (STaR-style): verified solver traces.
    buffer_capacity: int = 20000
    expert_fraction: float = 0.25  # share of each batch drawn from solved-trace replay

    # Red Queen coevolution (Step 1): a solver league (historical self-play) plus
    # a novelty/diversity term on task selection, with relative-fitness metrics.
    red_queen: bool = False
    league_capacity: int = 4
    league_snapshot_every: int = 200  # steps between frozen solver snapshots
    novelty_coef: float = 0.5         # diversity-maintenance weight on task selection
    novelty_decay: float = 0.98       # EMA decay for per-cell visitation

    # Open-ended task grammar (Red Queen Step 2): nested expressions with
    # minimal-criterion admission, so the task space grows without bound.
    open_ended: bool = False
    oe_max_depth: int = 4     # cap on expression nesting depth (space grows up to here)
    oe_max_digits: int = 4    # cap on operand width
    admit_every: int = 25     # steps between MCC admission passes
    factored_hidden: int = 64  # hidden width of the factored hypernetwork proposer

    eval_every: int = 100
    eval_batch: int = 256
    log_every: int = 50
    ckpt_dir: str = "runs"

    def difficulty_grid(self) -> List[Tuple[str, int, int]]:
        """All (op, a_digits, b_digits) cells the proposer can select among."""
        grid: List[Tuple[str, int, int]] = []
        for op in self.ops:
            for a in range(self.start_digits, self.max_digits + 1):
                for b in range(self.start_digits, self.max_digits + 1):
                    grid.append((op, a, b))
        return grid


@dataclass
class CoconutConfig:
    """Stage A: Coconut single-agent continuous-thought reasoning.

    A "thought" is a latent scratchpad position inserted between the prompt and
    the answer whose input embedding is the model's own last hidden state -- pure
    latent, never decoded. Coconut's published gains come from a curriculum that
    distils the thoughts from *language* chain-of-thought steps; LAMb has no
    language traces (only a verifiable final answer). The training signal is
    therefore plain answer-NLL back-propagated through the thought chain, with the
    thought count randomised so accuracy is robust to -- and scales with -- the
    test-time budget. Because the task is genuinely multi-step and the substrate
    is number-native, the scratchpad becomes useful without a language curriculum
    (in contrast to Coconut's language-domain ``w/o curriculum`` ablation).

    LAMb's asset the Coconut / arXiv:2510.12167 line lacked is an **exact
    verifier**. That paper showed sampling dropout-diverse latent trajectories
    raises Pass@N monotonically but could not *select* the right one (its trained
    reward models barely beat chance). LAMb selects with the verifier, so Pass@N
    becomes realised accuracy -- a test-time self-improvement loop (search the
    latent space, verify, keep the winner) that needs no labels at deploy time.
    The verifier only checks the emitted number; the thoughts stay a blackbox.
    """

    steps: int = 1000
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.01
    warmup: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    amp: Optional[bool] = None  # mixed precision; None => auto (on for CUDA, off for CPU)

    # Task space: nested-expression grammar (multi-step is where a latent
    # scratchpad can help). Depth 2 = ``(a op b) op (c op d)``.
    depth: int = 2
    digits: int = 1
    ops_key: int = 0             # 0 -> (+,-); 1 -> (+,-,*)

    # Latent thought budget. Randomising it over training makes accuracy robust
    # to the test-time budget (the recurrent-depth recipe, applied to thoughts),
    # which is what lets accuracy scale with more thoughts at inference.
    n_thoughts: int = 3
    sample_train_thoughts: bool = True
    train_min_thoughts: int = 0
    train_max_thoughts: int = 4

    # Test-time verifier-selected best-of-N: perturb the latent phase with dropout
    # to draw diverse trajectories, decode each greedily, keep any the verifier
    # accepts. ``thought_dropout`` is the diversity source (0 => deterministic).
    thought_dropout: float = 0.2

    # Evaluation.
    eval_every: int = 150
    eval_tasks: int = 256
    eval_thoughts: Tuple[int, ...] = (0, 1, 2, 3, 4)  # K sweep: accuracy vs #thoughts
    eval_bestof: Tuple[int, ...] = (1, 2, 4, 8)       # N sweep: Pass@N via the verifier
    log_every: int = 50
    ckpt_dir: str = "runs"

    def max_answer_len(self) -> int:
        # Generous bound on answer token length across the reachable space.
        return min(24, 2 + self.digits * (2 ** min(self.depth, 3)))


@dataclass
class LotusConfig:
    """Stage A, restructured: LOTUS-style *parallel supervised* latent reasoning.

    Coconut's continuous thoughts are generated **autoregressively** -- one latent
    at a time, each conditioned on the last -- and supervised only by the final
    answer. That family's gap to explicit chain-of-thought *widens* with scale
    (arXiv:2606.31779 measures -0.1 pts at 124M, -2.3 at 1B, -9.2 at 3B: a
    performance cliff). The looped/parallel-supervised family stays flat (-1.5 at
    3B), so it is the one worth scaling.

    So: ``n_latent`` latent positions are appended after the prompt **all at once**
    and refined by ``loops`` passes through the shared core (cost is O(loops)
    forwards regardless of how many latents there are -- the scalability argument),
    and **every latent position is supervised directly** through the LM head.

    LOTUS supervises latents against gold chain-of-thought tokens. LAMb has no
    language, but it has an exact evaluator, so it generates a gold **numeric**
    trace -- the intermediate sub-expression values -- for free, with no language
    and no external data (:meth:`TaskGrammar.sample_with_trace`). At inference the
    latent positions are never decoded: only the answer is emitted, so the model
    stays a blackbox. The trace is a training signal, not an output.
    """

    steps: int = 1000
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.01
    warmup: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    amp: Optional[bool] = None

    # Task space (matched to CoconutConfig so the two arms are comparable).
    depth: int = 2
    digits: int = 1
    ops_key: int = 0
    # Operand-width range for training. 0 means 'just `digits`'. A range is
    # what makes a width-extrapolation claim testable at all: the residue of
    # a number is a digit sum whose coefficients repeat every ord_p(10)
    # positions, so a run has to *see* that many positions before the pattern
    # is learnable -- and every wider number is then free. Training at one
    # width can only ever teach position 0.
    digits_min: int = 0
    digits_max: int = 0

    # The parallel latent block.
    n_latent: int = 8        # L latent positions, computed together
    loops: int = 3           # R refinement passes over the block
    # Trace tokens supervised per latent position (multi-token prediction on the
    # latent block, arXiv:2404.19737). ``1`` is one latent per trace token -- the
    # original layout, whose latent budget has to grow with trace length and so
    # cannot reach the long traces where arXiv:2607.16972 finds continuous CoT
    # collapsing. ``c > 1`` decouples the two: capacity is ``n_latent * c``.
    trace_compress: int = 1
    # Space supervision (arXiv:2606.20075's second dimension): a supervised-
    # contrastive term over the latent manifold, so the latents are *arranged* by
    # the value they stand for and not merely decodable to it. Off by default --
    # it is an opt-in arm to be measured, not a claim. ``space_dim`` is the
    # projection width the contrast is computed in.
    space_coef: float = 0.0
    space_dim: int = 32
    space_tau: float = 0.1
    # The latent ALU (:mod:`lamb.alu`). ``alu_coef > 0`` re-reads the latent block
    # as one *value* per slot, coded in a residue number system, instead of one
    # *digit token* per slot. The answer is then composed by exact arithmetic
    # rather than read out by the network, so magnitude extrapolation stops being
    # something the network has to learn. ``alu_consistency_coef`` weights the
    # label-free agreement between the model's stated answer and the composed one --
    # the term that is defined on problems with no known answer.
    alu_coef: float = 0.0
    alu_consistency_coef: float = 0.1
    # Short multiplicative order of 10 (see lamb.algebra.DEFAULT_MODULI): the
    # digit-coefficient pattern repeats every ord_p(10) positions, so a long
    # order makes a modulus useless at the operand widths training reaches.
    alu_moduli: tuple = (2, 5, 9, 11, 7, 13, 37)
    trace_coef: float = 0.5  # weight on per-position supervision (0 => answer-only ablation)

    # SWITCH-style boundary tokens (arXiv:2606.13106) wrapping the latent segment.
    # Entering latent reasoning becomes a *predicted* token, so the segment has a
    # well-defined probability (what makes on-policy RL over latent recurrence
    # possible at all -- arXiv:2512.11816 showed it otherwise is not) and a fixed
    # position for probes to attach to.
    # Default OFF. The entry boundary was added on the premise that it unblocks
    # on-policy RL over latent recurrence; `lamb/latent_rl.py` tested that and it
    # does not (GRPO moves the model ~0 either way). Its apparent +5.9 accuracy
    # gain was measured on a contaminated eval set; on a clean held-out partition
    # the sign flips to -1.1, i.e. it is indistinguishable from zero against a
    # run-to-run spread of several points. It costs a sequence position and two
    # vocabulary ids, so it is not on by default -- but it is kept, because it is
    # the only well-defined attachment point for probes and for any future RL.
    use_boundaries: bool = False
    switch_coef: float = 0.1  # weight on predicting the entry boundary
    # The *exit* marker is off by default, and measured: interposing a static EOT
    # embedding between the refined latents and the answer readout makes the first
    # answer token be predicted from a generic marker instead of the latent state,
    # which halved accuracy (0.176 vs 0.371 control at 500 steps). Entry-only both
    # fixes that and beats the control (0.500). This also matches arXiv:2606.13106's
    # own finding that the computation concentrates at the *entry* transition; with
    # a fixed latent budget the exit is deterministic and needs no marker.
    use_exit_boundary: bool = False

    eval_every: int = 250
    eval_tasks: int = 256
    log_every: int = 50
    ckpt_dir: str = "runs"

    def max_answer_len(self) -> int:
        return min(24, 2 + self.digits * (2 ** min(self.depth, 3)))


@dataclass
class CommConfig:
    """Stage B: latent inter-agent communication (message = a Coconut thought).

    Two LAMb agents solve a split problem neither can solve alone. The **speaker**
    sees operand ``X``; the **listener** sees the operator and operand ``Y`` and
    must output ``X op Y``. The listener cannot recover the answer without ``X``,
    so the speaker must communicate it -- but not in language. The speaker rolls
    ``n_msg`` Coconut thought vectors (Stage A's mechanism) and those *continuous*
    vectors, passed through a differentiable ``Channel``, are the message; the
    listener consumes them as latent input positions (receiving = thinking another
    agent's thought) and reasons on. The whole thing trains end to end: the
    listener's answer-NLL back-propagates through the message into the speaker
    (differentiable inter-agent learning, DIAL, arXiv:1605.06676). No token is
    ever exchanged -- the channel is pure latent, a blackbox.

    Diagnostics follow the emergent-communication literature: a **zeroed-message**
    ablation (does accuracy collapse to the no-information prior when the channel
    is blanked?) isolates the channel, and a **bandwidth sweep** over the channel
    bottleneck traces the capacity-accuracy tradeoff.
    """

    steps: int = 1500
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.01
    warmup: int = 60
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    amp: Optional[bool] = None  # mixed precision; None => auto (on for CUDA, off for CPU)

    # Split task: speaker sees X (a_digits), listener sees op and Y (b_digits) and
    # must output X op Y. Neither half determines the answer alone.
    a_digits: int = 1
    b_digits: int = 1
    ops: Tuple[str, ...] = ("+", "-")

    # Communication channel.
    n_msg: int = 2               # number of latent message vectors the speaker emits
    n_listen_thoughts: int = 1   # listener's own Coconut thoughts after receiving
    bottleneck: int = 0          # channel width; 0 => full (= d_model). The bandwidth knob.
    msg_dropout: float = 0.0     # optional latent-phase dropout on the speaker
    channel_noise: float = 0.0   # DRU-style Gaussian noise on the bottleneck code (train only);
                                 # >0 gives the channel finite capacity, so bandwidth trades off
                                 # against accuracy (DIAL, arXiv:1605.06676). 0 => noiseless.

    # Agents (kept small for a CPU pair).
    d_model: int = 96
    n_heads: int = 4
    recurrent_steps: int = 4

    # Partner randomization (population; comm_pop). P speakers + Q listeners paired
    # at random, with the diagonal (i, i) pairings held out from training so they
    # can be evaluated zero-shot -- the test of whether the code became canonical.
    pop_speakers: int = 3
    pop_listeners: int = 3

    # Evaluation.
    eval_every: int = 200
    eval_tasks: int = 512
    eval_bottlenecks: Tuple[int, ...] = (1, 2, 4, 8, 0)  # bandwidth sweep (0 => full)
    log_every: int = 100
    ckpt_dir: str = "runs"

    def max_answer_len(self) -> int:
        return min(24, max(self.a_digits, self.b_digits) + 3)


@dataclass
class POETConfig:
    """Red Queen Step 3: a POET-style population of (environment, agent) pairs.

    Each environment is a grammar descriptor paired with its own specialist LAMb
    solver. Agents are optimized on their environment; better agents are
    transferred onto other environments; competent environments reproduce harder,
    novel children (minimal-criterion + novelty gated), seeded by transfer. The
    population is the capacity-scaling mechanism -- specialists + transfer reach a
    frontier a single tiny solver cannot.
    """

    iters: int = 300
    seed: int = 0
    device: str = "cpu"
    amp: Optional[bool] = None  # mixed precision; None => auto (on for CUDA, off for CPU)

    # Agents (per-environment specialists). Kept small for a CPU population.
    agent_d_model: int = 96
    agent_recurrent_steps: int = 3
    adapter_rank: int = 16       # shared-backbone POET: per-environment adapter width
    adapter_type: str = "hidden"  # "hidden" (final-layer bottleneck) | "lora" | "dora" (deeper)
    lora_rank: int = 8
    lora_alpha: float = 8.0      # alpha == rank -> unit scaling (gentle for from-scratch)
    lora_targets: Tuple[str, ...] = ("qkv", "proj")  # which core Linears LoRA adapts

    # Inner optimisation (expert iteration on the environment's exact answers).
    opt_steps: int = 4
    batch_size: int = 64
    lr: float = 2e-3
    eval_tasks: int = 24         # tasks used to score an agent on an environment

    # Population dynamics.
    init_members: int = 2        # seed environments (depth 1, widths 1..init_members)
    pop_capacity: int = 6
    transfer_every: int = 8
    transfer_margin: float = 0.10
    reproduce_every: int = 12
    reproduce_threshold: float = 0.6   # a parent this competent may spawn children
    mc_high: float = 0.9         # a child not already (near-)solved by its seed agent
    mastery_threshold: float = 0.9

    # Behavioural-novelty admission (novelty search): admit a child only if it is
    # far, in behaviour space, from environments already admitted.
    behavioural_novelty: bool = True
    novelty_threshold: float = 0.15
    novelty_k: int = 3
    bc_tasks: int = 16

    # Grammar caps (the space is unbounded up to these for the demo).
    max_depth: int = 4
    max_digits: int = 4

    log_every: int = 20
