"""Configuration dataclasses for LAMb.

Two configs: :class:`ModelConfig` (architecture) and :class:`TrainConfig`
(self-play optimisation). Defaults are tuned for a *tiny CPU-first* model that
trains end-to-end in a couple of minutes while exercising every component:
number-native embeddings, the depth-recurrent latent core, the test-time neural
memory, and the self-play loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


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
    proposer_temp: float = 6.0   # softmax sharpness over per-cell learnability
    proposer_warmup: int = 60    # uniform-coverage steps before the proposer engages
    proposer_eps: float = 0.35   # exploration floor: fraction of cells drawn uniformly

    # Expert iteration replay buffer (STaR-style): verified solver traces.
    buffer_capacity: int = 20000
    expert_fraction: float = 0.25  # share of each batch drawn from solved-trace replay

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
