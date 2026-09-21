"""The self-play training loop.

Each step:

1. **Propose.** Sample a batch of difficulty cells from the proposer (uniformly
   during a short warmup, then from its learned policy).
2. **Generate & supervise.** Draw a concrete ``(problem, answer)`` per cell from
   the exact verifier and take one teacher-forcing SGD step on the solver, mixed
   with a fraction of replayed *solved* traces (expert iteration / STaR).
3. **Roll out.** Greedy-decode the solver on the fresh problems and check each
   with the exact verifier -- this is the only reward signal.
4. **Co-evolve.** Update per-cell success (EMA), push solved traces into the
   replay buffer, and update the proposer by REINFORCE toward learnability.

The observable signature of self-improvement: held-out accuracy climbs while the
*mastered difficulty frontier* expands over training.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import TrainConfig
from ..data import collate
from ..model.lamb import LAMb
from ..tokenizer import ArithmeticTokenizer
from .proposer import Proposer
from .verifier import Verifier


@dataclass
class StepStats:
    step: int
    loss: float
    token_acc: float
    batch_success: float
    mastered_frontier: int
    buffer_size: int
    proposer_entropy: float
    top_cell: str


def _answer_budget(cfg: TrainConfig) -> int:
    span = 2 * cfg.max_digits + 2 if "*" in cfg.ops else cfg.max_digits + 2
    return span + 1  # room for a sign / EOS


class SelfPlayTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: LAMb,
        tokenizer: ArithmeticTokenizer,
        proposer: Optional[Proposer] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.tok = tokenizer
        self.verifier = Verifier()
        self.grid: List[Tuple[str, int, int]] = cfg.difficulty_grid()
        # The proposer is injectable: the default is the learning-progress bandit,
        # but any object exposing ``sample(n, learnability)`` / ``probs`` /
        # ``entropy`` can be dropped in -- e.g. a GRPO-trained hypernetwork policy
        # that keeps the bandit as its KL/coverage anchor (see docs/ROADMAP.md).
        self.proposer = proposer or Proposer(
            len(self.grid),
            temperature=cfg.proposer_temp,
            eps=cfg.proposer_eps,
            rng=random.Random(cfg.seed + 1),
        )

        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.buffer: deque = deque(maxlen=cfg.buffer_capacity)
        self.success_ema: Dict[int, float] = {i: 0.0 for i in range(len(self.grid))}
        self._rng = random.Random(cfg.seed)
        self._seed_ctr = 1
        self._answer_len = _answer_budget(cfg)
        self.step = 0

    # -- helpers ----------------------------------------------------------
    def _next_seed(self) -> int:
        self._seed_ctr += 1
        return (self.cfg.seed * 1_000_003 + self._seed_ctr) & 0x7FFFFFFF

    def _lr_at(self, step: int) -> float:
        warmup = self.cfg.warmup
        if step < warmup:
            return self.cfg.lr * (step + 1) / max(1, warmup)
        # Cosine decay to 10% of peak over the remaining steps -- stabilises the
        # late phase where a constant LR oscillates.
        min_lr = 0.1 * self.cfg.lr
        progress = (step - warmup) / max(1, self.cfg.steps - warmup)
        progress = min(1.0, max(0.0, progress))
        return min_lr + 0.5 * (self.cfg.lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def _learnability(self) -> np.ndarray:
        return np.array(
            [4.0 * self.success_ema[i] * (1.0 - self.success_ema[i]) for i in range(len(self.grid))],
            dtype=np.float64,
        )

    def _sample_cells(self, n: int) -> List[int]:
        if self.step < self.cfg.proposer_warmup:
            return [self._rng.randrange(len(self.grid)) for _ in range(n)]
        cells, _ = self.proposer.sample(n, self._learnability())
        return cells

    def _mastered_frontier(self) -> int:
        frontier = 0
        for i, (_, a, b) in enumerate(self.grid):
            if self.success_ema[i] >= self.cfg.mastery_threshold:
                frontier = max(frontier, a + b)
        return frontier

    # -- one optimisation step -------------------------------------------
    def train_step(self) -> StepStats:
        cfg = self.cfg
        cells = self._sample_cells(cfg.batch_size)
        problems: List[str] = []
        answers: List[str] = []
        for idx in cells:
            op, a, b = self.grid[idx]
            expr, ans = self.verifier.sample(op, a, b, self._next_seed())
            problems.append(expr)
            answers.append(ans)

        examples = [self.tok.encode(p, a) for p, a in zip(problems, answers)]
        # Expert-iteration replay: mix in previously solved traces.
        n_replay = int(len(examples) * cfg.expert_fraction)
        if n_replay > 0 and len(self.buffer) > 0:
            replay = self._rng.choices(list(self.buffer), k=n_replay)
            examples += [self.tok.encode(p, a) for p, a in replay]

        batch = collate(examples, self.tok.PAD, device=cfg.device)

        for group in self.opt.param_groups:
            group["lr"] = self._lr_at(self.step)

        self.model.train()
        n_steps = None
        if cfg.sample_train_depth:
            n_steps = self._rng.randint(cfg.train_min_steps, cfg.train_max_steps)
        loss, metrics = self.model.compute_loss(batch, n_steps=n_steps)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.opt.step()

        # Roll out on the freshly proposed problems for the reward signal.
        preds = self.model.solve(
            problems, self.tok, max_answer_len=self._answer_len, device=cfg.device
        )
        correct = [self.verifier.check(p, pr) for p, pr in zip(problems, preds)]
        for p, a, ok in zip(problems, answers, correct):
            if ok:
                self.buffer.append((p, a))

        batch_success = sum(correct) / max(1, len(correct))
        self._update_success_ema(cells, correct)

        self.step += 1
        learn = self._learnability()
        top_idx = int(self.proposer.probs(learn).argmax())
        stats = StepStats(
            step=self.step,
            loss=float(loss.detach()),
            token_acc=metrics["token_acc"],
            batch_success=batch_success,
            mastered_frontier=self._mastered_frontier(),
            buffer_size=len(self.buffer),
            proposer_entropy=self.proposer.entropy(learn),
            top_cell="{}{}x{}".format(*self.grid[top_idx]),
        )
        return stats

    def _update_success_ema(self, cells: List[int], correct: List[bool]) -> None:
        per_cell: Dict[int, List[float]] = defaultdict(list)
        for idx, ok in zip(cells, correct):
            per_cell[idx].append(float(ok))
        for idx, vals in per_cell.items():
            batch_rate = sum(vals) / len(vals)
            self.success_ema[idx] = 0.9 * self.success_ema[idx] + 0.1 * batch_rate
