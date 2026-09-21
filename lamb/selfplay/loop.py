"""The self-play training loop.

Each step:

1. **Propose.** Sample a batch of difficulty cells from the proposer (uniformly
   during a short warmup, then from its policy: the bandit, or the GRPO-trained
   hypernetwork).
2. **Generate & supervise.** Draw a concrete ``(problem, answer)`` per cell from
   the exact verifier and take one teacher-forcing SGD step on the solver, mixed
   with a fraction of replayed *solved* traces (expert iteration / STaR). When
   ``solver_algo == "grpo"``, a GRPO/RLVR term is added after a warm-up.
3. **Roll out.** Greedy-decode the solver on the fresh problems and check each
   with the exact verifier -- this is the only reward signal.
4. **Co-evolve.** Update per-cell success (EMA), push solved traces into the
   replay buffer, and update the proposer toward learnability.

The observable signature of self-improvement: held-out accuracy climbs while the
*mastered difficulty frontier* expands over training.
"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import TrainConfig
from ..data import collate
from ..model.lamb import LAMb
from ..tokenizer import ArithmeticTokenizer
from . import grpo as grpo_utils
from .hyperproposer import GRPOHyperProposer
from .league import League
from .proposer import BanditProposer, BaseProposer, learnability
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
    extra: Dict[str, float] = field(default_factory=dict)


def _answer_budget(cfg: TrainConfig) -> int:
    span = 2 * cfg.max_digits + 2 if "*" in cfg.ops else cfg.max_digits + 2
    return span + 1  # room for a sign / EOS


class SelfPlayTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: LAMb,
        tokenizer: ArithmeticTokenizer,
        proposer: Optional[BaseProposer] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.tok = tokenizer
        self.verifier = Verifier()
        self.grid: List[Tuple[str, int, int]] = cfg.difficulty_grid()
        # The proposer is injectable and swappable via cfg.proposer_kind. Both
        # implementations share the BaseProposer interface: the learning-progress
        # bandit (default), or the GRPO-trained hypernetwork anchored to it.
        self.proposer = proposer or self._build_proposer()

        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # GRPO/RLVR keeps a frozen reference policy for the KL term.
        self.ref_model: Optional[LAMb] = None
        if cfg.solver_algo == "grpo":
            self.ref_model = copy.deepcopy(model).eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        # Red Queen: historical-self-play league + per-cell visitation for novelty.
        self.league: Optional[League] = League(cfg.league_capacity) if cfg.red_queen else None
        self.visit_ema = np.zeros(len(self.grid), dtype=np.float64)

        self.buffer: deque = deque(maxlen=cfg.buffer_capacity)
        self.success_ema: Dict[int, float] = {i: 0.0 for i in range(len(self.grid))}
        self._rng = random.Random(cfg.seed)
        self._seed_ctr = 1
        self._answer_len = _answer_budget(cfg)
        self.step = 0

    def _build_proposer(self) -> BaseProposer:
        cfg = self.cfg
        n = len(self.grid)
        if cfg.proposer_kind == "grpo_hyper":
            return GRPOHyperProposer(
                n, hidden=cfg.hyper_hidden, eps=cfg.proposer_eps, bandit_temp=cfg.proposer_temp,
                kl_coef=cfg.hyper_kl_coef, entropy_coef=cfg.hyper_entropy_coef, lr=cfg.hyper_lr,
                rng=random.Random(cfg.seed + 1),
            )
        return BanditProposer(
            n, temperature=cfg.proposer_temp, eps=cfg.proposer_eps,
            rng=random.Random(cfg.seed + 1),
        )

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

    def _state(self) -> np.ndarray:
        """The per-cell solver-competence state (smoothed success), the proposer input."""
        return np.array([self.success_ema[i] for i in range(len(self.grid))], dtype=np.float64)

    def _novelty(self) -> np.ndarray:
        """Count-based novelty from EMA visitation (higher = less recently proposed)."""
        nov = 1.0 / np.sqrt(1e-3 + self.visit_ema)
        return nov / (nov.max() + 1e-9)  # normalized to (0, 1]

    def _sample_cells(self, n: int, state: np.ndarray) -> List[int]:
        if self.step < self.cfg.proposer_warmup:
            return [self._rng.randrange(len(self.grid)) for _ in range(n)]
        p = self.proposer.probs(state)
        # Diversity maintenance: up-weight rarely-visited cells so the arms race
        # keeps exploring the frontier instead of collapsing onto one region.
        if self.cfg.red_queen and self.cfg.novelty_coef > 0 and len(self.grid) > 1:
            nov = self._novelty() - self._novelty().mean()
            p = p * np.exp(self.cfg.novelty_coef * nov)
            p = p / p.sum()
        return self._rng.choices(range(len(self.grid)), weights=p.tolist(), k=n)

    def _mastered_frontier(self) -> int:
        frontier = 0
        for i, (_, a, b) in enumerate(self.grid):
            if self.success_ema[i] >= self.cfg.mastery_threshold:
                frontier = max(frontier, a + b)
        return frontier

    def _grpo_term(self, problems: List[str]) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        cfg = self.cfg
        subset = problems[: cfg.grpo_problems]
        gloss, gmetrics, solved = grpo_utils.grpo_solver_loss(
            self.model, self.ref_model, self.tok, self.verifier, subset,
            group_size=cfg.grpo_group_size, temperature=cfg.grpo_temperature,
            kl_coef=cfg.grpo_kl_coef, normalize_std=cfg.grpo_normalize_std,
            dynamic_sampling=cfg.grpo_dynamic_sampling, max_answer_len=self._answer_len,
            device=cfg.device,
        )
        for p, pr in solved:
            self.buffer.append((p, pr))
        return gloss, gmetrics

    # -- one optimisation step -------------------------------------------
    def train_step(self) -> StepStats:
        cfg = self.cfg
        state = self._state()  # competence state actions are sampled from
        cells = self._sample_cells(cfg.batch_size, state)
        problems: List[str] = []
        answers: List[str] = []
        for idx in cells:
            op, a, b = self.grid[idx]
            expr, ans = self.verifier.sample(op, a, b, self._next_seed())
            problems.append(expr)
            answers.append(ans)

        # Track per-cell visitation (EMA) for the novelty / diversity term.
        counts = np.bincount(cells, minlength=len(self.grid)).astype(np.float64) / max(1, len(cells))
        self.visit_ema = cfg.novelty_decay * self.visit_ema + (1.0 - cfg.novelty_decay) * counts

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

        extra: Dict[str, float] = {}
        # GRPO/RLVR term on the solver (added to expert CE after a warm start).
        if cfg.solver_algo == "grpo" and self.step >= cfg.grpo_warmup:
            gloss, gmetrics = self._grpo_term(problems)
            extra.update(gmetrics)
            if gloss is not None:
                loss = loss + cfg.grpo_coef * gloss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.opt.step()

        if self.ref_model is not None and (self.step + 1) % cfg.grpo_ref_update_every == 0:
            self.ref_model.load_state_dict(self.model.state_dict())

        # Historical self-play: periodically freeze the solver into the league.
        if self.league is not None:
            self.league.maybe_snapshot(self.model, self.step + 1, cfg.league_snapshot_every)

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

        # Co-evolve the proposer: reward each proposed cell by its (updated)
        # learnability, plus a novelty bonus under Red Queen. The bandit ignores
        # this; the hypernetwork learns from it.
        if self.step >= cfg.proposer_warmup:
            reward_vec = learnability(self._state())
            if cfg.red_queen and cfg.novelty_coef > 0:
                reward_vec = reward_vec + cfg.novelty_coef * self._novelty()
            prop_metrics = self.proposer.update(state, cells, [reward_vec[c] for c in cells])
            extra.update(prop_metrics)

        if self.league is not None:
            extra["league_size"] = float(len(self.league))

        self.step += 1
        post = self._state()
        top_idx = int(self.proposer.probs(post).argmax())
        stats = StepStats(
            step=self.step,
            loss=float(loss.detach()),
            token_acc=metrics["token_acc"],
            batch_success=batch_success,
            mastered_frontier=self._mastered_frontier(),
            buffer_size=len(self.buffer),
            proposer_entropy=self.proposer.entropy(post),
            top_cell="{}{}x{}".format(*self.grid[top_idx]),
            extra=extra,
        )
        return stats

    def red_queen_report(self, n: int = 64) -> Optional[Dict[str, float]]:
        """Relative-fitness snapshot: current solver vs. its league on the current
        frontier (dominance) and on a fixed easy set (forgetting). ``None`` until
        the league has a snapshot."""
        if self.league is None or len(self.league) == 0:
            return None
        state = self._state()
        p = self.proposer.probs(state)
        frontier_cells = self._rng.choices(range(len(self.grid)), weights=p.tolist(), k=n)
        frontier = [self._instance(c) for c in frontier_cells]

        easy = [i for i, (_, a, b) in enumerate(self.grid)
                if a == self.cfg.start_digits and b == self.cfg.start_digits] or list(range(len(self.grid)))
        retention = [self._instance(self._rng.choice(easy)) for _ in range(n)]

        return self.league.relative_fitness(
            self.model, self.tok, frontier, retention, device=self.cfg.device,
            max_answer_len=self._answer_len,
        )

    def _instance(self, cell_idx: int) -> Tuple[str, str]:
        op, a, b = self.grid[cell_idx]
        return self.verifier.sample(op, a, b, self._next_seed())

    def _update_success_ema(self, cells: List[int], correct: List[bool]) -> None:
        per_cell: Dict[int, List[float]] = defaultdict(list)
        for idx, ok in zip(cells, correct):
            per_cell[idx].append(float(ok))
        for idx, vals in per_cell.items():
            batch_rate = sum(vals) / len(vals)
            self.success_ema[idx] = 0.9 * self.success_ema[idx] + 0.1 * batch_rate
