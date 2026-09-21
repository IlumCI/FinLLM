"""Shared-backbone POET -- one backbone, tiny per-environment adapters.

The plain POET population (`lamb/poet.py`) keeps a *separate full solver* per
environment, so N environments cost N models. This variant keeps **one shared
backbone** and gives each environment a **small low-rank adapter** that
specialises the backbone's hidden state before the LM head. Two consequences:

* **Parameter efficiency.** A population of N environments costs one backbone plus
  N tiny adapters (a few thousand parameters each) instead of N full models.
* **Implicit transfer.** The shared backbone is updated by *every* environment, so
  general skill accumulates centrally and a newly reproduced environment inherits
  a competent backbone for free; explicit transfer then just copies the small
  adapter.

Same POET operators as `poet.py` -- optimize / transfer / reproduce / graduate --
but "the agent" of an environment is its adapter, and the backbone is shared.
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import __version__
from ._native import backend
from .config import ModelConfig, POETConfig
from .data import collate
from .model.lamb import LAMb, build_model
from .poet import Descriptor, complexity, neighbours
from .selfplay.grammar import TaskGrammar
from .selfplay.novelty import NoveltyArchive, behaviour_characterization
from .tokenizer import ArithmeticTokenizer


class Adapter(nn.Module):
    """Low-rank residual bottleneck; zero-initialised so it starts as identity."""

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(F.gelu(self.down(h)))


@dataclass
class SharedMember:
    env: Descriptor
    adapter: Adapter
    opt: torch.optim.Optimizer
    success: float = 0.0
    born: int = 0


class SharedBackbonePOETTrainer:
    def __init__(self, cfg: POETConfig, tokenizer: ArithmeticTokenizer):
        self.cfg = cfg
        self.tok = tokenizer
        self.grammar = TaskGrammar()
        self.n_ops = len(self.grammar.ops_sets)
        self.backbone: LAMb = build_model(
            ModelConfig(d_model=cfg.agent_d_model, recurrent_steps=cfg.agent_recurrent_steps),
            tokenizer,
        ).to(cfg.device)
        self.backbone_opt = torch.optim.AdamW(self.backbone.parameters(), lr=cfg.lr, weight_decay=0.01)
        self.members: List[SharedMember] = []
        self._rng = random.Random(cfg.seed)
        self._seed_ctr = 1
        self.iter = 0
        self.transfers = 0
        self.novelty_rejects = 0
        self.best_conquered = 0
        self._answer_len = self.grammar.max_answer_len(cfg.max_depth, cfg.max_digits)
        self.archive = NoveltyArchive(k=cfg.novelty_k, threshold=cfg.novelty_threshold)
        for g in range(1, cfg.init_members + 1):
            m = self._add(Descriptor(1, g, 0))
            if cfg.behavioural_novelty:
                self.archive.add(self._bc(m.adapter, m.env))

    def _next_seed(self) -> int:
        self._seed_ctr += 1
        return (self.cfg.seed * 1_000_003 + self._seed_ctr) & 0x7FFFFFFF

    def _new_adapter(self) -> Adapter:
        return Adapter(self.cfg.agent_d_model, self.cfg.adapter_rank).to(self.cfg.device)

    def _add(self, env: Descriptor, adapter: Optional[Adapter] = None) -> SharedMember:
        adapter = adapter or self._new_adapter()
        opt = torch.optim.AdamW(adapter.parameters(), lr=self.cfg.lr * 3, weight_decay=0.0)
        m = SharedMember(env=env, adapter=adapter, opt=opt, born=self.iter)
        self.members.append(m)
        return m

    def _batch(self, env: Descriptor, n: int):
        examples = [self.tok.encode(*self.grammar.sample(env, self._next_seed())) for _ in range(n)]
        return collate(examples, self.tok.PAD, device=self.cfg.device)

    @torch.no_grad()
    def _score(self, adapter: Adapter, env: Descriptor, n: Optional[int] = None) -> float:
        n = n or self.cfg.eval_tasks
        tasks = [self.grammar.sample(env, self._next_seed()) for _ in range(n)]
        preds = self.backbone.solve([p for p, _ in tasks], self.tok,
                                    max_answer_len=self._answer_len, device=self.cfg.device,
                                    hidden_adapter=adapter)
        ok = sum(1 for (_, a), pr in zip(tasks, preds) if pr is not None and pr == a)
        return ok / max(1, n)

    @torch.no_grad()
    def _bc(self, adapter: Adapter, env: Descriptor):
        tasks = [self.grammar.sample(env, self._next_seed()) for _ in range(self.cfg.bc_tasks)]
        solve = lambda probs, t: self.backbone.solve(  # noqa: E731
            probs, self.tok, max_answer_len=self._answer_len, n_steps=t,
            device=self.cfg.device, hidden_adapter=adapter)
        return behaviour_characterization(solve, tasks)

    def _optimize(self) -> None:
        self.backbone.train()
        for m in self.members:
            m.adapter.train()
            for _ in range(self.cfg.opt_steps):
                loss, _ = self.backbone.compute_loss(self._batch(m.env, self.cfg.batch_size),
                                                     hidden_adapter=m.adapter)
                self.backbone_opt.zero_grad(set_to_none=True)
                m.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
                self.backbone_opt.step()   # backbone updated by every environment
                m.opt.step()               # adapter specialises its environment
            m.success = 0.6 * m.success + 0.4 * self._score(m.adapter, m.env)

    def _transfer(self) -> None:
        adapters = [copy.deepcopy(m.adapter) for m in self.members]
        for i, m in enumerate(self.members):
            best_j, best_s = -1, m.success + self.cfg.transfer_margin
            for j, adj in enumerate(adapters):
                if j == i:
                    continue
                s = self._score(adj, m.env)
                if s > best_s:
                    best_s, best_j = s, j
            if best_j >= 0:
                m.adapter.load_state_dict(adapters[best_j].state_dict())
                m.opt = torch.optim.AdamW(m.adapter.parameters(), lr=self.cfg.lr * 3, weight_decay=0.0)
                m.success = best_s
                self.transfers += 1

    def _reproduce(self) -> None:
        existing = {m.env for m in self.members}
        births: List[SharedMember] = []
        for m in list(self.members):
            if m.success < self.cfg.reproduce_threshold:
                continue
            for child in neighbours(m.env, self.cfg.max_depth, self.cfg.max_digits, self.n_ops):
                if child in existing or child in {b.env for b in births}:
                    continue
                seed = copy.deepcopy(m.adapter)  # inherit the parent's specialisation
                if self._score(seed, child) > self.cfg.mc_high:
                    continue
                if self.cfg.behavioural_novelty:
                    bc = self._bc(seed, child)
                    if not self.archive.is_novel(bc):
                        self.novelty_rejects += 1
                        continue
                    self.archive.add(bc)
                opt = torch.optim.AdamW(seed.parameters(), lr=self.cfg.lr * 3, weight_decay=0.0)
                births.append(SharedMember(env=child, adapter=seed, opt=opt, born=self.iter))
                existing = existing | {child}
        self.members.extend(births)

    def _graduate(self) -> None:
        while len(self.members) > self.cfg.pop_capacity:
            idx = min(range(len(self.members)), key=lambda i: complexity(self.members[i].env))
            self.members.pop(idx)

    def iterate(self) -> None:
        self._optimize()
        if (self.iter + 1) % self.cfg.transfer_every == 0:
            self._transfer()
        if (self.iter + 1) % self.cfg.reproduce_every == 0:
            self._reproduce()
            self._graduate()
        self.iter += 1
        self.best_conquered = max(self.best_conquered, self.conquered_frontier())

    def attempted_frontier(self) -> int:
        return max((complexity(m.env) for m in self.members), default=0)

    def conquered_frontier(self) -> int:
        return max((complexity(m.env) for m in self.members
                    if m.success >= self.cfg.mastery_threshold), default=0)

    def backbone_params(self) -> int:
        return self.backbone.num_params()

    def adapter_params(self) -> int:
        return sum(p.numel() for p in Adapter(self.cfg.agent_d_model, self.cfg.adapter_rank).parameters())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shared-backbone POET for LAMb (Red Queen Step 3+).")
    p.add_argument("--iters", type=int, default=POETConfig.iters)
    p.add_argument("--pop-capacity", type=int, default=POETConfig.pop_capacity)
    p.add_argument("--agent-d-model", type=int, default=POETConfig.agent_d_model)
    p.add_argument("--agent-recurrent-steps", type=int, default=POETConfig.agent_recurrent_steps)
    p.add_argument("--adapter-rank", type=int, default=POETConfig.adapter_rank)
    p.add_argument("--opt-steps", type=int, default=POETConfig.opt_steps)
    p.add_argument("--eval-tasks", type=int, default=POETConfig.eval_tasks)
    p.add_argument("--max-depth", type=int, default=POETConfig.max_depth)
    p.add_argument("--max-digits", type=int, default=POETConfig.max_digits)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=POETConfig.log_every)
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    cfg = POETConfig(
        iters=args.iters, pop_capacity=args.pop_capacity, agent_d_model=args.agent_d_model,
        agent_recurrent_steps=args.agent_recurrent_steps, adapter_rank=args.adapter_rank,
        opt_steps=args.opt_steps, eval_tasks=args.eval_tasks, max_depth=args.max_depth,
        max_digits=args.max_digits, seed=args.seed, device=args.device, log_every=args.log_every,
    )
    tok = ArithmeticTokenizer()
    trainer = SharedBackbonePOETTrainer(cfg, tok)

    bb, ad = trainer.backbone_params(), trainer.adapter_params()
    print(f"LAMb shared-backbone POET v{__version__} | native kernels: {backend()}")
    print(f"backbone {bb:,} params + {ad:,}/adapter | pop_capacity {cfg.pop_capacity}")
    print(f"vs plain POET at capacity {cfg.pop_capacity}: {bb + ad * cfg.pop_capacity:,} "
          f"here vs {bb * cfg.pop_capacity:,} full models "
          f"({100 * (bb + ad * cfg.pop_capacity) / (bb * cfg.pop_capacity):.0f}% of the parameters)")
    print("-" * 88)
    start = time.time()
    for _ in range(cfg.iters):
        trainer.iterate()
        if trainer.iter % cfg.log_every == 0:
            print(f"iter {trainer.iter:4d} | pop {len(trainer.members):2d} "
                  f"| attempted {trainer.attempted_frontier():2d} | conquered {trainer.conquered_frontier():2d} "
                  f"| transfers {trainer.transfers:3d}")
    dur = time.time() - start
    print("-" * 88)
    print(f"final population ({len(trainer.members)} adapters) after {dur:.1f}s:")
    for m in sorted(trainer.members, key=lambda m: complexity(m.env)):
        print(f"    {m.env.label():10s} complexity {complexity(m.env):2d} | success {m.success:.2f}")
    print(f"attempted frontier {trainer.attempted_frontier()} | peak conquered frontier "
          f"{trainer.best_conquered} | total transfers {trainer.transfers}")


if __name__ == "__main__":
    main()
