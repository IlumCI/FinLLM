"""POET-style population trainer -- Red Queen Step 3.

A population of ``(environment, agent)`` pairs, where an environment is a grammar
descriptor and an agent is its own specialist LAMb solver. Each iteration:

1. **Optimize.** Every agent takes a few expert-iteration steps on the exact
   answers of its own environment.
2. **Transfer.** Periodically, if some other agent outperforms an environment's
   incumbent (by a margin), it replaces the incumbent -- innovations from one
   environment aid another (POET's key mechanism against stagnation).
3. **Reproduce.** Competent environments spawn harder, novel children (single-step
   grammar neighbours), each seeded by its parent's agent (transfer at birth),
   admitted only if not already solved by that seed (a minimal criterion).
4. **Graduate.** When over capacity, the easiest environments are retired.

This is the capacity-scaling answer to Step 2: a population of specialists plus
transfer reaches a frontier a single tiny solver cannot. Following POET / Enhanced
POET (arXiv:1901.01753, 2003.08536).
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch

from . import __version__
from ._native import backend, verify
from .config import ModelConfig, POETConfig
from .data import collate
from .model.lamb import LAMb, build_model
from .selfplay.grammar import Descriptor, TaskGrammar
from .tokenizer import ArithmeticTokenizer


def complexity(d: Descriptor) -> int:
    return 2 * d.depth + d.digits + d.ops_key


def neighbours(d: Descriptor, max_depth: int, max_digits: int, n_ops: int) -> List[Descriptor]:
    out = []
    if d.depth + 1 <= max_depth:
        out.append(Descriptor(d.depth + 1, d.digits, d.ops_key))
    if d.digits + 1 <= max_digits:
        out.append(Descriptor(d.depth, d.digits + 1, d.ops_key))
    if d.ops_key + 1 < n_ops:
        out.append(Descriptor(d.depth, d.digits, d.ops_key + 1))
    return out


@dataclass
class Member:
    env: Descriptor
    agent: LAMb
    opt: torch.optim.Optimizer
    success: float = 0.0
    born: int = 0


class POETTrainer:
    def __init__(self, cfg: POETConfig, tokenizer: ArithmeticTokenizer):
        self.cfg = cfg
        self.tok = tokenizer
        self.grammar = TaskGrammar()
        self.n_ops = len(self.grammar.ops_sets)
        self.population: List[Member] = []
        self._rng = random.Random(cfg.seed)
        self._seed_ctr = 1
        self.iter = 0
        self.transfers = 0
        self.best_conquered = 0
        self._answer_len = self.grammar.max_answer_len(cfg.max_depth, cfg.max_digits)
        for g in range(1, cfg.init_members + 1):
            self._add(Descriptor(1, g, 0), self._new_agent())

    # -- helpers ----------------------------------------------------------
    def _next_seed(self) -> int:
        self._seed_ctr += 1
        return (self.cfg.seed * 1_000_003 + self._seed_ctr) & 0x7FFFFFFF

    def _new_agent(self) -> LAMb:
        mcfg = ModelConfig(d_model=self.cfg.agent_d_model, recurrent_steps=self.cfg.agent_recurrent_steps)
        return build_model(mcfg, self.tok).to(self.cfg.device)

    def _add(self, env: Descriptor, agent: LAMb) -> Member:
        opt = torch.optim.AdamW(agent.parameters(), lr=self.cfg.lr, weight_decay=0.01)
        m = Member(env=env, agent=agent, opt=opt, born=self.iter)
        self.population.append(m)
        return m

    def _env_set(self):
        return {m.env for m in self.population}

    def _batch(self, env: Descriptor, n: int):
        examples = [self.tok.encode(*self.grammar.sample(env, self._next_seed())) for _ in range(n)]
        return collate(examples, self.tok.PAD, device=self.cfg.device)

    @torch.no_grad()
    def _score(self, agent: LAMb, env: Descriptor, n: Optional[int] = None) -> float:
        n = n or self.cfg.eval_tasks
        tasks = [self.grammar.sample(env, self._next_seed()) for _ in range(n)]
        preds = agent.solve([p for p, _ in tasks], self.tok, max_answer_len=self._answer_len, device=self.cfg.device)
        ok = sum(1 for (_, a), pr in zip(tasks, preds) if pr is not None and pr == a)
        return ok / max(1, n)

    # -- POET operators ---------------------------------------------------
    def _optimize(self) -> None:
        for m in self.population:
            m.agent.train()
            for _ in range(self.cfg.opt_steps):
                loss, _ = m.agent.compute_loss(self._batch(m.env, self.cfg.batch_size))
                m.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.agent.parameters(), 1.0)
                m.opt.step()
            m.success = 0.6 * m.success + 0.4 * self._score(m.agent, m.env)

    def _transfer(self) -> None:
        # Try replacing each incumbent with a better-performing agent.
        snapshots = [(i, m.agent.state_dict()) for i, m in enumerate(self.population)]
        for i, m in enumerate(self.population):
            best_j, best_s = -1, m.success + self.cfg.transfer_margin
            for j, sd in snapshots:
                if j == i:
                    continue
                s = self._score(self.population[j].agent, m.env)
                if s > best_s:
                    best_s, best_j = s, j
            if best_j >= 0:
                m.agent.load_state_dict(self.population[best_j].agent.state_dict())
                m.opt = torch.optim.AdamW(m.agent.parameters(), lr=self.cfg.lr, weight_decay=0.01)
                m.success = best_s
                self.transfers += 1

    def _reproduce(self) -> None:
        existing = self._env_set()
        births: List[Member] = []
        for m in list(self.population):
            if m.success < self.cfg.reproduce_threshold:
                continue
            for child in neighbours(m.env, self.cfg.max_depth, self.cfg.max_digits, self.n_ops):
                if child in existing or child in {b.env for b in births}:
                    continue
                seed_agent = self._new_agent()
                seed_agent.load_state_dict(m.agent.state_dict())
                # Minimal criterion: not already (near-)solved by the seed agent.
                if self._score(seed_agent, child) <= self.cfg.mc_high:
                    opt = torch.optim.AdamW(seed_agent.parameters(), lr=self.cfg.lr, weight_decay=0.01)
                    births.append(Member(env=child, agent=seed_agent, opt=opt, born=self.iter))
        self.population.extend(births)

    def _graduate(self) -> None:
        while len(self.population) > self.cfg.pop_capacity:
            # Retire the easiest environment (graduate it out of the active set).
            idx = min(range(len(self.population)), key=lambda i: complexity(self.population[i].env))
            self.population.pop(idx)

    # -- driver -----------------------------------------------------------
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
        return max((complexity(m.env) for m in self.population), default=0)

    def conquered_frontier(self) -> int:
        return max((complexity(m.env) for m in self.population
                    if m.success >= self.cfg.mastery_threshold), default=0)

    def best_member(self) -> Optional[Member]:
        if not self.population:
            return None
        return max(self.population, key=lambda m: (complexity(m.env), m.success))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="POET population trainer for LAMb (Red Queen Step 3).")
    p.add_argument("--iters", type=int, default=POETConfig.iters)
    p.add_argument("--pop-capacity", type=int, default=POETConfig.pop_capacity)
    p.add_argument("--agent-d-model", type=int, default=POETConfig.agent_d_model)
    p.add_argument("--agent-recurrent-steps", type=int, default=POETConfig.agent_recurrent_steps)
    p.add_argument("--max-depth", type=int, default=POETConfig.max_depth)
    p.add_argument("--max-digits", type=int, default=POETConfig.max_digits)
    p.add_argument("--opt-steps", type=int, default=POETConfig.opt_steps,
                   help="expert-iteration steps per member per POET iteration")
    p.add_argument("--eval-tasks", type=int, default=POETConfig.eval_tasks)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=POETConfig.log_every)
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    cfg = POETConfig(
        iters=args.iters, pop_capacity=args.pop_capacity, agent_d_model=args.agent_d_model,
        agent_recurrent_steps=args.agent_recurrent_steps, max_depth=args.max_depth,
        max_digits=args.max_digits, opt_steps=args.opt_steps, eval_tasks=args.eval_tasks,
        seed=args.seed, device=args.device, log_every=args.log_every,
    )
    tok = ArithmeticTokenizer()
    trainer = POETTrainer(cfg, tok)

    print(f"LAMb POET v{__version__} | native kernels: {backend()}")
    print(f"agents: d_model={cfg.agent_d_model} recurrent_steps={cfg.agent_recurrent_steps} "
          f"| pop_capacity={cfg.pop_capacity} | caps depth<={cfg.max_depth} digits<={cfg.max_digits}")
    print("-" * 88)
    start = time.time()
    for _ in range(cfg.iters):
        trainer.iterate()
        if trainer.iter % cfg.log_every == 0:
            best = trainer.best_member()
            print(f"iter {trainer.iter:4d} | pop {len(trainer.population):2d} "
                  f"| attempted {trainer.attempted_frontier():2d} | conquered {trainer.conquered_frontier():2d} "
                  f"| transfers {trainer.transfers:3d} "
                  f"| best {best.env.label()}@{best.success:.2f}")

    dur = time.time() - start
    print("-" * 88)
    print(f"final population ({len(trainer.population)} members) after {dur:.1f}s:")
    for m in sorted(trainer.population, key=lambda m: complexity(m.env)):
        print(f"    {m.env.label():10s} complexity {complexity(m.env):2d} | success {m.success:.2f} "
              f"| born@{m.born}")
    print(f"attempted frontier {trainer.attempted_frontier()} | peak conquered frontier "
          f"{trainer.best_conquered} | total transfers {trainer.transfers}")


if __name__ == "__main__":
    main()
