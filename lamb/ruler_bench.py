"""RULER / BABILong-style long-context suite over LAMb's symbol space.

BABILong and RULER are *natural-language* benchmarks (bAbI reasoning embedded in
book text; synthetic retrieval/tracking probes in English). LAMb has no language
vocabulary, so the real datasets need the language-bridge fork. What is faithful
and runnable now is RULER's *design* -- a synthetic, model-agnostic battery of
long-context task families with configurable length -- reconstructed over LAMb's
own symbols:

* **NIAH** (needle-in-a-haystack): retrieve the value bound to a queried key.
* **NIAH multi-key**: the same, but among many distractor keys (hard retrieval).
* **Variable tracking (VT)**: resolve a chain ``v1 := v2 := ... := literal`` to its
  root -- the multi-hop reasoning bAbI/BABILong is built on.

All three are one generator (assignments + filler + a query) at different
settings. The model exercises **memory and latent multi-hop together**: it writes
every assignment into the test-time memory, then *iteratively dereferences* the
query by reading the built state, feeding each read back as the next lookup -- no
quadratic attention, so retrieval is the memory's job. A single-hop variant is the
ablation for VT.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.memory import NeuralMemory
from .model.transformer import RMSNorm

FILLER, ASSIGN, QUERY = 0, 1, 2


@dataclass
class RulerConfig:
    n_literals: int = 16
    n_vars: int = 64
    n_filler: int = 16
    d_model: int = 64
    d_mem: int = 96
    hops: int = 4


class RulerTask:
    def __init__(self, cfg: RulerConfig):
        self.cfg = cfg

    def generate(self, bs: int, length: int, chain_len: int, n_distractor: int,
                 rng: random.Random, device: str = "cpu", front: bool = True):
        cfg = self.cfg
        kind = torch.zeros(bs, length, dtype=torch.long)
        lhs = torch.zeros(bs, length, dtype=torch.long)
        rhs = torch.zeros(bs, length, dtype=torch.long)
        filler_ids = torch.randint(0, cfg.n_filler, (bs, length))
        target = torch.zeros(bs, dtype=torch.long)
        variables = list(range(cfg.n_literals, cfg.n_literals + cfg.n_vars))

        for b in range(bs):
            chain = rng.sample(variables, chain_len)
            lit = rng.randrange(cfg.n_literals)
            assigns: List[Tuple[int, int]] = []
            for i in range(chain_len - 1):
                assigns.append((chain[i], chain[i + 1]))
            assigns.append((chain[-1], lit))            # last var -> literal
            query_var, target[b] = chain[0], lit

            free = [v for v in variables if v not in chain]
            rng.shuffle(free)
            for dv in free[: min(n_distractor, len(free))]:
                assigns.append((dv, rng.randrange(cfg.n_literals)))  # distractors bind literals

            n_assign = min(len(assigns), length - 1)
            assigns = assigns[:n_assign]
            if front:
                slots = [0] + rng.sample(range(1, length - 1), n_assign - 1)
            else:
                slots = rng.sample(range(0, length - 1), n_assign)
            for (a, val), pos in zip(assigns, slots):
                kind[b, pos] = ASSIGN
                lhs[b, pos] = a
                rhs[b, pos] = val
            kind[b, length - 1] = QUERY
            lhs[b, length - 1] = query_var

        return (
            {"kind": kind.to(device), "lhs": lhs.to(device), "rhs": rhs.to(device),
             "filler_ids": filler_ids.to(device)},
            target.to(device),
        )


class HopMemoryModel(nn.Module):
    def __init__(self, cfg: RulerConfig, hops: int = None):
        super().__init__()
        self.cfg = cfg
        self.hops = cfg.hops if hops is None else hops
        n_sym = cfg.n_literals + cfg.n_vars
        # A shared symbol embedding with learned key/value projections: because
        # key(s) and value(s) are both linear in one sym(s), dereferencing (map a
        # retrieved value back to the key that looks up the next hop) is a single
        # linear map that works for every symbol -- multi-hop is learnable.
        self.sym = nn.Embedding(n_sym, cfg.d_model)
        self.key_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.val_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.filler_emb = nn.Embedding(cfg.n_filler, cfg.d_model)
        self.assign_marker = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.query_marker = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.memory = NeuralMemory(cfg.d_model, cfg.d_mem)
        self.deref = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm = RMSNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, cfg.n_literals)

    def _embed(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        kind = batch["kind"]
        key = self.key_proj(self.sym(batch["lhs"]))
        val = self.val_proj(self.sym(batch["rhs"]))
        assign_e = key + val + self.assign_marker
        query_e = key + self.query_marker
        filler_e = self.filler_emb(batch["filler_ids"])
        is_a = (kind == ASSIGN).unsqueeze(-1).float()
        is_q = (kind == QUERY).unsqueeze(-1).float()
        is_f = (kind == FILLER).unsqueeze(-1).float()
        return is_a * assign_e + is_q * query_e + is_f * filler_e

    def forward(self, batch: Dict[str, torch.Tensor], hops: int = None) -> torch.Tensor:
        # Exactly `hops` associative reads, dereferencing between them, reading out
        # from the last read. hops=1 is a plain single retrieval (needle); hops=k
        # resolves a k-step assignment chain (variable tracking).
        hops = self.hops if hops is None else hops
        emb = self._embed(batch)
        mem = self.memory.build_state(emb)
        r = self.memory.read_state(mem, emb[:, -1])   # read from the query token
        for _ in range(max(0, hops - 1)):
            r = self.memory.read_state(mem, self.deref(r))
        return self.readout(self.norm(r))


# -- training / evaluation -------------------------------------------------
def train(model: HopMemoryModel, task: RulerTask, steps: int, length: int, max_chain: int,
          n_distractor: int, bs: int = 64, lr: float = 2e-3, seed: int = 0, device: str = "cpu") -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    model.train()
    for _ in range(steps):
        cur_len = rng.randint(max(6, length // 3), length)
        chain = rng.randint(1, max_chain)
        batch, target = task.generate(bs, cur_len, chain, n_distractor, rng, device, front=False)
        # Hops matched to the chain length: k reads supervise k-hop resolution.
        loss = F.cross_entropy(model(batch, hops=chain), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def score(model: HopMemoryModel, task: RulerTask, length: int, chain_len: int, n_distractor: int,
          hops: int = None, bs: int = 128, reps: int = 4, seed: int = 999, device: str = "cpu") -> float:
    model.eval()
    rng = random.Random(seed)
    hops = chain_len if hops is None else hops
    ok = tot = 0
    for _ in range(reps):
        batch, target = task.generate(bs, length, chain_len, n_distractor, rng, device, front=True)
        ok += int((model(batch, hops=hops).argmax(-1) == target).sum())
        tot += target.numel()
    return ok / max(1, tot)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="LAMb RULER/BABILong-style long-context suite.")
    p.add_argument("--steps", type=int, default=1400)
    p.add_argument("--train-length", type=int, default=64)
    p.add_argument("--hops", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)

    cfg = RulerConfig(hops=args.hops)
    task = RulerTask(cfg)
    chance = 1.0 / cfg.n_literals
    lengths = [16, args.train_length, 128, 256]

    model = HopMemoryModel(cfg)
    train(model, task, args.steps, args.train_length, max_chain=args.hops, n_distractor=4,
          seed=args.seed, device=args.device)

    print(f"RULER/BABILong-style suite | chance {chance:.3f} | trained length {args.train_length}, "
          f"hops<={args.hops}, d_mem {cfg.d_mem}")
    print("-" * 72)
    print(f"NIAH (retrieve; 4 distractors) -> accuracy vs length  [> {args.train_length} = extrapolation]")
    for L in lengths:
        print(f"    L={L:4d} | {score(model, task, L, 1, 4, hops=1, device=args.device):.2f}")
    print("-" * 72)
    print("NIAH multi-key (40 distractors) -> accuracy vs length")
    for L in lengths:
        print(f"    L={L:4d} | {score(model, task, L, 1, 40, hops=1, device=args.device):.2f}")
    print("-" * 72)
    print(f"variable tracking -> accuracy vs chain length, at length {args.train_length}")
    print("    (multi-hop = k reads for a k-chain; single-read = 1 read, the ablation)")
    for c in range(1, args.hops + 1):
        full = score(model, task, args.train_length, c, 6, hops=c, device=args.device)
        abl = score(model, task, args.train_length, c, 6, hops=1, device=args.device)
        print(f"    chain={c} | multi-hop {full:.2f} | single-read {abl:.2f}")


if __name__ == "__main__":
    main()
