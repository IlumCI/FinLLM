"""Long-context memory benchmark -- needle/passkey retrieval at length.

Exercises the ``infContext`` pillar: LAMb's test-time neural memory
(:class:`lamb.model.memory.NeuralMemory`). A target key->value binding (the
"needle") is planted at the FRONT of a sequence; distractor bindings and a long
run of uninformative "filler" tokens fill the "haystack"; a query for the target
key is appended at the END. The model must recall the target's value across the
whole sequence.

Two properties are measured, both signatures of a fixed-size test-time memory:

* **Unbounded context length.** Accuracy stays high as the needle-to-query
  distance (sequence length) grows, *including lengths not seen in training* --
  the memory's O(1) state is length-agnostic, so retrieval extrapolates.
* **Bounded capacity.** Accuracy degrades as the number of simultaneous bindings
  grows past what the fixed-size state can hold -- the honest Titans/ATLAS
  tradeoff (unbounded length, finite capacity).

A memoryless ablation (readout on the query token alone, no cross-token memory)
sits at chance, confirming the neural memory is what performs the retrieval.
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

FILLER, PAIR, QUERY = 0, 1, 2


@dataclass
class RecallConfig:
    n_keys: int = 64
    n_vals: int = 32
    n_filler: int = 16
    d_model: int = 64
    d_mem: int = 64


class RecallTask:
    """Generates needle-in-a-haystack associative-recall episodes."""

    def __init__(self, cfg: RecallConfig):
        self.cfg = cfg

    def batch(self, bs: int, length: int, n_pairs: int, rng: random.Random,
              device: str = "cpu", front: bool = True):
        n_pairs = min(n_pairs, self.cfg.n_keys, max(1, length - 1))
        key_ids = torch.zeros(bs, length, dtype=torch.long)
        val_ids = torch.zeros(bs, length, dtype=torch.long)
        kind = torch.zeros(bs, length, dtype=torch.long)
        filler_ids = torch.randint(0, self.cfg.n_filler, (bs, length))
        target_val = torch.zeros(bs, dtype=torch.long)

        for b in range(bs):
            keys = rng.sample(range(self.cfg.n_keys), n_pairs)
            vals = [rng.randrange(self.cfg.n_vals) for _ in range(n_pairs)]
            # `front`: needle at position 0 (retrieval distance = full length, the
            # eval stress case). Otherwise a random placement (varied distance),
            # used in training so the memory learns retention across distances.
            slots = rng.sample(range(0, length - 1), n_pairs)
            if front:
                slots = [0] + rng.sample([s for s in range(1, length - 1) if s != 0], n_pairs - 1)
            positions = slots
            for pos, k, v in zip(positions, keys, vals):
                kind[b, pos] = PAIR
                key_ids[b, pos] = k
                val_ids[b, pos] = v
            kind[b, length - 1] = QUERY
            key_ids[b, length - 1] = keys[0]  # query the target key
            target_val[b] = vals[0]

        return (
            {"key_ids": key_ids.to(device), "val_ids": val_ids.to(device),
             "kind": kind.to(device), "filler_ids": filler_ids.to(device)},
            target_val.to(device),
        )


class _RecallBase(nn.Module):
    def __init__(self, cfg: RecallConfig):
        super().__init__()
        self.cfg = cfg
        self.key_emb = nn.Embedding(cfg.n_keys, cfg.d_model)
        self.val_emb = nn.Embedding(cfg.n_vals, cfg.d_model)
        self.filler_emb = nn.Embedding(cfg.n_filler, cfg.d_model)
        self.query_marker = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.norm = RMSNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, cfg.n_vals)

    def _embed(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        kind = batch["kind"]
        pair_e = self.key_emb(batch["key_ids"]) + self.val_emb(batch["val_ids"])
        query_e = self.key_emb(batch["key_ids"]) + self.query_marker
        filler_e = self.filler_emb(batch["filler_ids"])
        is_pair = (kind == PAIR).unsqueeze(-1).float()
        is_query = (kind == QUERY).unsqueeze(-1).float()
        is_filler = (kind == FILLER).unsqueeze(-1).float()
        return is_filler * filler_e + is_pair * pair_e + is_query * query_e


class MemoryRecallModel(_RecallBase):
    def __init__(self, cfg: RecallConfig):
        super().__init__(cfg)
        self.memory = NeuralMemory(cfg.d_model, cfg.d_mem)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self._embed(batch)
        read = self.memory(emb)          # (B, L, d) causal associative read
        return self.readout(self.norm(read[:, -1]))


class MemorylessModel(_RecallBase):
    """Ablation: no cross-token memory -- the query token alone drives the readout."""

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self._embed(batch)
        return self.readout(self.norm(emb[:, -1]))


def train_recall(model: nn.Module, task: RecallTask, steps: int, length: int, n_pairs: int,
                 bs: int = 64, lr: float = 2e-3, seed: int = 0, device: str = "cpu") -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    model.train()
    for _ in range(steps):
        # Vary length, needle placement, and binding count so the memory learns
        # retention across distances and generalises across capacity loads.
        cur_len = rng.randint(max(6, length // 3), length)
        cur_pairs = rng.randint(2, max(2, n_pairs))
        batch, target = task.batch(bs, cur_len, cur_pairs, rng, device, front=False)
        loss = F.cross_entropy(model(batch), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def eval_recall(model: nn.Module, task: RecallTask, length: int, n_pairs: int,
                bs: int = 128, reps: int = 4, seed: int = 999, device: str = "cpu") -> float:
    model.eval()
    rng = random.Random(seed)
    ok = tot = 0
    for _ in range(reps):
        batch, target = task.batch(bs, length, n_pairs, rng, device)
        pred = model(batch).argmax(dim=-1)
        ok += int((pred == target).sum())
        tot += target.numel()
    return ok / max(1, tot)


def length_curve(model, task, lengths, n_pairs, device="cpu") -> Dict[int, float]:
    return {L: eval_recall(model, task, L, n_pairs, device=device) for L in lengths}


def capacity_curve(model, task, length, pair_counts, device="cpu") -> Dict[int, float]:
    return {p: eval_recall(model, task, length, p, device=device) for p in pair_counts}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="LAMb long-context memory benchmark (needle/passkey).")
    p.add_argument("--train-length", type=int, default=48)
    p.add_argument("--train-pairs", type=int, default=12,
                   help="max simultaneous bindings seen in training (sampled 2..this)")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--d-mem", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)

    cfg = RecallConfig(d_mem=args.d_mem)
    task = RecallTask(cfg)
    lengths = [16, args.train_length, 96, 192, 384]
    chance = 1.0 / cfg.n_vals

    mem = MemoryRecallModel(cfg)
    train_recall(mem, task, args.steps, args.train_length, args.train_pairs, seed=args.seed, device=args.device)
    ablation = MemorylessModel(cfg)
    train_recall(ablation, task, args.steps, args.train_length, args.train_pairs, seed=args.seed, device=args.device)

    print(f"long-context memory benchmark | chance {chance:.3f} | trained at length {args.train_length}, "
          f"up to {args.train_pairs} bindings, d_mem {cfg.d_mem}")
    print("-" * 72)
    curve_pairs = 4
    mem_len = length_curve(mem, task, lengths, curve_pairs, device=args.device)
    abl_len = length_curve(ablation, task, lengths, curve_pairs, device=args.device)
    print(f"length (distance) -> retrieval accuracy, {curve_pairs} bindings  "
          f"[> {args.train_length} = extrapolation]")
    for L in lengths:
        tag = "  (train)" if L == args.train_length else ("  (extrapolation)" if L > args.train_length else "")
        print(f"    L={L:4d} | memory {mem_len[L]:.2f} | memoryless {abl_len[L]:.2f}{tag}")

    print("-" * 72)
    caps = [4, 16, 32, 64]
    mem_cap = capacity_curve(mem, task, 96, caps, device=args.device)
    print("simultaneous bindings -> accuracy at length 96  (fixed d_mem capacity tradeoff)")
    for c in caps:
        print(f"    pairs={c:2d} | memory {mem_cap[c]:.2f}")


if __name__ == "__main__":
    main()
