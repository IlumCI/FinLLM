"""Stage B partner randomization -- pressuring a canonical latent code.

The held-out-partner test showed a single (speaker, listener) pair invents a
*private* code: a zero-shot partner swap collapses below the no-message prior. The
standard fix for zero-shot coordination (Other-Play, arXiv:2003.02979) is to stop
an agent from overfitting one partner -- train a **population** and pair members at
random, so every speaker must be understood by many listeners and every listener
must understand many speakers. That pressures a common, partner-independent code.

Setup: ``P`` speakers and ``Q`` listeners (each listener owns its receiver stack:
a channel + a LAMb, as in Stage B). The message is a speaker's raw Coconut thought;
any listener may decode it. The diagonal pairings ``(i, i)`` are **held out** from
training; every off-diagonal pairing is trained by random pairing. Then the
held-out pairings are evaluated zero-shot -- if partner randomization worked they
now coordinate (unlike the single co-adapted pair's 0.004 swap), because the code
is shared rather than private.

One joint objective (every listener's answer cross-entropy, on a randomly assigned
speaker each step) is back-propagated through the messages into the speakers --
population differentiable inter-agent learning. Still pure latent; no token is
exchanged.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Tuple

import torch

from ._native import verify
from .coconut import _masked_ce, coconut_collate
from .comm import Channel, CommTask, _decode_answer
from .config import CommConfig, ModelConfig
from .model.lamb import LAMb
from .model.lamb import build_model
from .tokenizer import ArithmeticTokenizer

Sample = Tuple[str, str, str, str]


def _message(speaker: LAMb, cfg: CommConfig, tok: ArithmeticTokenizer,
             a_views: List[str], device: str) -> torch.Tensor:
    prompt, *_ = coconut_collate([(v, "0") for v in a_views], tok, device)
    x = speaker.embed(prompt["input_ids"], prompt["abacus_ids"], prompt["value"], prompt["value_mask"])
    x, _ = speaker._roll_thoughts(x, prompt["pad_mask"], cfg.n_msg)
    return x[:, -cfg.n_msg:, :]


def _prefix(listener: LAMb, channel: Channel, cfg: CommConfig, tok: ArithmeticTokenizer,
            b_views: List[str], messages: torch.Tensor, device: str):
    prompt, *_ = coconut_collate([(v, "0") for v in b_views], tok, device)
    x = listener.embed(prompt["input_ids"], prompt["abacus_ids"], prompt["value"], prompt["value_mask"])
    inj = channel(messages)
    pad = torch.cat([prompt["pad_mask"],
                     torch.zeros(x.size(0), inj.size(1), dtype=torch.bool, device=device)], dim=1)
    x = torch.cat([x, inj], dim=1)
    return listener._roll_thoughts(x, pad, cfg.n_listen_thoughts)


def _answer_logits(listener: LAMb, channel: Channel, cfg: CommConfig, tok: ArithmeticTokenizer,
                   samples: List[Sample], messages: torch.Tensor, device: str):
    x, pad = _prefix(listener, channel, cfg, tok, [s[1] for s in samples], messages, device)
    _, aids, aab, apad, targets, tmask = coconut_collate(
        [(s[1], s[2]) for s in samples], tok, device)
    z = torch.zeros_like(aab, dtype=torch.float32)
    a = listener.embed(aids, aab, z, z)
    x = torch.cat([x, a], dim=1)
    pad = torch.cat([pad, apad], dim=1)
    h, _ = listener.core(x, pad)
    logits = listener._readout(h)
    return logits[:, -(aids.size(1) + 1):, :], targets, tmask


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


class PopulationComm:
    def __init__(self, cfg: CommConfig, tok: ArithmeticTokenizer,
                 n_speakers: Optional[int] = None, n_listeners: Optional[int] = None):
        self.cfg = cfg
        self.tok = tok
        self.device = cfg.device
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        self.P = n_speakers or cfg.pop_speakers
        self.Q = n_listeners or cfg.pop_listeners
        mcfg = ModelConfig(d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=2 * cfg.d_model,
                           n_prelude=1, n_recurrent=1, n_coda=1, recurrent_steps=cfg.recurrent_steps)
        self.speakers = [build_model(mcfg, tok).to(self.device) for _ in range(self.P)]
        self.listeners = [build_model(mcfg, tok).to(self.device) for _ in range(self.Q)]
        self.channels = [Channel(cfg.d_model, cfg.bottleneck, cfg.channel_noise).to(self.device)
                         for _ in range(self.Q)]
        # Held-out pairings: the diagonal (i, i). Never trained; evaluated zero-shot.
        self.holdout = {(i, i) for i in range(min(self.P, self.Q))}

        params: List[torch.nn.Parameter] = []
        for m in self.speakers + self.listeners + self.channels:
            params += list(m.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.task = CommTask(cfg.a_digits, cfg.b_digits, cfg.ops, cfg.seed)
        self.max_ans = cfg.max_answer_len()

    def _speakers_for(self, j: int) -> List[int]:
        return [i for i in range(self.P) if (i, j) not in self.holdout]

    def _lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup:
            return c.lr * (step + 1) / max(1, c.warmup)
        prog = (step - c.warmup) / max(1, c.steps - c.warmup)
        return 0.5 * c.lr * (1.0 + torch.cos(torch.tensor(prog * 3.141592653589793)).item())

    def _train_step(self, step: int) -> float:
        for m in self.speakers + self.listeners + self.channels:
            m.train()
        loss = torch.zeros((), device=self.device)
        for j in range(self.Q):  # every listener trained each step, on a random partner
            i = random.choice(self._speakers_for(j))
            samples = self.task.sample(self.cfg.batch_size)
            messages = _message(self.speakers[i], self.cfg, self.tok, [s[0] for s in samples], self.device)
            logits, targets, tmask = _answer_logits(
                self.listeners[j], self.channels[j], self.cfg, self.tok, samples, messages, self.device)
            loss = loss + _masked_ce(logits, targets, tmask)
        loss = loss / self.Q

        for g in self.opt.param_groups:
            g["lr"] = self._lr_at(step)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.opt.param_groups[0]["params"], self.cfg.grad_clip)
        self.opt.step()
        return float(loss.detach())

    @torch.no_grad()
    def accuracy(self, i: int, j: int, n_tasks: Optional[int] = None, blank: bool = False) -> float:
        for m in self.speakers + self.listeners + self.channels:
            m.eval()
        n = n_tasks or self.cfg.eval_tasks
        samples = self._eval_set(n)
        messages = _message(self.speakers[i], self.cfg, self.tok, [s[0] for s in samples], self.device)
        if blank:
            messages = torch.zeros_like(messages)
        x, pad = _prefix(self.listeners[j], self.channels[j], self.cfg, self.tok,
                         [s[1] for s in samples], messages, self.device)
        ans = _decode_answer(self.listeners[j], x, pad, self.tok, self.max_ans, self.device)
        return sum(a is not None and verify(s[3], a) for s, a in zip(samples, ans)) / n

    def _eval_set(self, n: int) -> List[Sample]:
        holdout = CommTask(self.cfg.a_digits, self.cfg.b_digits, self.cfg.ops,
                           self.cfg.seed * 7 + 99_991)
        return holdout.sample(n)

    @torch.no_grad()
    def eval_grid(self, n_tasks: Optional[int] = None) -> Dict[str, object]:
        trained, held = [], []
        for i in range(self.P):
            for j in range(self.Q):
                acc = self.accuracy(i, j, n_tasks)
                (held if (i, j) in self.holdout else trained).append(acc)
        blank = self.accuracy(0, 0, n_tasks, blank=True)
        return {"trained_mean": _mean(trained), "holdout_mean": _mean(held),
                "holdout_min": min(held) if held else float("nan"),
                "holdout_max": max(held) if held else float("nan"), "blank": blank}

    def train(self) -> Dict[str, object]:
        c = self.cfg
        print(f"[comm-pop] {self.P} speakers x {self.Q} listeners, diagonal held out "
              f"({len(self.holdout)} pairings zero-shot); task X({c.a_digits}d) op Y({c.b_digits}d)")
        run = 0.0
        grid: Dict[str, object] = {}
        for step in range(c.steps):
            run += self._train_step(step)
            if (step + 1) % c.log_every == 0:
                print(f"  step {step+1:5d}/{c.steps}  loss {run / c.log_every:.4f}  lr {self._lr_at(step):.2e}")
                run = 0.0
            if (step + 1) % c.eval_every == 0 or step == c.steps - 1:
                grid = self.eval_grid()
                print(f"  [eval @ {step+1}] trained-pair {grid['trained_mean']:.3f}   "
                      f"held-out zero-shot {grid['holdout_mean']:.3f} "
                      f"(min {grid['holdout_min']:.3f}, max {grid['holdout_max']:.3f})   "
                      f"blank {grid['blank']:.3f}")
        return grid


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LAMb Stage B: partner randomization (canonical code)")
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--speakers", type=int, default=CommConfig.pop_speakers)
    p.add_argument("--listeners", type=int, default=CommConfig.pop_listeners)
    p.add_argument("--a-digits", type=int, default=CommConfig.a_digits)
    p.add_argument("--b-digits", type=int, default=CommConfig.b_digits)
    p.add_argument("--n-msg", type=int, default=CommConfig.n_msg)
    p.add_argument("--d-model", type=int, default=CommConfig.d_model)
    p.add_argument("--eval-tasks", type=int, default=256)
    p.add_argument("--seed", type=int, default=CommConfig.seed)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = CommConfig(steps=args.steps, batch_size=args.batch_size, a_digits=args.a_digits,
                     b_digits=args.b_digits, n_msg=args.n_msg, d_model=args.d_model,
                     pop_speakers=args.speakers, pop_listeners=args.listeners,
                     eval_tasks=args.eval_tasks, seed=args.seed)
    tok = ArithmeticTokenizer()
    grid = PopulationComm(cfg, tok).train()
    verdict = "CANONICAL (zero-shot coordination emerges)" if grid.get("holdout_mean", 0) >= 0.6 \
        else "still private (held-out pairings do not coordinate)"
    print(f"  => held-out zero-shot mean {grid.get('holdout_mean', float('nan')):.3f} vs "
          f"single-pair swap baseline ~0.02  =>  {verdict}")


if __name__ == "__main__":
    main()
