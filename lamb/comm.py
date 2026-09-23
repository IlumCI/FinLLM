"""Stage B -- latent inter-agent communication (the message *is* a thought).

Two LAMb agents solve a split problem neither can solve alone. The **speaker**
sees operand ``X``; the **listener** sees the operator and operand ``Y`` and must
emit ``X op Y``. The listener cannot recover the answer without ``X`` -- so the
speaker has to communicate it, but not in language. The speaker rolls ``n_msg``
Coconut thought vectors (Stage A's exact continuous-thought mechanism) from its
view of ``X``; those *continuous* vectors, passed through a differentiable
``Channel``, are the message. The listener injects them as latent input positions
-- receiving is literally thinking another agent's thought, the same
`_roll_thoughts` interface it uses for its own -- then reasons on and answers.

Training is joint and end to end: the listener's answer cross-entropy
back-propagates *through the message* into the speaker (differentiable inter-agent
learning; Foerster et al., DIAL, arXiv:1605.06676). No token is ever exchanged;
the channel is pure latent, a blackbox -- the reason for a latent-space model in
the first place (nothing is verbalised, nothing computes a justification).

Two diagnostics from the emergent-communication literature make the claim honest:

  * **Zeroed-message ablation** -- blank the channel at eval. If accuracy collapses
    to the no-information prior, the channel is doing the work (the causal
    influence of communication; the gap ``comm - blank`` is the channel's value).
  * **Bandwidth sweep** -- vary the channel bottleneck. Accuracy vs. capacity is
    the honest tradeoff: a 1-wide channel cannot carry a 1-digit operand, a wide
    one can.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._native import evaluate, verify
from .holdout import is_heldout
from .coconut import coconut_collate, _masked_ce
from .config import CommConfig, ModelConfig
from .device import Amp, add_hardware_args, device_report, resolve_device, resolve_hardware
from .model.lamb import LAMb, build_model
from .model.transformer import RMSNorm
from .tokenizer import ArithmeticTokenizer

# (a_view, b_view, answer, full_expr)
Sample = Tuple[str, str, str, str]


class CommTask:
    """Sampler for the split problem: speaker sees ``X``, listener sees ``op Y``."""

    def __init__(self, a_digits: int, b_digits: int, ops: Tuple[str, ...], seed: int,
                 split: str = "any"):
        """``split``: 'train' excludes the eval partition, 'eval' keeps only it.

        The partition is by a hash of the problem (:mod:`lamb.holdout`), not by
        seed -- at 1 digit the whole space is 200 problems, so seed separation
        leaves an eval set that is 100% memorised.
        """
        self.a_digits, self.b_digits, self.ops = a_digits, b_digits, tuple(ops)
        self.split = split
        self.rng = random.Random(seed)

    def _num(self, digits: int) -> int:
        if digits <= 1:
            return self.rng.randint(0, 9)
        return self.rng.randint(10 ** (digits - 1), 10 ** digits - 1)

    def sample(self, n: int) -> List[Sample]:
        """``n`` samples from this split.

        Flat rejection sampling, not recursion: topping up by calling ``sample``
        again let the *inner* call's repeat-padding fire before the outer loop had
        drawn a single fresh sample, so a partition that merely needed more draws
        was padded with duplicates instead. Draws are bounded, then -- and only
        then -- a genuinely small partition is cycled to length.
        """
        out: List[Sample] = []
        budget = 64 * max(1, n)          # bounded: the eval split accepts ~15%
        while len(out) < n and budget > 0:
            budget -= 1
            x, y = self._num(self.a_digits), self._num(self.b_digits)
            op = self.rng.choice(self.ops)
            full = f"{x}{op}{y}"
            if self.split == "train" and is_heldout(full):
                continue
            if self.split == "eval" and not is_heldout(full):
                continue
            val = evaluate(full)
            if val is None:  # never for +/- at these widths, but stay safe
                continue
            out.append((str(x), f"{op}{y}", str(val), full))
        if out:                       # a tiny partition must repeat; cycle, do not
            base = list(out)          # hammer one element
            while len(out) < n:
                out.append(base[len(out) % len(base)])
        return out[:n]

    def unique_count(self, split: Optional[str] = None) -> int:
        """Exact number of distinct problems in a split -- the eval resolution.

        At 1 digit the whole space is 200 problems, so the eval partition holds
        about 30: an accuracy measured there has a granularity of ~3 points
        however many samples are drawn. Worth knowing before reading a number.
        """
        sp = self.split if split is None else split
        def bounds(d: int):
            return (0, 9) if d <= 1 else (10 ** (d - 1), 10 ** d - 1)

        lo_a, hi_a = bounds(self.a_digits)
        lo_b, hi_b = bounds(self.b_digits)
        k = 0
        for x in range(lo_a, hi_a + 1):
            for y in range(lo_b, hi_b + 1):
                for op in self.ops:
                    full = f"{x}{op}{y}"
                    if sp == "train" and is_heldout(full):
                        continue
                    if sp == "eval" and not is_heldout(full):
                        continue
                    k += 1
        return k


class Channel(nn.Module):
    """The differentiable communication medium: speaker space -> listener space.

    A low-rank bottleneck ``c`` (< d) throttles bandwidth -- the capacity knob. The
    RMSNorm lands the message in input-embedding statistics so the listener's core
    consumes it stably; ``recv_marker`` (learned, zero-init) tags a received
    message. ``down`` then ``up`` with ``c == d`` is a full-width learned channel.
    """

    def __init__(self, d: int, bottleneck: int = 0, noise: float = 0.0):
        super().__init__()
        c = d if (not bottleneck or bottleneck >= d) else bottleneck
        self.width = c
        self.noise = noise
        self.down = nn.Linear(d, c, bias=False)
        self.up = nn.Linear(c, d, bias=False)
        self.norm = RMSNorm(d)
        self.recv_marker = nn.Parameter(torch.zeros(d))

    def forward(self, msg: torch.Tensor) -> torch.Tensor:
        code = self.down(msg)                                  # (B, M, c) -- the transmitted code
        if self.noise > 0.0:
            # DRU (DIAL, arXiv:1605.06676): bound the signal with tanh so a fixed
            # additive corruption has a fixed relative effect (the sender cannot
            # escape noise by amplifying the code). A wider code then carries more
            # signal against that per-dimension noise, so capacity -- and accuracy
            # -- trades off against width, and the sender learns a robust code.
            code = torch.tanh(code)
            if self.training:
                code = code + self.noise * torch.randn_like(code)
        return self.norm(self.up(code)) + self.recv_marker


def _decode_answer(model: LAMb, x: torch.Tensor, pad: torch.Tensor, tok: ArithmeticTokenizer,
                   max_len: int, device: str) -> List[Optional[str]]:
    """Greedy-decode an answer from an already-assembled embedding sequence.

    ``x`` ends at the listener's ``=`` (plus any injected message / thought
    positions); the last real *token* is ``=``, so Abacus tracking starts there.
    """
    b = x.size(0)
    last_id = torch.full((b,), tok.EQ, dtype=torch.long, device=device)
    last_ab = torch.zeros((b,), dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    cols: List[torch.Tensor] = []
    for _ in range(max_len):
        h_core, _ = model.core(x, pad)
        nxt = model._readout(h_core)[:, -1, :].argmax(dim=-1)
        nxt = torch.where(done, torch.full_like(nxt, tok.PAD), nxt)
        new_ab = torch.zeros_like(nxt)
        for i in range(b):
            new_ab[i] = tok.abacus_after(int(last_id[i]), int(last_ab[i]), int(nxt[i]))
        z = torch.zeros((b, 1), dtype=torch.float32, device=device)
        x = torch.cat([x, model.embed(nxt[:, None], new_ab[:, None], z, z)], dim=1)
        pad = torch.cat([pad, done.unsqueeze(1)], dim=1)
        cols.append(nxt)
        last_id, last_ab = nxt, new_ab
        done = done | (nxt == tok.EOS)
        if bool(done.all()):
            break
    gen = torch.stack(cols, dim=1) if cols else torch.zeros((b, 0), dtype=torch.long)
    out: List[Optional[str]] = []
    for row in gen.tolist():
        ans: List[int] = []
        for tid in row:
            if tid == tok.PAD:
                break
            ans.append(tid)
            if tid == tok.EOS:
                break
        out.append(tok.decode_answer(ans))
    return out


class CommTrainer:
    def __init__(self, cfg: CommConfig, tokenizer: ArithmeticTokenizer):
        self.cfg = cfg
        self.tok = tokenizer
        self.device = resolve_device(cfg.device)
        self.amp = Amp(self.device, cfg.amp)
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        mcfg = ModelConfig(d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=2 * cfg.d_model,
                           n_prelude=1, n_recurrent=1, n_coda=1, recurrent_steps=cfg.recurrent_steps)
        self.speaker = build_model(mcfg, tokenizer).to(self.device)
        self.listener = build_model(mcfg, tokenizer).to(self.device)
        self.channel = Channel(cfg.d_model, cfg.bottleneck, cfg.channel_noise).to(self.device)

        params = (list(self.speaker.parameters()) + list(self.listener.parameters())
                  + list(self.channel.parameters()))
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.task = CommTask(cfg.a_digits, cfg.b_digits, cfg.ops, cfg.seed, split="train")
        self.max_ans = cfg.max_answer_len()

    # -- communication primitives ----------------------------------------
    def _message(self, a_views: List[str], dropout: float = 0.0,
                 speaker: Optional[LAMb] = None) -> torch.Tensor:
        """Speaker rolls ``n_msg`` latent thoughts from its view of X -> (B, M, d).

        ``speaker`` overrides ``self.speaker`` -- used by the held-out-partner test
        to feed a *foreign* speaker's message into this pair's receiver stack.
        """
        spk = speaker if speaker is not None else self.speaker
        prompt, *_ = coconut_collate([(v, "0") for v in a_views], self.tok, self.device)
        x = spk.embed(prompt["input_ids"], prompt["abacus_ids"],
                      prompt["value"], prompt["value_mask"])
        x, _ = spk._roll_thoughts(x, prompt["pad_mask"], self.cfg.n_msg, thought_dropout=dropout)
        return x[:, -self.cfg.n_msg:, :]

    def _listen_prefix(self, b_views: List[str], messages: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Listener prompt embeddings + injected message + own thoughts."""
        prompt, *_rest = coconut_collate([(v, "0") for v in b_views], self.tok, self.device)
        x = self.listener.embed(prompt["input_ids"], prompt["abacus_ids"],
                                prompt["value"], prompt["value_mask"])
        inj = self.channel(messages)                                   # (B, M, d)
        pad = torch.cat([prompt["pad_mask"],
                         torch.zeros(x.size(0), inj.size(1), dtype=torch.bool, device=self.device)], dim=1)
        x = torch.cat([x, inj], dim=1)
        x, pad = self.listener._roll_thoughts(x, pad, self.cfg.n_listen_thoughts)
        return x, pad, prompt

    def _answer_logits(self, samples: List[Sample], messages: torch.Tensor):
        b_views = [s[1] for s in samples]
        answers = [s[2] for s in samples]
        x, pad, _ = self._listen_prefix(b_views, messages)
        _, aids, aab, apad, targets, tmask = coconut_collate(
            [(bv, a) for bv, a in zip(b_views, answers)], self.tok, self.device)
        z = torch.zeros_like(aab, dtype=torch.float32)
        a = self.listener.embed(aids, aab, z, z)
        x = torch.cat([x, a], dim=1)
        pad = torch.cat([pad, apad], dim=1)
        h_core, _ = self.listener.core(x, pad)
        logits = self.listener._readout(h_core)
        return logits[:, -(aids.size(1) + 1):, :], targets, tmask

    # -- training ---------------------------------------------------------
    def _lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup:
            return c.lr * (step + 1) / max(1, c.warmup)
        prog = (step - c.warmup) / max(1, c.steps - c.warmup)
        return 0.5 * c.lr * (1.0 + torch.cos(torch.tensor(prog * 3.141592653589793)).item())

    def _train_step(self, step: int) -> float:
        self.speaker.train(); self.listener.train(); self.channel.train()
        samples = self.task.sample(self.cfg.batch_size)
        for g in self.opt.param_groups:
            g["lr"] = self._lr_at(step)
        with self.amp.autocast():
            messages = self._message([s[0] for s in samples], dropout=self.cfg.msg_dropout)
            logits, targets, tmask = self._answer_logits(samples, messages)
            loss = _masked_ce(logits, targets, tmask)
        self.amp.backward_step(loss, self.opt, self.opt.param_groups[0]["params"], self.cfg.grad_clip)
        return float(loss.detach())

    # -- evaluation -------------------------------------------------------
    @torch.no_grad()
    def accuracy(self, n_tasks: Optional[int] = None, blank: bool = False,
                 speaker: Optional[LAMb] = None) -> float:
        """Exact-match accuracy. ``blank`` zeroes the message (no-information
        ablation): the listener then sees only ``op Y`` and the constant channel
        marker, so this is the no-communication prior. ``speaker`` overrides the
        message source (a foreign partner, for the held-out-partner test)."""
        self.speaker.eval(); self.listener.eval(); self.channel.eval()
        if speaker is not None:
            speaker.eval()
        n = n_tasks or self.cfg.eval_tasks
        samples = self._eval_set(n)
        messages = self._message([s[0] for s in samples], speaker=speaker)
        if blank:
            messages = torch.zeros_like(messages)
        x, pad, _ = self._listen_prefix([s[1] for s in samples], messages)
        ans = _decode_answer(self.listener, x, pad, self.tok, self.max_ans, self.device)
        return sum(a is not None and verify(s[3], a) for s, a in zip(samples, ans)) / n

    @torch.no_grad()
    def channel_report(self, n: int = 64) -> Tuple[float, float]:
        """Message diagnostics (positive *signalling*, and collapse).

        Returns ``(signal_std, msg_cos)`` over messages for a batch of distinct
        inputs: ``signal_std`` is how much the message varies with the speaker's
        input (~0 => the speaker emits a constant, no signalling), ``msg_cos`` is
        the mean pairwise cosine of the message vectors (~1 => collapsed to a
        single direction; the representational-collapse detector of
        arXiv:2604.03809). The comm-minus-blank accuracy gap is the complementary
        *listening* test (arXiv:1903.05168): signalling alone is not enough.
        """
        self.speaker.eval(); self.channel.eval()
        samples = self._eval_set(n)
        msg = self._message([s[0] for s in samples]).reshape(n, -1)  # (n, M*d)
        signal_std = float(msg.std(dim=0).mean())
        fn = F.normalize(msg, dim=-1)
        sim = fn @ fn.t()
        msg_cos = float((sim.sum() - sim.diag().sum()) / (n * (n - 1)))
        return signal_std, msg_cos

    def _eval_set(self, n: int) -> List[Sample]:
        holdout = CommTask(self.cfg.a_digits, self.cfg.b_digits, self.cfg.ops,
                           self.cfg.seed * 7 + 99_991, split="eval")
        return holdout.sample(n)

    def train(self) -> None:
        c = self.cfg
        print(f"[comm] {device_report(self.device, self.amp)}")
        print(f"[comm] task=X({c.a_digits}d) op Y({c.b_digits}d) ops={''.join(c.ops)} "
              f"n_msg={c.n_msg} listen_thoughts={c.n_listen_thoughts} "
              f"channel_width={self.channel.width}/{c.d_model} "
              f"params={self.speaker.num_params()}x2+{sum(p.numel() for p in self.channel.parameters())}")
        run = 0.0
        for step in range(c.steps):
            run += self._train_step(step)
            if (step + 1) % c.log_every == 0:
                print(f"  step {step+1:5d}/{c.steps}  loss {run / c.log_every:.4f}  lr {self._lr_at(step):.2e}")
                run = 0.0
            if (step + 1) % c.eval_every == 0 or step == c.steps - 1:
                comm = self.accuracy()
                blank = self.accuracy(blank=True)
                sig, cos = self.channel_report()
                print(f"  [eval @ {step+1}] comm acc {comm:.3f}   blank(no-msg) {blank:.3f}   "
                      f"channel gain {comm - blank:+.3f}   [signal_std {sig:.3f} msg_cos {cos:.3f}]")


def sweep_bandwidth(cfg: CommConfig, tokenizer: ArithmeticTokenizer) -> Dict[int, Tuple[float, float]]:
    """Train a fresh pair at each channel width; return {width: (comm, blank)}."""
    out: Dict[int, Tuple[float, float]] = {}
    for bw in cfg.eval_bottlenecks:
        sub = CommConfig(**{**cfg.__dict__, "bottleneck": bw})
        tr = CommTrainer(sub, tokenizer)
        for step in range(sub.steps):
            tr._train_step(step)
        width = tr.channel.width
        out[width] = (tr.accuracy(), tr.accuracy(blank=True))
        print(f"  [bandwidth] width={width:>3}/{cfg.d_model}  comm {out[width][0]:.3f}  "
              f"blank {out[width][1]:.3f}  gain {out[width][0] - out[width][1]:+.3f}")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LAMb Stage B: latent inter-agent communication")
    p.add_argument("--steps", type=int, default=CommConfig.steps)
    p.add_argument("--batch-size", type=int, default=CommConfig.batch_size)
    p.add_argument("--a-digits", type=int, default=CommConfig.a_digits)
    p.add_argument("--b-digits", type=int, default=CommConfig.b_digits)
    p.add_argument("--n-msg", type=int, default=CommConfig.n_msg)
    p.add_argument("--n-listen-thoughts", type=int, default=CommConfig.n_listen_thoughts)
    p.add_argument("--bottleneck", type=int, default=CommConfig.bottleneck,
                   help="channel width (0 => full d_model); the bandwidth knob")
    p.add_argument("--channel-noise", type=float, default=CommConfig.channel_noise,
                   help="DRU-style bottleneck noise (train only); >0 gives the channel finite capacity")
    p.add_argument("--d-model", type=int, default=CommConfig.d_model)
    p.add_argument("--recurrent-steps", type=int, default=CommConfig.recurrent_steps)
    p.add_argument("--seed", type=int, default=CommConfig.seed)
    p.add_argument("--sweep", action="store_true", help="sweep channel bandwidth instead of one run")
    add_hardware_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device, amp = resolve_hardware(args)
    cfg = CommConfig(
        steps=args.steps, batch_size=args.batch_size, a_digits=args.a_digits,
        b_digits=args.b_digits, n_msg=args.n_msg, n_listen_thoughts=args.n_listen_thoughts,
        bottleneck=args.bottleneck, channel_noise=args.channel_noise, d_model=args.d_model,
        recurrent_steps=args.recurrent_steps, seed=args.seed, device=device, amp=amp,
    )
    tok = ArithmeticTokenizer()
    if args.sweep:
        print(f"[comm] bandwidth sweep over {cfg.eval_bottlenecks}")
        sweep_bandwidth(cfg, tok)
    else:
        CommTrainer(cfg, tok).train()


if __name__ == "__main__":
    main()
