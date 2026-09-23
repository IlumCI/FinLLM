"""Stage A, restructured -- LOTUS-style parallel supervised latent reasoning.

Stage A's first design followed Coconut: continuous thoughts generated
**autoregressively**, one latent at a time, each conditioned on the last, with only
the final answer supervised. It worked at tiny scale -- and the literature says
that is exactly where it works. Measured across backbones (arXiv:2606.31779), the
sequential continuous-thought family's gap to explicit chain-of-thought *widens*
with scale (-0.1 pts at 124M, -2.3 at 1B, **-9.2 at 3B**), while the looped,
parallel-supervised family stays flat (-1.5 at 3B). If the point is to scale, the
parallel family is the one to build.

The restructure, then:

* **Parallel, not autoregressive.** ``n_latent`` latent positions are appended
  after the prompt *all at once* and refined by ``loops`` passes through the shared
  core. Cost is ``O(loops)`` forwards **regardless of how many latents there are** --
  where Coconut needed one sequential forward per thought. That is the scalability
  argument: the latent budget grows for free.
* **Every latent position is supervised**, directly through the LM head, instead of
  hoping one answer gradient reaches back through a chain of continuous states.

LOTUS supervises latents against gold chain-of-thought *tokens*. LAMb has no
language -- but it has an exact evaluator, so it generates a gold **numeric** trace
(the intermediate sub-expression values) for free, with no language, no human
annotation and no external data. The self-play verifier supplies precisely the
signal the method needs.

Crucially this costs nothing at inference: the latent positions are **never
decoded**. Only the answer is emitted. The trace is a training signal, not an
output, so the model remains a blackbox that computes its intermediates latently.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .coconut import _masked_ce, coconut_collate
from .comm import _decode_answer
from .config import LotusConfig, ModelConfig
from .device import Amp, add_hardware_args, device_report, resolve_device, resolve_hardware
from .model.lamb import LAMb, build_model
from .model.transformer import RMSNorm
from .selfplay.grammar import Descriptor, TaskGrammar
from .selfplay.verifier import Verifier
from .tokenizer import ArithmeticTokenizer

# (expression, exact answer, intermediate trace values)
Task = Tuple[str, str, List[int]]


def trace_targets(tok: ArithmeticTokenizer, trace: List[int]) -> List[int]:
    """Tokenise the numeric trace into supervision targets (ids only).

    The trace is a *target*, never an input, so no Abacus/value channels are
    needed. Each intermediate is emitted exactly as an answer would be (LSB-first
    digits, a leading ``-`` for negatives); consecutive intermediates are simply
    concatenated -- the Abacus reset at each number boundary is what separates them.
    """
    ids: List[int] = []
    for v in trace:
        e = tok.encode("0", str(v))       # dummy problem; take only the answer span
        ids += e.ids[e.ans_start:-1]      # drop the trailing EOS
    return ids


class LotusReasoner(nn.Module):
    """A LAMb core plus a parallel block of supervised latent positions."""

    def __init__(self, model: LAMb, n_latent: int, loops: int):
        super().__init__()
        self.model = model
        self.n_latent = int(n_latent)
        self.loops = max(1, int(loops))
        d = model.cfg.d_model
        # Learned initial content for each latent slot. They become input-dependent
        # through attention over the prompt, which precedes them causally.
        self.latent_emb = nn.Embedding(self.n_latent, d)
        nn.init.normal_(self.latent_emb.weight, mean=0.0, std=0.02)
        # Feedback interface, mirroring the Coconut path: bounded magnitude plus a
        # learned tag marking a position as latent rather than a token.
        self.latent_norm = RMSNorm(d)
        self.latent_marker = nn.Parameter(torch.zeros(d))

    def latent_block(self, x_prompt: torch.Tensor, pad_prompt: torch.Tensor,
                     n_steps: Optional[int] = None):
        """Append L latent positions and refine them with ``loops`` passes.

        Returns ``(x, pad, latent_hidden)``. One core forward per loop iteration --
        independent of ``n_latent``, because the positions are refined together.
        """
        b, p, _ = x_prompt.shape
        idx = torch.arange(self.n_latent, device=x_prompt.device)
        lat = self.latent_emb(idx).unsqueeze(0).expand(b, -1, -1)
        pad = torch.cat(
            [pad_prompt, torch.zeros(b, self.n_latent, dtype=torch.bool, device=x_prompt.device)],
            dim=1)
        x = torch.cat([x_prompt, lat], dim=1)
        h = None
        for _ in range(self.loops):
            h, _ = self.model.core(x, pad, n_steps)
            lat = self.latent_norm(h[:, p:, :]) + self.latent_marker
            x = torch.cat([x_prompt, lat], dim=1)   # prompt stays as embedded input
        return x, pad, h[:, p:, :]

    def forward(self, prompt: Dict[str, torch.Tensor], answer_ids: torch.Tensor,
                answer_abacus: torch.Tensor, answer_pad: torch.Tensor,
                n_steps: Optional[int] = None):
        """Teacher-forced answer logits **and** per-latent-position logits."""
        m = self.model
        x_prompt = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                           prompt["value"], prompt["value_mask"])
        x, pad, latent_h = self.latent_block(x_prompt, prompt["pad_mask"], n_steps)
        latent_logits = m._readout(latent_h)                     # (B, L, V) -> trace supervision

        z = torch.zeros_like(answer_abacus, dtype=torch.float32)
        x = torch.cat([x, m.embed(answer_ids, answer_abacus, z, z)], dim=1)
        pad = torch.cat([pad, answer_pad], dim=1)
        h, aux = m.core(x, pad, n_steps)
        ans_logits = m._readout(h)[:, -(answer_ids.size(1) + 1):, :]
        return ans_logits, latent_logits, aux

    @torch.no_grad()
    def solve(self, problems: List[str], tok: ArithmeticTokenizer, max_answer_len: int = 20,
              n_steps: Optional[int] = None, device: str = "cpu") -> List[Optional[str]]:
        """Greedy-decode answers. The latent positions are **never decoded**."""
        self.eval()
        prompt, *_ = coconut_collate([(p, "0") for p in problems], tok, device)
        m = self.model
        x_prompt = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                           prompt["value"], prompt["value_mask"])
        x, pad, _ = self.latent_block(x_prompt, prompt["pad_mask"], n_steps)
        return _decode_answer(m, x, pad, tok, max_answer_len, device)


class LotusTrainer:
    def __init__(self, cfg: LotusConfig, tokenizer: ArithmeticTokenizer,
                 model_cfg: Optional[ModelConfig] = None):
        self.cfg = cfg
        self.tok = tokenizer
        self.device = resolve_device(cfg.device)
        self.amp = Amp(self.device, cfg.amp)
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        mcfg = model_cfg or ModelConfig(d_model=96, n_heads=4, d_ff=192,
                                        n_prelude=1, n_recurrent=1, n_coda=1,
                                        recurrent_steps=4)
        model = build_model(mcfg, tokenizer).to(self.device)
        self.reasoner = LotusReasoner(model, cfg.n_latent, cfg.loops).to(self.device)
        self.opt = torch.optim.AdamW(self.reasoner.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.grammar = TaskGrammar()
        self.verifier = Verifier()
        self.descriptor = Descriptor(depth=cfg.depth, digits=cfg.digits, ops_key=cfg.ops_key)
        self.max_ans = cfg.max_answer_len()
        self._seed = cfg.seed * 1_000_003 + 1
        self.truncated = 0   # traces that did not fit in n_latent (diagnostic)

    # -- data -------------------------------------------------------------
    def _sample_batch(self, n: int) -> List[Task]:
        out: List[Task] = []
        for _ in range(n):
            self._seed += 1
            out.append(self.grammar.sample_with_trace(self.descriptor, self._seed))
        return out

    def _eval_set(self, n: int) -> List[Task]:
        rng = random.Random(self.cfg.seed * 7 + 12345)
        return [self.grammar.sample_with_trace(self.descriptor, rng.randint(0, 2 ** 31 - 1))
                for _ in range(n)]

    def _collate(self, tasks: List[Task]):
        prompt, aids, aab, apad, targets, tmask = coconut_collate(
            [(e, a) for e, a, _ in tasks], self.tok, self.device)
        L = self.cfg.n_latent
        b = len(tasks)
        tr_ids = torch.full((b, L), self.tok.PAD, dtype=torch.long, device=self.device)
        tr_mask = torch.zeros((b, L), dtype=torch.float32, device=self.device)
        for i, (_, _, trace) in enumerate(tasks):
            toks = trace_targets(self.tok, trace)
            if len(toks) > L:
                self.truncated += 1
                toks = toks[:L]
            if toks:
                tr_ids[i, :len(toks)] = torch.tensor(toks, device=self.device)
                tr_mask[i, :len(toks)] = 1.0
        return prompt, aids, aab, apad, targets, tmask, tr_ids, tr_mask

    # -- training ---------------------------------------------------------
    def _lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup:
            return c.lr * (step + 1) / max(1, c.warmup)
        prog = (step - c.warmup) / max(1, c.steps - c.warmup)
        return 0.5 * c.lr * (1.0 + torch.cos(torch.tensor(prog * 3.141592653589793)).item())

    def _train_step(self, step: int) -> Dict[str, float]:
        c = self.cfg
        self.reasoner.train()
        tasks = self._sample_batch(c.batch_size)
        prompt, aids, aab, apad, targets, tmask, tr_ids, tr_mask = self._collate(tasks)
        for g in self.opt.param_groups:
            g["lr"] = self._lr_at(step)
        with self.amp.autocast():
            ans_logits, latent_logits, _ = self.reasoner(prompt, aids, aab, apad)
            ans_loss = _masked_ce(ans_logits, targets, tmask)
            # Per-position supervision on the latent block. With trace_coef=0 this
            # reduces to the answer-only ablation (parallel latents, no trace).
            tr_loss = (_masked_ce(latent_logits, tr_ids, tr_mask)
                       if c.trace_coef > 0 and float(tr_mask.sum()) > 0
                       else torch.zeros((), device=ans_logits.device))
            loss = ans_loss + c.trace_coef * tr_loss
        self.amp.backward_step(loss, self.opt, self.reasoner.parameters(), c.grad_clip)
        return {"loss": float(loss.detach()), "ans": float(ans_loss.detach()),
                "trace": float(tr_loss.detach())}

    # -- evaluation -------------------------------------------------------
    @torch.no_grad()
    def accuracy(self, n_tasks: Optional[int] = None) -> float:
        """Exact-match accuracy, decoding only the answer (latents stay latent)."""
        n = n_tasks or self.cfg.eval_tasks
        tasks = self._eval_set(n)
        ans = self.reasoner.solve([e for e, _, _ in tasks], self.tok, self.max_ans,
                                  device=self.device)
        return sum(self.verifier.check(e, a) for (e, _, _), a in zip(tasks, ans)) / n

    @torch.no_grad()
    def trace_probe(self, n_tasks: int = 128) -> float:
        """Diagnostic only: how well the latent positions predict the gold trace.

        This is never used at inference -- it just reports whether the latent block
        actually carries the intermediate results it was supervised on.
        """
        self.reasoner.eval()
        tasks = self._eval_set(n_tasks)
        prompt, aids, aab, apad, _, _, tr_ids, tr_mask = self._collate(tasks)
        _, latent_logits, _ = self.reasoner(prompt, aids, aab, apad)
        correct = (latent_logits.argmax(dim=-1) == tr_ids).float() * tr_mask
        return float(correct.sum() / tr_mask.sum().clamp_min(1.0))

    def train(self) -> Dict[str, float]:
        c = self.cfg
        print(f"[lotus] {device_report(self.device, self.amp)} backend={self.verifier.backend}")
        print(f"[lotus] task=depth{c.depth}/digits{c.digits}/ops{c.ops_key} "
              f"latents={c.n_latent} loops={c.loops} trace_coef={c.trace_coef} "
              f"params={self.reasoner.model.num_params()}")
        run = {"loss": 0.0, "ans": 0.0, "trace": 0.0}
        final: Dict[str, float] = {}
        for step in range(c.steps):
            m = self._train_step(step)
            for k in run:
                run[k] += m[k]
            if (step + 1) % c.log_every == 0:
                print(f"  step {step+1:5d}/{c.steps}  loss {run['loss']/c.log_every:.4f} "
                      f"(ans {run['ans']/c.log_every:.4f} trace {run['trace']/c.log_every:.4f})  "
                      f"lr {self._lr_at(step):.2e}")
                run = {k: 0.0 for k in run}
            if (step + 1) % c.eval_every == 0 or step == c.steps - 1:
                acc, probe = self.accuracy(), self.trace_probe()
                final = {"acc": acc, "trace_probe": probe}
                print(f"  [eval @ {step+1}] answer acc {acc:.3f}   "
                      f"latent trace-probe {probe:.3f} (diagnostic; never decoded)")
        return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LAMb Stage A (restructured): LOTUS-style parallel supervised latent reasoning")
    p.add_argument("--steps", type=int, default=LotusConfig.steps)
    p.add_argument("--batch-size", type=int, default=LotusConfig.batch_size)
    p.add_argument("--depth", type=int, default=LotusConfig.depth)
    p.add_argument("--digits", type=int, default=LotusConfig.digits)
    p.add_argument("--ops-key", type=int, default=LotusConfig.ops_key)
    p.add_argument("--n-latent", type=int, default=LotusConfig.n_latent)
    p.add_argument("--loops", type=int, default=LotusConfig.loops)
    p.add_argument("--trace-coef", type=float, default=LotusConfig.trace_coef,
                   help="weight on per-latent-position supervision (0 = answer-only ablation)")
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--recurrent-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=LotusConfig.seed)
    add_hardware_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device, amp = resolve_hardware(args)
    cfg = LotusConfig(steps=args.steps, batch_size=args.batch_size, depth=args.depth,
                      digits=args.digits, ops_key=args.ops_key, n_latent=args.n_latent,
                      loops=args.loops, trace_coef=args.trace_coef, seed=args.seed,
                      device=device, amp=amp)
    tok = ArithmeticTokenizer()
    mcfg = ModelConfig(d_model=args.d_model, n_heads=4, d_ff=2 * args.d_model,
                       n_prelude=1, n_recurrent=1, n_coda=1, recurrent_steps=args.recurrent_steps)
    LotusTrainer(cfg, tok, mcfg).train()


if __name__ == "__main__":
    main()
