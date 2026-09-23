"""Stage A -- Coconut single-agent continuous-thought self-training.

LAMb reasons in latent space two ways. The depth-recurrent core already thinks
*vertically* (iterate a block ``T`` times at fixed positions). Coconut adds
*horizontal* latent thinking: insert ``K`` scratchpad positions between the
prompt and the answer whose input embedding is the model's own last hidden state,
fed straight back in continuous space and never decoded. Those thoughts become
part of the attention context, a working memory the answer can read.

The catch, from the literature: Coconut's published gains come from a curriculum
that distils the thoughts from *language* chain-of-thought steps. LAMb has no
language traces -- only a verifiable final answer -- and Coconut's own
``w/o curriculum`` ablation (feed the state back, supervise only the answer) is
*this* setting; in the language domain it underperforms even a no-thoughts
baseline. The follow-up diagnosis (arXiv:2510.12167) shows why: answer-only latent
supervision yields geometrically homogeneous thoughts, and -- their fatal gap --
they had no reliable way to *select* a good latent trajectory.

Two findings here, both honest:

  * **Training.** Plain answer-NLL back-propagated through the thought chain, with
    the thought count randomised, *does* make the scratchpad useful -- greedy
    accuracy rises with the number of thoughts. The number-native substrate and
    genuinely multi-step tasks are why this works without a language curriculum,
    unlike the language-domain ablation. (Hard STaR-filtering to the verifier-
    solved subset was tried and rejected: it starves the tiny model and collapses
    the thoughts to a constant. Since the grammar already supplies exact labels,
    teacher forcing already carries the verifier's information.)
  * **Inference.** LAMb has what arXiv:2510.12167 lacked -- an **exact verifier**.
    Perturb the latent phase with dropout to draw diverse trajectories, decode
    each, and keep any the verifier accepts (best-of-N). Their monotone Pass@N
    becomes *realised* accuracy, a test-time self-improvement loop (search the
    latent space, verify, keep the winner) needing no labels at deploy time.

Nothing is verbalised -- the verifier only checks the emitted number; the thoughts
stay a blackbox. ``collapse_metric`` tracks the 2510.12167 homogeneity signal.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import CoconutConfig, ModelConfig
from .device import Amp, add_hardware_args, device_report, resolve_device, resolve_hardware
from .model.lamb import build_model
from .selfplay.grammar import Descriptor, TaskGrammar
from .selfplay.verifier import Verifier
from .tokenizer import ArithmeticTokenizer

Pair = Tuple[str, str]


def coconut_collate(
    pairs: List[Pair], tok: ArithmeticTokenizer, device: str = "cpu"
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad ``[BOS] expr =`` prompts and right-pad answer content tokens.

    Left-padding fixes the prompt boundary at a common column for every row, so
    the "last thought predicts the first answer token" position aligns across the
    batch. Returns ``(prompt, answer_ids, answer_abacus, answer_pad, targets,
    target_mask)`` where ``targets = [answer_content, EOS]`` and ``target_mask``
    covers the real answer tokens plus EOS.
    """
    prompts = [tok.encode_prompt(p) for p, _ in pairs]
    width = max(len(p) for p in prompts)
    b = len(prompts)

    ids = torch.full((b, width), tok.PAD, dtype=torch.long, device=device)
    ab = torch.zeros((b, width), dtype=torch.long, device=device)
    val = torch.zeros((b, width), dtype=torch.float32, device=device)
    vm = torch.zeros((b, width), dtype=torch.float32, device=device)
    pad = torch.ones((b, width), dtype=torch.bool, device=device)
    for i, e in enumerate(prompts):
        off = width - len(e)
        ids[i, off:] = torch.tensor(e.ids, device=device)
        ab[i, off:] = torch.tensor(e.abacus, device=device)
        val[i, off:] = torch.tensor(e.value, device=device)
        vm[i, off:] = torch.tensor(e.value_mask, device=device)
        pad[i, off:] = False
    prompt = dict(input_ids=ids, abacus_ids=ab, value=val, value_mask=vm, pad_mask=pad)

    # Answer content tokens (strictly between '=' and EOS), with Abacus positions.
    content: List[List[int]] = []
    cabs: List[List[int]] = []
    for _, a in pairs:
        e = tok.encode("0", a)  # any dummy problem; we take only the answer span
        content.append(e.ids[e.ans_start:-1])   # drop trailing EOS
        cabs.append(e.abacus[e.ans_start:-1])
    amax = max(1, max(len(c) for c in content))

    aids = torch.full((b, amax), tok.PAD, dtype=torch.long, device=device)
    aab = torch.zeros((b, amax), dtype=torch.long, device=device)
    apad = torch.ones((b, amax), dtype=torch.bool, device=device)
    targets = torch.full((b, amax + 1), tok.PAD, dtype=torch.long, device=device)
    tmask = torch.zeros((b, amax + 1), dtype=torch.float32, device=device)
    for i, (c, cab) in enumerate(zip(content, cabs)):
        L = len(c)
        if L:
            aids[i, :L] = torch.tensor(c, device=device)
            aab[i, :L] = torch.tensor(cab, device=device)
            apad[i, :L] = False
            targets[i, :L] = torch.tensor(c, device=device)
        targets[i, L] = tok.EOS
        tmask[i, : L + 1] = 1.0
    return prompt, aids, aab, apad, targets, tmask


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, t, v = logits.shape
    ce = F.cross_entropy(logits.reshape(b * t, v), targets.reshape(b * t), reduction="none").view(b, t)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


class CoconutTrainer:
    def __init__(self, cfg: CoconutConfig, tokenizer: ArithmeticTokenizer, model_cfg: Optional[ModelConfig] = None):
        self.cfg = cfg
        self.tok = tokenizer
        self.device = resolve_device(cfg.device)
        self.amp = Amp(self.device, cfg.amp)
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        mcfg = model_cfg or ModelConfig(d_model=96, n_heads=4, d_ff=192,
                                        n_prelude=1, n_recurrent=1, n_coda=1,
                                        recurrent_steps=4)
        self.model = build_model(mcfg, tokenizer).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.grammar = TaskGrammar()
        self.verifier = Verifier()
        self.descriptor = Descriptor(depth=cfg.depth, digits=cfg.digits, ops_key=cfg.ops_key)
        self.max_ans = cfg.max_answer_len()
        self._seed = cfg.seed * 1_000_003 + 1

    # -- data -------------------------------------------------------------
    def _sample_batch(self, n: int) -> List[Pair]:
        out: List[Pair] = []
        for _ in range(n):
            self._seed += 1
            out.append(self.grammar.sample(self.descriptor, self._seed))
        return out

    def _train_thoughts(self) -> int:
        c = self.cfg
        if c.sample_train_thoughts:
            return random.randint(c.train_min_thoughts, c.train_max_thoughts)
        return c.n_thoughts

    def _lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup:
            return c.lr * (step + 1) / max(1, c.warmup)
        prog = (step - c.warmup) / max(1, c.steps - c.warmup)
        return 0.5 * c.lr * (1.0 + torch.cos(torch.tensor(prog * 3.141592653589793)).item())

    # -- verifier-selected best-of-N --------------------------------------
    @torch.no_grad()
    def _solve_bestof(self, pairs: List[Pair], k: int, n: int) -> List[bool]:
        """True per problem iff any of ``n`` dropout trajectories verifiably solves it."""
        if n <= 1:
            ans = self.model.coconut_solve([p for p, _ in pairs], self.tok, k,
                                           self.max_ans, device=self.device, thought_dropout=0.0)
            return [self.verifier.check(p, a) for (p, _), a in zip(pairs, ans)]
        big = [p for p, _ in pairs for _ in range(n)]
        ans = self.model.coconut_solve(big, self.tok, k, self.max_ans, device=self.device,
                                       thought_dropout=self.cfg.thought_dropout)
        solved: List[bool] = []
        for i, (p, _) in enumerate(pairs):
            group = ans[i * n:(i + 1) * n]
            solved.append(any(self.verifier.check(p, a) for a in group))
        return solved

    # -- training ---------------------------------------------------------
    def _train_step(self, step: int) -> Dict[str, float]:
        """One step of answer-NLL back-propagated through a randomly sized thought
        chain. (Hard STaR-filtering to the verifier-solved subset was tried and
        rejected: on a tiny model it starves the gradient and collapses the
        thoughts to a constant, and -- since the grammar already supplies exact
        labels -- teacher forcing already uses the verifier's information. The
        verifier's unique leverage is at inference, via best-of-N.)"""
        c = self.cfg
        pairs = self._sample_batch(c.batch_size)
        k = self._train_thoughts()

        self.model.train()
        prompt, aids, aab, apad, targets, tmask = coconut_collate(pairs, self.tok, self.device)
        for g in self.opt.param_groups:
            g["lr"] = self._lr_at(step)
        with self.amp.autocast():
            logits, _ = self.model.coconut_logits(prompt, aids, aab, apad, k)
            loss = _masked_ce(logits, targets, tmask)
        self.amp.backward_step(loss, self.opt, self.model.parameters(), c.grad_clip)
        return {"loss": float(loss.detach()), "K": float(k)}

    # -- evaluation -------------------------------------------------------
    @torch.no_grad()
    def greedy_accuracy(self, k: int, n_tasks: Optional[int] = None) -> float:
        n = n_tasks or self.cfg.eval_tasks
        pairs = self._eval_set(n)
        ans = self.model.coconut_solve([p for p, _ in pairs], self.tok, k,
                                       self.max_ans, device=self.device)
        return sum(self.verifier.check(p, a) for (p, _), a in zip(pairs, ans)) / n

    @torch.no_grad()
    def bestof_accuracy(self, n: int, k: Optional[int] = None, n_tasks: Optional[int] = None) -> float:
        k = self.cfg.n_thoughts if k is None else k
        m = n_tasks or self.cfg.eval_tasks
        pairs = self._eval_set(m)
        return sum(self._solve_bestof(pairs, k, n)) / m

    @torch.no_grad()
    def collapse_metric(self, k: int, n_tasks: int = 128) -> Tuple[float, float]:
        """Diagnose latent collapse (the 2510.12167 failure signal).

        Returns ``(mean_pairwise_cosine, mean_slot_std)`` of the ``K`` thought
        vectors across a batch of problems. High cosine + low std => thoughts have
        collapsed to a homogeneous cluster (no problem-dependent information).
        """
        if k <= 0:
            return float("nan"), float("nan")
        pairs = self._eval_set(n_tasks)
        prompt, *_ = coconut_collate(pairs, self.tok, self.device)
        self.model.eval()
        x = self.model.embed(prompt["input_ids"], prompt["abacus_ids"], prompt["value"], prompt["value_mask"])
        x, _ = self.model._roll_thoughts(x, prompt["pad_mask"], k, thought_dropout=0.0)
        thoughts = x[:, -k:, :]  # (B, K, d)
        cos_sum, std_sum = 0.0, 0.0
        for s in range(k):
            t = thoughts[:, s, :]                       # (B, d)
            tn = F.normalize(t, dim=-1)
            sim = tn @ tn.t()                           # (B, B)
            off = (sim.sum() - sim.diag().sum()) / (t.size(0) * (t.size(0) - 1))
            cos_sum += float(off)
            std_sum += float(t.std(dim=0).mean())
        return cos_sum / k, std_sum / k

    def _eval_set(self, n: int) -> List[Pair]:
        # Deterministic held-out set (seed range disjoint from training's running seed).
        rng = random.Random(self.cfg.seed * 7 + 12345)
        return [self.grammar.sample(self.descriptor, rng.randint(0, 2 ** 31 - 1)) for _ in range(n)]

    # -- driver -----------------------------------------------------------
    def train(self) -> None:
        c = self.cfg
        print(f"[coconut] {device_report(self.device, self.amp)} backend={self.verifier.backend}")
        print(f"[coconut] task=depth{c.depth}/digits{c.digits}/ops{c.ops_key} "
              f"thoughts~[{c.train_min_thoughts},{c.train_max_thoughts}] "
              f"bestof_dropout={c.thought_dropout} params={self.model.num_params()}")
        run_loss = 0.0
        for step in range(c.steps):
            run_loss += self._train_step(step)["loss"]
            if (step + 1) % c.log_every == 0:
                print(f"  step {step+1:5d}/{c.steps}  loss {run_loss / c.log_every:.4f}  "
                      f"lr {self._lr_at(step):.2e}")
                run_loss = 0.0
            if (step + 1) % c.eval_every == 0 or step == c.steps - 1:
                self._report(step + 1)

    def _report(self, step: int) -> None:
        c = self.cfg
        greedy = {k: self.greedy_accuracy(k) for k in c.eval_thoughts}
        gstr = "  ".join(f"K={k}:{a:.3f}" for k, a in greedy.items())
        print(f"  [eval @ {step}] greedy acc vs #thoughts   {gstr}")
        best = {n: self.bestof_accuracy(n) for n in c.eval_bestof}
        bstr = "  ".join(f"N={n}:{a:.3f}" for n, a in best.items())
        print(f"  [eval @ {step}] verifier best-of-N (K={c.n_thoughts})  {bstr}")
        cos, std = self.collapse_metric(c.n_thoughts)
        print(f"  [eval @ {step}] latent diag (K={c.n_thoughts})  cos={cos:.3f} std={std:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LAMb Stage A: Coconut continuous-thought reasoning")
    p.add_argument("--steps", type=int, default=CoconutConfig.steps)
    p.add_argument("--batch-size", type=int, default=CoconutConfig.batch_size)
    p.add_argument("--depth", type=int, default=CoconutConfig.depth)
    p.add_argument("--digits", type=int, default=CoconutConfig.digits)
    p.add_argument("--ops-key", type=int, default=CoconutConfig.ops_key)
    p.add_argument("--n-thoughts", type=int, default=CoconutConfig.n_thoughts)
    p.add_argument("--train-max-thoughts", type=int, default=CoconutConfig.train_max_thoughts)
    p.add_argument("--thought-dropout", type=float, default=CoconutConfig.thought_dropout,
                   help="latent-phase dropout for best-of-N trajectory diversity")
    p.add_argument("--seed", type=int, default=CoconutConfig.seed)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--recurrent-steps", type=int, default=4)
    add_hardware_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device, amp = resolve_hardware(args)
    cfg = CoconutConfig(
        steps=args.steps, batch_size=args.batch_size, depth=args.depth,
        digits=args.digits, ops_key=args.ops_key, n_thoughts=args.n_thoughts,
        train_max_thoughts=args.train_max_thoughts,
        thought_dropout=args.thought_dropout, seed=args.seed, device=device, amp=amp,
    )
    tok = ArithmeticTokenizer()
    mcfg = ModelConfig(d_model=args.d_model, n_heads=4, d_ff=2 * args.d_model,
                       n_prelude=1, n_recurrent=1, n_coda=1, recurrent_steps=args.recurrent_steps)
    CoconutTrainer(cfg, tok, mcfg).train()


if __name__ == "__main__":
    main()
