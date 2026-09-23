"""Does on-policy RL actually move a latent reasoner? The test, not the claim.

arXiv:2512.11816 is the negative result this exists to check: GRPO takes explicit
chain-of-thought from 62.6 -> 72.6 while a latent model goes 22.6 -> 21.8, i.e. RL
does essentially nothing. The diagnosis is that a continuous segment has no
well-defined policy ratio, so there is nothing for an on-policy objective to act
on. SWITCH (arXiv:2606.13106) claims boundary tokens fix this.

Testing that claim honestly requires being precise about *what* RL could shape.
Sampling answer tokens gives a well-defined log-probability with or without
boundaries -- but it only shapes the answer head, never the latent computation.
That is the real reason latent RL is inert. So the boundary has to buy a **sampled
action that changes the latent computation itself**:

    prompt -> [BOT] -> boundary hidden state -> switch head -> latent budget
           -> latent block run at that budget -> answer

``switch_head`` is a categorical over how many latent positions stay active. It is
sampled, it has an exact log-probability, and it changes the latent computation --
so GRPO can credit-assign into latent compute allocation. Budget is applied by
masking latent positions, which keeps the batch rectangular.

Three arms, all from one shared supervised checkpoint so nothing differs but the
treatment:

* ``sft``        -- keep training supervised (the control for "more steps help")
* ``rl_switch``  -- GRPO over (switch action + answer tokens)
* ``rl_noswitch``-- GRPO over answer tokens only, fixed budget (the 2512.11816 setting)

If ``rl_switch`` beats both, boundaries unblock latent RL. If neither RL arm beats
``sft``, latent RL is inert here too and the roadmap changes.
"""

from __future__ import annotations

import argparse
import copy
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coconut import coconut_collate
from .config import LotusConfig, ModelConfig
from .device import add_hardware_args, device_report, resolve_hardware
from .lotus import LotusTrainer
from .selfplay.grpo import dynamic_keep_mask, group_advantages
from .tokenizer import ArithmeticTokenizer

BUDGETS: Tuple[int, ...] = (2, 4, 8)   # active latent positions the switch may choose


def answer_tensors(tok: ArithmeticTokenizer, seqs: List[List[int]], device: str):
    """Build teacher-forcing tensors from *sampled* answer tokens.

    ``seqs`` are rollout token lists (possibly ending in EOS). Input tokens are the
    content; targets are content + EOS, so the scored span matches the supervised
    path exactly and the log-probs correspond to what was actually sampled.
    """
    content = []
    for s in seqs:
        c = [t for t in s if t != tok.PAD]
        if c and c[-1] == tok.EOS:
            c = c[:-1]
        content.append(c)
    amax = max(1, max(len(c) for c in content))
    b = len(content)
    aids = torch.full((b, amax), tok.PAD, dtype=torch.long, device=device)
    aab = torch.zeros((b, amax), dtype=torch.long, device=device)
    apad = torch.ones((b, amax), dtype=torch.bool, device=device)
    targets = torch.full((b, amax + 1), tok.PAD, dtype=torch.long, device=device)
    tmask = torch.zeros((b, amax + 1), dtype=torch.float32, device=device)
    for i, c in enumerate(content):
        prev_id, prev_ab = tok.EQ, 0
        for j, t in enumerate(c):
            ab = tok.abacus_after(prev_id, prev_ab, t)
            aids[i, j] = t
            aab[i, j] = ab
            apad[i, j] = False
            targets[i, j] = t
            prev_id, prev_ab = t, ab
        targets[i, len(c)] = tok.EOS
        tmask[i, : len(c) + 1] = 1.0
    return aids, aab, apad, targets, tmask


class LatentPolicy(nn.Module):
    """A LOTUS reasoner plus a switch head that allocates latent compute."""

    def __init__(self, reasoner, budgets: Tuple[int, ...] = BUDGETS, use_switch: bool = True):
        super().__init__()
        self.r = reasoner
        self.budgets = tuple(budgets)
        if use_switch and reasoner.bot_id is None:
            # Silently dropping the switch would turn the treatment arm into the
            # control and quietly invalidate the whole comparison.
            raise ValueError(
                "use_switch=True needs a boundary: build the reasoner with "
                "LotusConfig(use_boundaries=True)")
        self.use_switch = bool(use_switch)
        d = reasoner.model.cfg.d_model
        self.switch_head = nn.Linear(d, len(self.budgets)) if self.use_switch else None

    # -- pieces -----------------------------------------------------------
    def _prompt_x(self, prompt):
        m = self.r.model
        return m.embed(prompt["input_ids"], prompt["abacus_ids"],
                       prompt["value"], prompt["value_mask"])

    def switch_logits(self, prompt) -> Optional[torch.Tensor]:
        """Boundary hidden state -> categorical over latent budgets (B, n_budgets)."""
        if not self.use_switch:
            return None
        m = self.r.model
        x_p = self._prompt_x(prompt)
        b = x_p.size(0)
        bot = self.r._marker(self.r.bot_id, b, x_p.device)
        x = torch.cat([x_p, bot], dim=1)
        pad = torch.cat(
            [prompt["pad_mask"], torch.zeros(b, 1, dtype=torch.bool, device=x_p.device)], dim=1)
        h, _ = m.core(x, pad)
        return self.switch_head(h[:, -1, :])

    def _latent_prefix(self, prompt, budget_idx: Optional[torch.Tensor]):
        """Latent block run at a per-example budget. Returns (x, pad) after the loops."""
        m = self.r.model
        x_p = self._prompt_x(prompt)
        b, p, _ = x_p.shape
        dev = x_p.device
        L = self.r.n_latent
        head = [x_p]
        if self.r.bot_id is not None:
            head.append(self.r._marker(self.r.bot_id, b, dev))
        head_x = torch.cat(head, dim=1)
        l0 = head_x.size(1)

        idx = torch.arange(L, device=dev)
        lat = self.r.latent_emb(idx).unsqueeze(0).expand(b, -1, -1)
        if budget_idx is None:
            lat_pad = torch.zeros(b, L, dtype=torch.bool, device=dev)
        else:
            bud = torch.tensor(self.budgets, device=dev)[budget_idx]     # (B,)
            lat_pad = idx.unsqueeze(0) >= bud.unsqueeze(1)               # True = inactive
        pad = torch.cat([prompt["pad_mask"],
                         torch.zeros(b, l0 - p, dtype=torch.bool, device=dev),
                         lat_pad], dim=1)
        x = torch.cat([head_x, lat], dim=1)
        for _ in range(self.r.loops):
            h, _ = m.core(x, pad)
            lat = self.r.latent_norm(h[:, l0:l0 + L, :]) + self.r.latent_marker
            x = torch.cat([head_x, lat], dim=1)
        return x, pad

    def forward_logits(self, prompt, aids, aab, apad, budget_idx):
        """Teacher-forced answer logits at a replayed budget (B, A+1, V)."""
        m = self.r.model
        x, pad = self._latent_prefix(prompt, budget_idx)
        z = torch.zeros_like(aab, dtype=torch.float32)
        x = torch.cat([x, m.embed(aids, aab, z, z)], dim=1)
        pad = torch.cat([pad, apad], dim=1)
        h, _ = m.core(x, pad)
        return m._readout(h)[:, -(aids.size(1) + 1):, :]

    @torch.no_grad()
    def rollout(self, prompt, tok, max_len: int, temperature: float = 1.0,
                greedy: bool = False, budget_idx: Optional[torch.Tensor] = None):
        """Sample (or greedily decode) answers at the given latent budget."""
        m = self.r.model
        x, pad = self._latent_prefix(prompt, budget_idx)
        b = x.size(0)
        dev = x.device
        last_id = torch.full((b,), tok.EQ, dtype=torch.long, device=dev)
        last_ab = torch.zeros((b,), dtype=torch.long, device=dev)
        done = torch.zeros(b, dtype=torch.bool, device=dev)
        cols: List[torch.Tensor] = []
        for _ in range(max_len):
            h, _ = m.core(x, pad)
            logits = m._readout(h)[:, -1, :]
            if greedy:
                nxt = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, tok.PAD), nxt)
            new_ab = torch.zeros_like(nxt)
            for i in range(b):
                new_ab[i] = tok.abacus_after(int(last_id[i]), int(last_ab[i]), int(nxt[i]))
            z = torch.zeros((b, 1), dtype=torch.float32, device=dev)
            x = torch.cat([x, m.embed(nxt[:, None], new_ab[:, None], z, z)], dim=1)
            pad = torch.cat([pad, done.unsqueeze(1)], dim=1)
            cols.append(nxt)
            last_id, last_ab = nxt, new_ab
            done = done | (nxt == tok.EOS)
            if bool(done.all()):
                break
        gen = torch.stack(cols, dim=1) if cols else torch.zeros((b, 0), dtype=torch.long)
        out: List[List[int]] = []
        for row in gen.tolist():
            seq: List[int] = []
            for t in row:
                if t == tok.PAD:
                    break
                seq.append(t)
                if t == tok.EOS:
                    break
            out.append(seq)
        return out

    @torch.no_grad()
    def greedy_budget(self, prompt) -> Optional[torch.Tensor]:
        sl = self.switch_logits(prompt)
        return None if sl is None else sl.argmax(dim=-1)


# -- GRPO over (switch action + answer tokens) ---------------------------------
def grpo_step(policy: LatentPolicy, ref: LatentPolicy, tok: ArithmeticTokenizer, verifier,
              problems: List[str], opt, group_size: int, temperature: float,
              kl_coef: float, max_len: int, device: str, grad_clip: float = 1.0,
              entropy_coef: float = 0.0):
    """One GRPO update. Returns metrics; the policy is updated in place."""
    rep = [p for p in problems for _ in range(group_size)]
    group_ids = torch.tensor([i for i in range(len(problems)) for _ in range(group_size)],
                             device=device)
    prompt, *_ = coconut_collate([(p, "0") for p in rep], tok, device)

    # Sample the latent-budget action (the part answer-only RL cannot reach).
    modes = None
    if policy.use_switch:
        with torch.no_grad():
            sl = policy.switch_logits(prompt)
            modes = torch.multinomial(torch.softmax(sl, dim=-1), num_samples=1).squeeze(1)

    seqs = policy.rollout(prompt, tok, max_len, temperature=temperature, budget_idx=modes)
    preds = [tok.decode_answer(s) for s in seqs]
    rewards = torch.tensor([1.0 if verifier.check(p, pr) else 0.0 for p, pr in zip(rep, preds)],
                           device=device)

    metrics: Dict[str, float] = {"reward": float(rewards.mean())}
    if modes is not None:
        for j in range(len(policy.budgets)):
            metrics[f"mode{policy.budgets[j]}"] = float((modes == j).float().mean())

    keep = dynamic_keep_mask(rewards, group_ids)      # DAPO: drop zero-variance groups
    metrics["kept"] = float(keep.float().mean())
    if int(keep.sum()) == 0:
        return metrics

    adv = group_advantages(rewards, group_ids)
    aids, aab, apad, targets, tmask = answer_tensors(tok, seqs, device)

    policy.train()
    ans_logits = policy.forward_logits(prompt, aids, aab, apad, modes)
    logp = F.log_softmax(ans_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    seq_logp = (logp * tmask).sum(dim=1)
    if policy.use_switch:
        sl = policy.switch_logits(prompt)             # with grad, replaying the sampled mode
        seq_logp = seq_logp + F.log_softmax(sl, dim=-1).gather(1, modes.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        ref_logits = ref.forward_logits(prompt, aids, aab, apad, modes)
        ref_logp = F.log_softmax(ref_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    delta = ref_logp - logp
    seq_kl = ((torch.exp(delta) - delta - 1.0) * tmask).sum(dim=1)   # k3, >= 0

    loss = (-(adv.detach() * seq_logp) + kl_coef * seq_kl)[keep].mean()
    if policy.use_switch and entropy_coef > 0:
        # The switch collapsed to a single budget within ~100 steps without this,
        # which turns the latent-affecting action back into a constant.
        logq = F.log_softmax(sl, dim=-1)
        ent = -(logq.exp() * logq).sum(dim=-1).mean()
        metrics["switch_entropy"] = float(ent.detach())
        loss = loss - entropy_coef * ent
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    opt.step()
    metrics["loss"] = float(loss.detach())
    metrics["kl"] = float(seq_kl[keep].mean().detach())
    return metrics


@torch.no_grad()
def evaluate(policy: LatentPolicy, tasks, tok, verifier, max_len: int, device: str) -> float:
    """Greedy exact-match, each arm evaluated exactly as it operates."""
    policy.eval()
    prompt, *_ = coconut_collate([(e, "0") for e, _, _ in tasks], tok, device)
    modes = policy.greedy_budget(prompt) if policy.use_switch else None
    seqs = policy.rollout(prompt, tok, max_len, greedy=True, budget_idx=modes)
    preds = [tok.decode_answer(s) for s in seqs]
    return sum(verifier.check(e, p) for (e, _, _), p in zip(tasks, preds)) / len(tasks)


def run_experiment(cfg: LotusConfig, tok: ArithmeticTokenizer, mcfg: ModelConfig,
                   sft_steps: int, rl_steps: int, rl_lr: float, group_size: int,
                   rl_problems: int, temperature: float, kl_coef: float,
                   eval_tasks: int, entropy_coef: float = 0.0,
                   arms: Tuple[str, ...] = ("sft", "rl_switch", "rl_noswitch")) -> Dict[str, float]:
    """One shared SFT checkpoint, three arms, same evaluation."""
    print(f"[latent-rl] {device_report(cfg.device)}")
    print(f"[latent-rl] shared SFT {sft_steps} steps, then {rl_steps} steps per arm "
          f"(group={group_size} problems={rl_problems} budgets={BUDGETS})")

    base = LotusTrainer(cfg, tok, mcfg)
    for s in range(sft_steps):
        base._train_step(s)
    snapshot = copy.deepcopy(base.reasoner.state_dict())
    held = base._eval_set(eval_tasks)
    device, max_len, verifier = base.device, base.max_ans, base.verifier

    start_policy = LatentPolicy(base.reasoner, use_switch=False).to(device)
    acc0 = evaluate(start_policy, held, tok, verifier, max_len, device)
    print(f"  [start] after SFT: acc {acc0:.3f}")
    results = {"sft_start": acc0}

    def fresh_policy(use_switch: bool) -> LatentPolicy:
        trainer = LotusTrainer(cfg, tok, mcfg)
        trainer.reasoner.load_state_dict(snapshot)
        return LatentPolicy(trainer.reasoner, use_switch=use_switch).to(device)

    # Arm 1: keep training supervised (controls for "more steps help").
    if "sft" in arms:
        sft_arm = LotusTrainer(cfg, tok, mcfg)
        sft_arm.reasoner.load_state_dict(snapshot)
        for s in range(sft_steps, sft_steps + rl_steps):
            sft_arm._train_step(s)
        results["sft_continued"] = evaluate(
            LatentPolicy(sft_arm.reasoner, use_switch=False).to(device),
            held, tok, verifier, max_len, device)
        print(f"  [arm sft]         continued supervised: acc {results['sft_continued']:.3f}")

    # Arms 2 and 3: GRPO, with and without the switch action.
    rng = random.Random(cfg.seed + 7)
    for name, use_switch in (("rl_switch", True), ("rl_noswitch", False)):
        if name not in arms:
            continue
        policy = fresh_policy(use_switch)
        ref = fresh_policy(use_switch)
        for p in ref.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(policy.parameters(), lr=rl_lr, weight_decay=0.0)
        last: Dict[str, float] = {}
        for s in range(rl_steps):
            # Train the policy on the training partition only. Drawing from the
            # whole space put the RL arms' own evaluation problems in their
            # rollouts, which biases every arm upward -- harmlessly for the
            # negative result this experiment reported, but it is still wrong.
            tasks = [base.grammar.sample_with_trace(base.descriptor,
                                                    rng.randint(0, 2 ** 31 - 1),
                                                    exclude_heldout=True)
                     for _ in range(rl_problems)]
            last = grpo_step(policy, ref, tok, verifier, [e for e, _, _ in tasks], opt,
                             group_size, temperature, kl_coef, max_len, device,
                             entropy_coef=entropy_coef)
            if (s + 1) % 100 == 0:
                extra = " ".join(f"{k}={v:.2f}" for k, v in last.items()
                                 if k.startswith("mode") or k == "switch_entropy")
                print(f"    {name} step {s+1}/{rl_steps} reward {last.get('reward', 0):.3f} "
                      f"kept {last.get('kept', 0):.2f} {extra}")
        results[name] = evaluate(policy, held, tok, verifier, max_len, device)
        print(f"  [arm {name}] acc {results[name]:.3f}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Does on-policy RL move a latent reasoner?")
    p.add_argument("--sft-steps", type=int, default=500)
    p.add_argument("--rl-steps", type=int, default=300)
    p.add_argument("--rl-lr", type=float, default=2e-4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--rl-problems", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="entropy bonus on the switch policy (prevents mode collapse)")
    p.add_argument("--arms", type=str, default="sft,rl_switch,rl_noswitch")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--digits", type=int, default=1)
    p.add_argument("--n-latent", type=int, default=8)
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--eval-tasks", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    add_hardware_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device, amp = resolve_hardware(args)
    cfg = LotusConfig(steps=args.sft_steps + args.rl_steps, batch_size=args.batch_size,
                      depth=args.depth, digits=args.digits, n_latent=args.n_latent,
                      loops=args.loops, eval_tasks=args.eval_tasks, seed=args.seed,
                      device=device, amp=amp,
                      use_boundaries=True)   # the switch head attaches to the boundary
    tok = ArithmeticTokenizer()
    mcfg = ModelConfig(d_model=args.d_model, n_heads=4, d_ff=2 * args.d_model,
                       n_prelude=1, n_recurrent=1, n_coda=1, recurrent_steps=4)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    r = run_experiment(cfg, tok, mcfg, args.sft_steps, args.rl_steps, args.rl_lr,
                       args.group_size, args.rl_problems, args.temperature, args.kl_coef,
                       args.eval_tasks, entropy_coef=args.entropy_coef, arms=arms)
    print("\n  === verdict ===")
    print(f"  SFT start            {r['sft_start']:.3f}")
    for k in ("sft_continued", "rl_switch", "rl_noswitch"):
        if k in r:
            print(f"  {k:20s} {r[k] - r['sft_start']:+.3f} -> {r[k]:.3f}")


if __name__ == "__main__":
    main()
