"""GRPO (Group Relative Policy Optimization) utilities for LAMb.

GRPO is critic-free: instead of a learned value baseline, the advantage of a
sample is its reward normalized *within a group* of samples drawn from the same
state (DeepSeekMath, arXiv:2402.03300). Here it powers two things:

* the **solver** (RLVR): groups are G sampled answers to the same problem, reward
  is the exact verifier's 0/1;
* the **proposer** (:mod:`lamb.selfplay.hyperproposer`): groups are proposed
  tasks from the same solver-competence state, reward is learnability.

Refinements folded in from the 2025-26 literature: **dynamic sampling** (DAPO --
drop zero-variance groups that carry no gradient) and an optional **no-std**
normalization (Dr.GRPO -- avoid difficulty/length bias). A KL penalty to a frozen
reference policy (k3 estimator) keeps updates trust-region-like.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..data import collate
from ..tokenizer import ArithmeticTokenizer


def group_advantages(
    rewards: torch.Tensor, group_ids: torch.Tensor, normalize_std: bool = True
) -> torch.Tensor:
    """Group-relative advantages: within each group, center (and optionally scale)."""
    adv = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        mask = group_ids == g
        r = rewards[mask]
        centered = r - r.mean()
        if normalize_std:
            centered = centered / (r.std(unbiased=False) + 1e-6)
        adv[mask] = centered
    return adv


def dynamic_keep_mask(rewards: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """DAPO dynamic sampling: keep only groups with non-uniform reward (real signal)."""
    keep = torch.zeros_like(rewards, dtype=torch.bool)
    for g in torch.unique(group_ids):
        mask = group_ids == g
        r = rewards[mask]
        if float(r.max()) != float(r.min()):
            keep |= mask
    return keep


def answer_token_logprobs(
    model, batch: Dict[str, torch.Tensor], n_steps: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of the target tokens and the answer mask, shaped (B, T)."""
    logits, _ = model.forward(
        batch["input_ids"], batch["abacus_ids"], batch["value"], batch["value_mask"],
        batch.get("pad_mask"), n_steps,
    )
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, batch["target_ids"].unsqueeze(-1)).squeeze(-1)
    return tok_logp, batch["loss_mask"]


def grpo_solver_loss(
    model,
    ref_model,
    tokenizer: ArithmeticTokenizer,
    verifier,
    problems: List[str],
    group_size: int = 4,
    temperature: float = 1.0,
    kl_coef: float = 0.02,
    normalize_std: bool = True,
    dynamic_sampling: bool = True,
    max_answer_len: int = 12,
    device: str = "cpu",
) -> Tuple[Optional[torch.Tensor], Dict[str, float], List[Tuple[str, str]]]:
    """One GRPO/RLVR objective for the solver.

    Samples ``group_size`` answers per problem, scores them with the exact
    verifier, forms group-relative advantages, and returns the policy-gradient +
    KL loss. Returns ``(loss_or_None, metrics, solved_traces)``; the loss is
    ``None`` when dynamic sampling filters every group (all-correct or all-wrong),
    which the caller treats as "no GRPO update this step".
    """
    rep = [p for p in problems for _ in range(group_size)]
    group_ids = torch.tensor(
        [i for i in range(len(problems)) for _ in range(group_size)], device=device
    )
    answer_ids = model.sample(
        rep, tokenizer, max_answer_len=max_answer_len, device=device, temperature=temperature
    )
    preds = [tokenizer.decode_answer(a) for a in answer_ids]
    rewards = torch.tensor(
        [1.0 if verifier.check(p, pr) else 0.0 for p, pr in zip(rep, preds)],
        device=device,
    )

    solved = [(p, pr) for p, pr, r in zip(rep, preds, rewards) if float(r) > 0 and pr is not None]

    metrics = {
        "grpo_reward": float(rewards.mean()),
        "grpo_solved": float((rewards > 0).float().mean()),
    }

    keep = dynamic_keep_mask(rewards, group_ids) if dynamic_sampling else torch.ones_like(rewards, dtype=torch.bool)
    metrics["grpo_kept"] = float(keep.float().mean())
    if int(keep.sum()) == 0:
        return None, metrics, solved

    examples = [tokenizer.build_from_answer_ids(p, a) for p, a in zip(rep, answer_ids)]
    batch = collate(examples, tokenizer.PAD, device=device)

    advantages = group_advantages(rewards, group_ids, normalize_std=normalize_std)

    model.train()  # sampling put the model in eval; log-probs use the training graph
    tok_logp, mask = answer_token_logprobs(model, batch)
    with torch.no_grad():
        ref_logp, _ = answer_token_logprobs(ref_model, batch)

    seq_logp = (tok_logp * mask).sum(dim=1)
    delta = ref_logp - tok_logp
    kl_tok = torch.exp(delta) - delta - 1.0  # k3 estimator, >= 0
    seq_kl = (kl_tok * mask).sum(dim=1)

    per_seq = -(advantages.detach() * seq_logp) + kl_coef * seq_kl
    loss = per_seq[keep].mean()
    metrics["grpo_kl"] = float(seq_kl[keep].mean().detach())
    return loss, metrics, solved
