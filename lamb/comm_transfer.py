"""Held-out-partner test for Stage B latent communication.

Stage B trains a speaker and a listener *together*, so a natural worry (the
zero-shot-coordination / Other-Play problem, arXiv:2003.02979) is that the pair
co-adapts to a **private** latent code rather than a shareable protocol. Two
probes settle it:

  1. **Cross-pair swap (zero shot).** Train two independent
     (speaker, channel, listener) pairs from different seeds, each to comm ~ 1.0,
     then feed each listener the *other* pair's speaker. If accuracy collapses to
     the no-information prior, each pair invented its own private code (strong
     co-adaptation); if it holds, the code is canonical and partner-independent.

  2. **Fresh-partner learnability.** Freeze one trained speaker and train a
     brand-new receiver (its own channel + listener) against it. If the new
     partner reaches high accuracy, the speaker's fixed code is a well-formed
     language a new agent can *learn* to read -- even when it is not zero-shot
     compatible. This separates "unshareable" from "merely idiosyncratic".

The honest expectation from the literature is a private-but-learnable code:
zero-shot swap fails, a freshly trained partner succeeds. Partner randomization
(training against many speakers) is the standard fix and the natural next step.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch

from .comm import CommTrainer
from .config import CommConfig
from .model.lamb import LAMb
from .tokenizer import ArithmeticTokenizer


def _train_pair(cfg: CommConfig, tok: ArithmeticTokenizer, seed: int) -> CommTrainer:
    tr = CommTrainer(replace(cfg, seed=seed), tok)
    for step in range(cfg.steps):
        tr._train_step(step)
    return tr


def _train_fresh_receiver(cfg: CommConfig, tok: ArithmeticTokenizer,
                          frozen_speaker: LAMb, seed: int, steps: int) -> CommTrainer:
    """Train a new receiver (channel + listener) against a frozen foreign speaker.

    Only the fresh channel and listener learn; the speaker is held fixed, so this
    measures whether a new partner can *acquire* the speaker's existing code.
    """
    tr = CommTrainer(replace(cfg, seed=seed), tok)
    tr.speaker = frozen_speaker
    for p in tr.speaker.parameters():
        p.requires_grad_(False)
    # Rebuild the optimizer over only the new receiver (exclude the frozen speaker,
    # so decoupled weight decay never touches it).
    params = list(tr.listener.parameters()) + list(tr.channel.parameters())
    tr.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    for step in range(steps):
        tr._train_step(step)
    return tr


def held_out_partner(cfg: CommConfig, tok: ArithmeticTokenizer,
                     seeds: Tuple[int, int] = (0, 1),
                     fresh_steps: Optional[int] = None,
                     n_tasks: Optional[int] = None) -> Dict[str, float]:
    n = n_tasks or cfg.eval_tasks
    a = _train_pair(cfg, tok, seeds[0])
    b = _train_pair(cfg, tok, seeds[1])

    res: Dict[str, float] = {
        # Matched vs swapped share each receiver's own held-out set, so the drop is
        # the pure effect of changing the partner.
        "A_matched": a.accuracy(n_tasks=n),                     # A.listener <- A.speaker
        "A_swapped": a.accuracy(n_tasks=n, speaker=b.speaker),  # A.listener <- B.speaker
        "B_matched": b.accuracy(n_tasks=n),
        "B_swapped": b.accuracy(n_tasks=n, speaker=a.speaker),
        "blank": a.accuracy(n_tasks=n, blank=True),             # no-information prior
    }
    fresh = _train_fresh_receiver(cfg, tok, a.speaker, seeds[0] + 100, fresh_steps or cfg.steps)
    res["fresh_partner"] = fresh.accuracy(n_tasks=n, speaker=a.speaker)
    return res


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LAMb Stage B: held-out-partner (co-adaptation) test")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--fresh-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=CommConfig.batch_size)
    p.add_argument("--a-digits", type=int, default=CommConfig.a_digits)
    p.add_argument("--b-digits", type=int, default=CommConfig.b_digits)
    p.add_argument("--n-msg", type=int, default=CommConfig.n_msg)
    p.add_argument("--d-model", type=int, default=CommConfig.d_model)
    p.add_argument("--seeds", type=int, nargs=2, default=[0, 1])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = CommConfig(steps=args.steps, batch_size=args.batch_size, a_digits=args.a_digits,
                     b_digits=args.b_digits, n_msg=args.n_msg, d_model=args.d_model)
    tok = ArithmeticTokenizer()
    print(f"[comm-transfer] two pairs @ {args.steps} steps (seeds {args.seeds}), "
          f"fresh receiver @ {args.fresh_steps} steps; task X({cfg.a_digits}d) op Y({cfg.b_digits}d)")
    r = held_out_partner(cfg, tok, seeds=tuple(args.seeds), fresh_steps=args.fresh_steps)

    print(f"  matched     A {r['A_matched']:.3f}   B {r['B_matched']:.3f}   "
          f"(each listener with its own trained speaker)")
    print(f"  swapped     A<-B {r['A_swapped']:.3f}   B<-A {r['B_swapped']:.3f}   "
          f"(zero-shot: listener with the other pair's speaker)")
    print(f"  blank prior {r['blank']:.3f}   (no-information ablation)")
    print(f"  fresh partner (new receiver trained vs frozen speaker A) {r['fresh_partner']:.3f}")

    swapped = 0.5 * (r["A_swapped"] + r["B_swapped"])
    matched = 0.5 * (r["A_matched"] + r["B_matched"])
    private = swapped <= r["blank"] + 0.10
    learnable = r["fresh_partner"] >= matched - 0.10
    print(f"  => zero-shot swap {'FAILS (private, co-adapted code)' if private else 'holds (canonical code)'}; "
          f"fresh partner {'LEARNS it (idiosyncratic but shareable)' if learnable else 'struggles'}")


if __name__ == "__main__":
    main()
