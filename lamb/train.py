"""CPU-first end-to-end self-play training for LAMb.

Run ``python -m lamb.train`` (or the ``lamb-train`` console script). A tiny model
teaches itself integer arithmetic from zero data: the proposer invents problems,
the exact verifier scores them, and the solver improves. Watch ``eval_acc`` climb
and ``frontier`` (the mastered operand-digit sum) expand.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict

import torch

from . import __version__
from ._native import backend
from .config import ModelConfig, TrainConfig
from .eval import evaluate, test_time_scaling
from .model.lamb import build_model
from .selfplay.loop import SelfPlayTrainer
from .tokenizer import ArithmeticTokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LAMb by arithmetic self-play (CPU-first).")
    p.add_argument("--steps", type=int, default=TrainConfig.steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--max-digits", type=int, default=TrainConfig.max_digits)
    p.add_argument("--ops", type=str, default="+,-", help="comma-separated subset of + - *")
    p.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    p.add_argument("--recurrent-steps", type=int, default=ModelConfig.recurrent_steps)
    p.add_argument("--use-memory", action="store_true", help="enable test-time neural memory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    p.add_argument("--ckpt-dir", type=str, default=TrainConfig.ckpt_dir)
    p.add_argument("--no-save", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    ops = tuple(o.strip() for o in args.ops.split(",") if o.strip())
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_digits=args.max_digits,
        ops=ops,
        seed=args.seed,
        device=args.device,
        eval_every=args.eval_every,
        log_every=args.log_every,
        ckpt_dir=args.ckpt_dir,
    )
    model_cfg = ModelConfig(
        d_model=args.d_model,
        recurrent_steps=args.recurrent_steps,
        use_memory=args.use_memory,
    )

    tok = ArithmeticTokenizer(max_number_len=model_cfg.max_number_len)
    model = build_model(model_cfg, tok).to(args.device)
    trainer = SelfPlayTrainer(train_cfg, model, tok)

    print(f"LAMb v{__version__} | native kernels: {backend()}")
    print(f"model params: {model.num_params():,} | difficulty cells: {len(trainer.grid)}")
    print(f"ops={ops} max_digits={args.max_digits} recurrent_steps={args.recurrent_steps} "
          f"use_memory={args.use_memory} device={args.device}")
    print("-" * 88)

    eval_budget = 2 * args.max_digits + 2
    start = time.time()
    best = 0.0
    for _ in range(args.steps):
        stats = trainer.train_step()
        if stats.step % args.log_every == 0:
            print(
                f"step {stats.step:5d} | loss {stats.loss:6.3f} | tok_acc {stats.token_acc:5.2f} "
                f"| solve {stats.batch_success:4.2f} | frontier {stats.mastered_frontier:2d} "
                f"| buf {stats.buffer_size:6d} | H(prop) {stats.proposer_entropy:4.2f} "
                f"| top {stats.top_cell}"
            )
        if stats.step % args.eval_every == 0:
            res = evaluate(model, tok, trainer.grid, n_per_cell=12, device=args.device,
                           max_answer_len=eval_budget)
            best = max(best, float(res["overall"]))
            print(f"    [eval] overall exact-match acc: {res['overall']:.3f}  (best {best:.3f})")

    dur = time.time() - start
    print("-" * 88)
    final = evaluate(model, tok, trainer.grid, n_per_cell=32, device=args.device,
                     max_answer_len=eval_budget)
    print(f"final overall exact-match acc: {final['overall']:.3f}  |  {dur:.1f}s")
    hardest = sorted(final["per_cell"].items(), key=lambda kv: (kv[0][1] + kv[0][2]))
    for cell, acc in hardest:
        print(f"    {cell[0]}  {cell[1]}d x {cell[2]}d : {acc:.2f}")

    scaling = test_time_scaling(
        model, tok, trainer.grid, steps_list=sorted({1, model_cfg.recurrent_steps,
                                                     model_cfg.recurrent_steps + 2}),
        n_per_cell=16, device=args.device,
    )
    print("test-time latent-step scaling (T -> acc):",
          " ".join(f"{t}:{a:.2f}" for t, a in scaling.items()))

    if not args.no_save:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        path = os.path.join(args.ckpt_dir, "lamb.pt")
        torch.save(
            {
                "version": __version__,
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
                "state_dict": model.state_dict(),
                "final_acc": final["overall"],
            },
            path,
        )
        print(f"saved checkpoint -> {path}")


if __name__ == "__main__":
    main()
