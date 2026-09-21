"""Minimal LAMb quickstart.

Builds a tiny model, runs a short self-play burst, then shows the solver
answering fresh problems and the effect of the test-time latent-step budget.

    python examples/quickstart.py
"""

from __future__ import annotations

import torch

from lamb import ArithmeticTokenizer, ModelConfig, TrainConfig, backend
from lamb.eval import evaluate, test_time_scaling
from lamb.model.lamb import build_model
from lamb.selfplay.loop import SelfPlayTrainer


def main() -> None:
    torch.manual_seed(0)
    print(f"native kernels: {backend()}")

    tok = ArithmeticTokenizer()
    model_cfg = ModelConfig(d_model=96, recurrent_steps=3)
    train_cfg = TrainConfig(batch_size=64, max_digits=1, ops=("+", "-"), steps=400)

    model = build_model(model_cfg, tok)
    trainer = SelfPlayTrainer(train_cfg, model, tok)
    print(f"params: {model.num_params():,}")

    for _ in range(train_cfg.steps):
        stats = trainer.train_step()
        if stats.step % 100 == 0:
            print(f"  step {stats.step:4d}  solve {stats.batch_success:.2f}  "
                  f"frontier {stats.mastered_frontier}  H(prop) {stats.proposer_entropy:.2f}")

    print("\nsolver on fresh problems:")
    for problem in ["3+4", "9+8", "7-2", "5-9"]:
        pred = model.solve([problem], tok, max_answer_len=4)[0]
        print(f"  {problem} = {pred}")

    res = evaluate(model, tok, trainer.grid, n_per_cell=48, max_answer_len=4)
    print(f"\nheld-out exact-match accuracy: {res['overall']:.3f}")

    scaling = test_time_scaling(model, tok, trainer.grid, steps_list=[1, 3, 5], n_per_cell=32)
    print("latent-step budget T -> accuracy:", {t: round(a, 3) for t, a in scaling.items()})


if __name__ == "__main__":
    main()
