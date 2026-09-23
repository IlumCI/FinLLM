"""Device / mixed-precision foundation (the GPU+CPU+RAM hybrid).

These run on any machine: on a CPU-only box the device resolves to ``cpu`` and the
mixed-precision paths are exercised by forcing them on (CPU bf16 autocast), so the
GPU code path is covered without a GPU present.
"""

from __future__ import annotations

import os

import torch

from lamb import ArithmeticTokenizer, CoconutConfig
from lamb.coconut import CoconutTrainer
from lamb.config import ModelConfig
from lamb.device import (Amp, configure_threads, device_report, resolve_device,
                         scale_preset)
from lamb.model.lamb import build_model


def test_resolve_device_honours_explicit_and_env():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:3") == "cuda:3"      # explicit is returned as-is
    assert resolve_device("auto") in ("cpu", "cuda", "mps")
    old = os.environ.get("LAMB_DEVICE")
    try:
        os.environ["LAMB_DEVICE"] = "cpu"
        assert resolve_device("auto") == "cpu"       # env overrides
    finally:
        if old is None:
            os.environ.pop("LAMB_DEVICE", None)
        else:
            os.environ["LAMB_DEVICE"] = old


def test_amp_default_off_on_cpu_no_regression():
    amp = Amp("cpu")                       # default: disabled on CPU
    assert amp.enabled is False
    import contextlib
    assert isinstance(amp.autocast(), contextlib.nullcontext)


def test_amp_forced_cpu_bf16_autocast_and_step():
    amp = Amp("cpu", enabled=True)
    assert amp.dtype is torch.bfloat16
    lin = torch.nn.Linear(16, 16)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    before = lin.weight.detach().clone()
    with amp.autocast():
        out = lin(torch.randn(8, 16))
        assert out.dtype is torch.bfloat16     # autocast lowered the matmul
        loss = out.square().mean()
    amp.backward_step(loss, opt, lin.parameters(), grad_clip=1.0)
    assert lin.weight.grad is not None
    assert not torch.equal(before, lin.weight.detach())  # a step was taken


def test_scale_presets_grow_monotonically():
    tok = ArithmeticTokenizer()
    sizes = {}
    for name in ("tiny", "small"):
        pre = scale_preset(name)
        m = build_model(ModelConfig(d_model=pre["d_model"], n_heads=pre["n_heads"],
                                    d_ff=2 * pre["d_model"], recurrent_steps=pre["recurrent_steps"]), tok)
        sizes[name] = m.num_params()
    assert sizes["small"] > sizes["tiny"]           # a bigger preset is a bigger model
    import pytest
    with pytest.raises(ValueError):
        scale_preset("gigantic")


def test_device_report_and_threads():
    assert configure_threads() >= 1
    rep = device_report("cpu")
    assert "device=cpu" in rep and "torch=" in rep


def test_trainer_trains_under_forced_amp():
    tok = ArithmeticTokenizer()
    cfg = CoconutConfig(steps=20, batch_size=16, device="cpu", amp=True, eval_every=10_000,
                        log_every=10_000, eval_tasks=16)
    mcfg = ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3)
    tr = CoconutTrainer(cfg, tok, mcfg)
    assert tr.amp.enabled is True and tr.device == "cpu"
    first = tr._train_step(0)["loss"]
    last = first
    for s in range(1, 20):
        last = tr._train_step(s)["loss"]
    assert last < first  # learning still works through the autocast path
