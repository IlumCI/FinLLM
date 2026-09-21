"""Long-context memory benchmark: task validity, retrieval, extrapolation."""

from __future__ import annotations

import random

import torch

from lamb.memory_bench import (
    MemorylessModel,
    MemoryRecallModel,
    RecallConfig,
    RecallTask,
    eval_recall,
    train_recall,
)


def test_recall_task_is_well_posed():
    cfg = RecallConfig(n_keys=8, n_vals=8, n_filler=4, d_model=16, d_mem=16)
    task = RecallTask(cfg)
    batch, target = task.batch(bs=32, length=20, n_pairs=3, rng=random.Random(0), front=True)
    # exactly one query at the end, needle at the front, target = needle's value
    assert (batch["kind"][:, -1] == 2).all()
    assert (batch["kind"] == 1).sum(dim=1).eq(3).all()             # n_pairs bindings
    assert torch.equal(batch["key_ids"][:, -1], batch["key_ids"][:, 0])  # query the needle key
    assert torch.equal(target, batch["val_ids"][:, 0])             # answer is its value


def test_memory_retrieves_and_extrapolates_while_ablation_is_chance():
    torch.manual_seed(0)
    cfg = RecallConfig(n_keys=8, n_vals=8, n_filler=4, d_model=32, d_mem=32)
    task = RecallTask(cfg)

    mem = MemoryRecallModel(cfg)
    train_recall(mem, task, steps=500, length=24, n_pairs=3, bs=64, seed=0)
    abl = MemorylessModel(cfg)
    train_recall(abl, task, steps=500, length=24, n_pairs=3, bs=64, seed=0)

    mem_train = eval_recall(mem, task, length=24, n_pairs=3)
    mem_extrap = eval_recall(mem, task, length=64, n_pairs=3)   # > training length
    abl_acc = eval_recall(abl, task, length=24, n_pairs=3)

    chance = 1.0 / cfg.n_vals
    assert mem_train > 0.5, f"memory failed at training length: {mem_train}"
    assert mem_extrap > 0.4, f"memory failed to extrapolate: {mem_extrap}"
    assert abl_acc < 0.3, f"memoryless ablation should be near chance {chance:.3f}, got {abl_acc}"
