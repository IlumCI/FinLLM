"""Test-time neural memory: shapes/grads, and in-context associative recall.

The recall task binds a *fresh* random key->label mapping every episode, so the
answer cannot be memorised in the weights -- it can only be produced by writing
the bindings into the memory during the forward pass and reading them back. A
model that learns it above chance is direct evidence the ``infContext`` memory
works.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lamb.model.memory import NeuralMemory


def test_memory_shapes_and_grad():
    torch.manual_seed(0)
    mem = NeuralMemory(d_model=32, d_mem=16)
    h = torch.randn(4, 10, 32, requires_grad=True)
    out = mem(h)
    assert out.shape == (4, 10, 32)
    out.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_memory_read_is_causal():
    """Reads depend only on strictly earlier tokens (memory written before read)."""
    torch.manual_seed(0)
    mem = NeuralMemory(d_model=16, d_mem=16)
    h = torch.randn(1, 6, 16)
    out_full = mem(h)
    # Perturbing the last token must not change the read at position 0.
    h2 = h.clone()
    h2[0, -1] += 5.0
    out_pert = mem(h2)
    assert torch.allclose(out_full[0, 0], out_pert[0, 0], atol=1e-5)


class _RecallModel(nn.Module):
    def __init__(self, n_keys: int, n_labels: int, d: int = 32, d_mem: int = 32):
        super().__init__()
        self.key_emb = nn.Embedding(n_keys, d)
        self.label_emb = nn.Embedding(n_labels, d)
        self.query_marker = nn.Parameter(torch.randn(d) * 0.02)
        self.mem = NeuralMemory(d, d_mem)
        self.readout = nn.Linear(d, n_labels)

    def forward(self, study_keys, study_labels, query_key):
        # study tokens carry (key + label); the query token carries (key + marker)
        study = self.key_emb(study_keys) + self.label_emb(study_labels)  # (B, N, d)
        query = (self.key_emb(query_key) + self.query_marker).unsqueeze(1)  # (B, 1, d)
        seq = torch.cat([study, query], dim=1)
        read = self.mem(seq)
        return self.readout(read[:, -1])  # predict the queried key's label


def test_in_context_associative_recall():
    torch.manual_seed(0)
    n_keys, n_labels, d = 6, 6, 32
    model = _RecallModel(n_keys, n_labels, d=d, d_mem=d)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)

    def episode_batch(bs: int):
        # each row studies all keys 0..n_keys-1 with a fresh random label mapping
        labels = torch.stack([torch.randperm(n_labels, generator=g)[:n_keys] for _ in range(bs)])
        keys = torch.arange(n_keys).unsqueeze(0).expand(bs, -1)
        q_idx = torch.randint(0, n_keys, (bs,), generator=g)
        q_key = keys[torch.arange(bs), q_idx]
        target = labels[torch.arange(bs), q_idx]
        return keys, labels, q_key, target

    for _ in range(400):
        keys, labels, q_key, target = episode_batch(64)
        logits = model(keys, labels, q_key)
        loss = nn.functional.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        keys, labels, q_key, target = episode_batch(256)
        acc = (model(keys, labels, q_key).argmax(-1) == target).float().mean().item()
    # Chance is 1/n_labels ~ 0.167; memory must beat it clearly.
    assert acc > 0.5, f"associative recall accuracy {acc:.3f} not above chance"
