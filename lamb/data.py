"""Collate tokenised examples into padded training tensors.

Kept separate from :mod:`lamb.tokenizer` so the tokenizer stays torch-free.
The autoregressive shift and the answer-only loss mask are applied here.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .tokenizer import Encoded


def collate(examples: List[Encoded], pad_id: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Pad a list of :class:`Encoded` and produce the next-token training batch.

    Returns a dict of tensors, all shaped ``(B, T)`` with ``T = maxlen - 1``:

    * ``input_ids``      -- context tokens (sequence minus the last position)
    * ``abacus_ids``     -- Abacus intra-number positions aligned to inputs
    * ``value``          -- operand magnitudes aligned to inputs
    * ``value_mask``     -- 1 where ``value`` is meaningful
    * ``target_ids``     -- next-token labels (sequence shifted left)
    * ``loss_mask``      -- 1 on answer-token targets only (incl. EOS)
    * ``pad_mask``       -- True where an input position is padding
    """
    b = len(examples)
    maxlen = max(len(e) for e in examples)
    t = maxlen - 1

    ids = np.full((b, maxlen), pad_id, dtype=np.int64)
    abacus = np.zeros((b, maxlen), dtype=np.int64)
    value = np.zeros((b, maxlen), dtype=np.float32)
    vmask = np.zeros((b, maxlen), dtype=np.float32)
    lengths = np.zeros((b,), dtype=np.int64)
    ans_start = np.zeros((b,), dtype=np.int64)

    for i, e in enumerate(examples):
        n = len(e)
        ids[i, :n] = e.ids
        abacus[i, :n] = e.abacus
        value[i, :n] = e.value
        vmask[i, :n] = e.value_mask
        lengths[i] = n
        ans_start[i] = e.ans_start

    input_ids = ids[:, :-1]
    target_ids = ids[:, 1:]

    # Answer-only loss: target position j (aligned to original index j+1) is
    # supervised iff it lies in [ans_start, length-1] (EOS is length-1).
    col = np.arange(t)[None, :] + 1  # original index of each target
    loss_mask = ((col >= ans_start[:, None]) & (col <= (lengths[:, None] - 1))).astype(np.float32)

    # Input position j is padding iff j >= length.
    pad_mask = (np.arange(t)[None, :] >= lengths[:, None])

    def to(x, dtype):
        return torch.as_tensor(x, dtype=dtype, device=device)

    return {
        "input_ids": to(input_ids, torch.long),
        "abacus_ids": to(abacus[:, :-1], torch.long),
        "value": to(value[:, :-1], torch.float32),
        "value_mask": to(vmask[:, :-1], torch.float32),
        "target_ids": to(target_ids, torch.long),
        "loss_mask": to(loss_mask, torch.float32),
        "pad_mask": to(pad_mask, torch.bool),
    }
