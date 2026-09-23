"""Language as a peripheral: frozen encoder in, latent core, structured actions out.

The model does not learn a token distribution. Text is encoded by a **frozen**
encoder whose weights are never touched, a small learned resampler compresses that
into ``K`` latent vectors, and the latent core turns those into a *program* over
registers (:mod:`lamb.regmachine`) which the residue algebra executes exactly. No
part of the language model is trained, and no part of the answer is generated as
tokens.

Three consequences, and the first is practical rather than architectural:

* **The encoder runs once, not once per step.** Frozen means its outputs can be
  computed for the whole dataset ahead of time and cached, so every training run
  afterwards reads tensors off disk. The expensive model never participates in
  training, which is what makes this affordable on hardware that could not train it.
  Everything here therefore takes *embeddings*, never text, and
  :func:`encode_dataset` is a separate one-off step.
* **The numbers are not learned.** A word problem's quantities are in its text;
  pulling them out is a regular expression and turning them into residues is a
  fixed linear map (ROADMAP 3a-vi measured that asking a network to induce that map
  leaves it at chance). So the learned surface shrinks to the one thing that
  actually requires understanding: *which computation to perform*.
* **What contamination remains is measurable rather than structural.** Training the
  reasoning core on self-generated problems keeps it clean, but a pretrained
  encoder has seen the benchmarks, and "frozen" means its weights do not move, not
  that the information is absent -- benchmark leakage is detectable in
  representations (arXiv:2608.12652). The claim that this repo's GSM8K/GSM1K gap is
  zero *by construction* does not survive a pretrained encoder and is not made. What
  survives is an experiment: run the same core on a pretrained encoder and on one
  trained only on generated text, and the difference prices the encoder's prior
  knowledge. Nobody runs that, and this design makes it a one-line arm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .model.transformer import RMSNorm

# Quantities as they appear in grade-school word problems: optional sign, digits
# with optional thousands separators, optional decimal part, optional trailing %.
_NUMBER = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+|-?\d+)(?:\.(\d+))?(%?)")


@dataclass(frozen=True)
class Quantity:
    """A number found in the text, kept with where it was found."""

    value: int          # scaled to an integer (see `scale`)
    scale: int          # power of ten the value was multiplied by, 0 for integers
    start: int
    end: int
    percent: bool

    @property
    def approx(self) -> float:
        return self.value / (10 ** self.scale)


def extract_quantities(text: str, max_decimals: int = 2) -> List[Quantity]:
    """Pull the numbers out of a problem, exactly.

    Decimals are carried as scaled integers rather than floats, because the algebra
    is a ring of integers and a rounded operand is a wrong operand. ``3.25`` becomes
    ``(325, scale=2)``; a program that combines mixed scales has to be emitted in a
    common scale, which is a constraint on the program rather than on the parser.

    This is deliberately not learned. The quantities are *in the text*, and a model
    that has to rediscover decimal notation is spending capacity on the one part of
    the problem that was never ambiguous.
    """
    out: List[Quantity] = []
    for m in _NUMBER.finditer(text):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or ""
        if len(frac) > max_decimals:
            frac = frac[:max_decimals]
        neg = whole.startswith("-")
        digits = (whole.lstrip("-") + frac) or "0"
        val = int(digits) * (-1 if neg else 1)
        out.append(Quantity(value=val, scale=len(frac), start=m.start(), end=m.end(),
                            percent=bool(m.group(3))))
    return out


class Resampler(nn.Module):
    """Query-conditioned cross-attention: ``(B, T, d_enc)`` -> ``(B, K, d_model)``.

    ``K`` learned queries attend over the frozen encoder's states, so the cost of
    everything downstream is fixed by ``K`` rather than by how long the problem is --
    a paragraph and a sentence both become ``K`` vectors. This is the
    perceiver-resampler/Q-Former arrangement that vision-language models use to hang
    a learned bridge off a frozen tower, applied to a latent reasoning core instead
    of a decoder.
    """

    def __init__(self, d_enc: int, d_model: int, n_latents: int = 32,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_latents = n_latents
        self.queries = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.in_proj = nn.Linear(d_enc, d_model)
        self.in_norm = RMSNorm(d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "q_norm": RMSNorm(d_model),
                "kv_norm": RMSNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True),
                "ff_norm": RMSNorm(d_model),
                "ff": nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                    nn.Linear(2 * d_model, d_model)),
            }))
        self.out_norm = RMSNorm(d_model)

    def forward(self, enc: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``enc`` is ``(B, T, d_enc)``; ``pad_mask`` is True at padding."""
        kv = self.in_norm(self.in_proj(enc))
        x = self.queries.unsqueeze(0).expand(enc.size(0), -1, -1)
        for layer in self.layers:
            q = layer["q_norm"](x)
            k = layer["kv_norm"](kv)
            a, _ = layer["attn"](q, k, k, key_padding_mask=pad_mask, need_weights=False)
            x = x + a
            x = x + layer["ff"](layer["ff_norm"](x))
        return self.out_norm(x)


class LanguageFront(nn.Module):
    """The whole peripheral: cached embeddings and text-extracted quantities in,
    latents and register contents out.

    Deliberately takes embeddings rather than text. The encoder is frozen, so its
    outputs belong in a cache computed once (:func:`encode_dataset`), and keeping it
    out of this module is what stops a training loop from paying for it.
    """

    def __init__(self, d_enc: int, d_model: int, n_latents: int = 32, **kw):
        super().__init__()
        self.resampler = Resampler(d_enc, d_model, n_latents=n_latents, **kw)

    def forward(self, enc: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        return self.resampler(enc, pad_mask)


def registers_from_quantities(quantities: Sequence[Sequence[Quantity]],
                              n_registers: int) -> Tuple[List[List[int]], List[int]]:
    """Operand register file from the text's numbers, padded to a fixed width.

    Returns the integer values and how many were real, so the program's pointer
    mask can be told which registers hold a quantity and which are padding -- a
    program that points at a slot no number went into is not a worse program.
    """
    vals, counts = [], []
    for qs in quantities:
        v = [q.value for q in qs][:n_registers]
        counts.append(len(v))
        vals.append(v + [0] * (n_registers - len(v)))
    return vals, counts
