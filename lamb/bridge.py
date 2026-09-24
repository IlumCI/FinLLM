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


# Constants that a word problem needs but never writes down. "Half as many",
# "twice", "a third", "20% off" all name an operation whose *other* operand is
# implicit, and the parser cannot see it because it is not in the text. Preloading
# them as registers keeps the decision in the program -- ``x / 2`` rather than a
# parser deciding that "half" means 0.5 -- which is the same argument the percent
# flag already makes, and the same reason the register machine exists at all.
# 1/2/3/100 are the *operation* constants: "half as many", "a third", "20% off" each
# name an operation whose other operand is never written down.
#
# 60/24/7/12/52/1000 are *unit* constants, and they are here because of a measurement,
# not a hunch. Categorising what still failed to align after compound decomposition
# found **59.6% of the remainder needed an integer that is nowhere in the problem** --
# "Weng earns $12 an hour ... 50 minutes" requires 60, and no parser can extract a
# number the text does not contain. Adding them moved coverage 0.614 -> 0.680.
#
# **They are only safe with a wider register file, and this is the sharp edge.** At
# ``n_operands=12`` these same constants *halve* coverage (0.614 -> 0.249) because 97%
# of rows fill the file and real quantities get pushed out of it. Constants and file
# width are one decision, not two: see ``BridgeConfig.n_operands``.
DEFAULT_CONSTANTS: Tuple[int, ...] = (1, 2, 3, 100, 60, 24, 7, 12, 52, 1000)


def registers_from_quantities(quantities: Sequence[Sequence[Quantity]],
                              n_registers: int,
                              constants: Sequence[int] = ()
                              ) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """Operand register file from the text's numbers, padded to a fixed width.

    Returns ``(values, scales, counts)``: the scaled integers, the power of ten each
    was scaled by, and how many were real -- the last so the program's pointer mask
    can be told which registers hold a quantity and which are padding, since a
    program that points at a slot no number went into is not a worse program.

    **The scale is returned because dropping it was the bug this whole layer exists
    to prevent.** This function used to return ``[q.value for q in qs]``, so ``3.25``
    entered the register file as the integer ``325`` and ``7`` as ``7``, and adding
    them gave ``332``. That is exactly the mixed-scale failure :mod:`lamb.rational`
    was written to kill -- *"a confidently wrong number is worse than a missing
    feature"* -- reintroduced at the one seam where nothing downstream could catch
    it. :meth:`lamb.rational.RationalAlgebra.encode_scaled` is the join that
    consumes both lists, and a scale there is simply a denominator.

    ``percent`` is deliberately *not* resolved here. ``20%`` is neither the number
    20 nor the number 0.2 until the program says which, and with ``1`` and ``100``
    available as constant registers that decision is expressible as a program --
    which is where it belongs. :func:`quantity_percent_flags` surfaces it.

    ``constants`` occupy the *first* slots, so their addresses are the same in every
    row and a pointer that learns "register 1 is two" learns something stable. The
    real-operand count includes them, since they are as readable as any quantity.
    """
    cs = list(constants)
    room = n_registers - len(cs)
    if room < 0:
        raise ValueError(f"{len(cs)} constants do not fit in {n_registers} registers")
    vals, scales, counts = [], [], []
    for qs in quantities:
        picked = list(qs)[:room]
        counts.append(len(cs) + len(picked))
        pad = room - len(picked)
        vals.append(cs + [q.value for q in picked] + [0] * pad)
        # Padding is 0/10**0 == 0, an ordinary representable value, so a masked
        # pointer that leaks weight onto a pad slot contributes a real zero rather
        # than an undefined one.
        scales.append([0] * len(cs) + [q.scale for q in picked] + [0] * pad)
    return vals, scales, counts


def quantity_percent_flags(quantities: Sequence[Sequence[Quantity]],
                           n_registers: int,
                           constants: Sequence[int] = ()) -> List[List[bool]]:
    """Which register slots came from a ``%`` literal, aligned to the register file.

    Takes ``constants`` for the same reason :func:`registers_from_quantities` does:
    the flags have to line up with the slots, and a flag list that is off by the
    number of constants is worse than no flags at all.
    """
    n_const = len(list(constants))
    room = n_registers - n_const
    out = []
    for qs in quantities:
        picked = list(qs)[:room]
        out.append([False] * n_const + [q.percent for q in picked]
                   + [False] * (room - len(picked)))
    return out


# -- lexical quantities ----------------------------------------------------
# Written cardinals only. "twice", "half" and "a third" are *operations* whose
# other operand is implicit, and they are served by DEFAULT_CONSTANTS plus the
# instruction set rather than by the parser inventing a number -- the same split
# the percent flag makes. What is left here is unambiguous: "three" is 3 wherever
# it appears, and leaving it invisible means a problem whose quantities are partly
# spelled out has an incomplete register file, which no program can recover from.
_LEXICAL: dict = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "dozen": 12, "couple": 2, "pair": 2,
}
_LEXICAL_RE = re.compile(r"\b(" + "|".join(sorted(_LEXICAL, key=len, reverse=True))
                         + r")\b", re.IGNORECASE)


def extract_lexical_quantities(text: str) -> List[Quantity]:
    """Written cardinals as :class:`Quantity`, with their spans.

    ``"a"``/``"an"`` are deliberately absent. They mean one often enough to be
    tempting and mean nothing often enough ("a discount", "an hour later") that
    admitting them would flood the register file with ones and push real quantities
    out of it, which is the failure mode that costs the most: a program cannot be
    right about a number that is not there.
    """
    return [Quantity(value=_LEXICAL[m.group(1).lower()], scale=0,
                     start=m.start(), end=m.end(), percent=False)
            for m in _LEXICAL_RE.finditer(text)]


def all_quantities(text: str, lexical: bool = True,
                   max_decimals: int = 2) -> List[Quantity]:
    """Digit and (optionally) written quantities, in the order they appear."""
    out = list(extract_quantities(text, max_decimals=max_decimals))
    if lexical:
        out.extend(extract_lexical_quantities(text))
    return sorted(out, key=lambda q: q.start)


# -- the frozen encoder, run once -----------------------------------------
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def encode_dataset(texts: Sequence[str], encoder: str = DEFAULT_ENCODER,
                   out_path: Optional[str] = None, max_len: int = 192,
                   batch_size: int = 32, device: str = "cpu") -> dict:
    """Run the frozen encoder over a dataset **once** and cache its token states.

    This is the step the rest of this module has always assumed and never had. The
    encoder is frozen, so its outputs are a function of the dataset alone; computing
    them per training step would pay for the largest model in the system on every
    batch, to recompute a constant. Cached, the expensive model never participates
    in training at all, which is what makes this affordable on hardware that could
    not train it -- and why everything else here takes embeddings rather than text.

    Token states, not the pooled sentence vector: the resampler cross-attends, so
    collapsing the problem to one vector before it gets there would throw away the
    structure it is there to read.

    ``truncated`` is returned rather than logged away. A problem whose question was
    cut off is not a hard problem, it is a different problem, and an accuracy that
    silently includes a few hundred of them is not measuring what it claims to.
    """
    try:
        import torch as _torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(
            "the language bridge needs `transformers`; it is an optional extra so "
            "that the pinned CPU environment behind every arithmetic number in this "
            "repo does not move underneath it. Install with: uv sync --extra bridge"
        ) from exc

    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    model.requires_grad_(False)                # frozen means frozen, not "untouched"

    n_trunc = 0
    chunks, masks = [], []
    with _torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            raw = tok(batch, add_special_tokens=True)["input_ids"]
            n_trunc += sum(1 for ids in raw if len(ids) > max_len)
            enc = tok(batch, padding="max_length", truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state                   # (b, T, d_enc)
            chunks.append(h.to(_torch.float16).cpu())
            masks.append((enc["attention_mask"] == 0).cpu())     # True at padding
    out = {
        "enc": _torch.cat(chunks),
        "pad_mask": _torch.cat(masks),
        "meta": {"encoder": encoder, "max_len": max_len, "n": len(texts),
                 "d_enc": int(chunks[0].shape[-1]), "truncated": n_trunc},
    }
    if out_path:
        _torch.save(out, out_path)
    return out


def load_encoded(path: str) -> dict:
    """Read a cache written by :func:`encode_dataset`."""
    return torch.load(path, map_location="cpu", weights_only=False)
