"""A residue number system for the latent space -- arithmetic the latents *do*.

Every latent-reasoning system so far supervises its latents against **tokens**:
LOTUS against gold chain-of-thought tokens, SIM-CoT against a decoder's, C-MTP
against averaged token embeddings, and this repo (until now) against the digits of
its own gold trace. A token target teaches a latent to *name* a number. It teaches
it nothing about what numbers *are*, so composition -- the thing reasoning actually
consists of -- has to be relearned by the network for every magnitude it meets.

That is the mechanism behind the arithmetic length-generalization wall. Models that
do extrapolate turn out to have discovered a *periodic* representation on their own:
tokens as phases, addition as rotation (arXiv:2511.23443 proves the benefit for
modular addition; arXiv:2606.17399 finds multiplication reduced to addition in
discrete-log space by the same mechanism). The standard response is to build
periodicity into the **positional** encoding -- Abacus (arXiv:2405.17399, already in
this model) and position coupling (arXiv:2405.20671) -- so the network is helped to
find the algorithm. It is still the network that has to run it.

LAMb is in a position no language model is: an exact evaluator hands it the true
value of *every intermediate* of *every problem*, free, unlimited and unannotated.
That is enough to skip the discovery entirely and give the latent space the algebra
outright.

So: a value is carried as its residues modulo a set of coprime moduli. By the
Chinese Remainder Theorem the residues determine the value uniquely inside
``prod(moduli)``, and -- the point -- the operations come along:

    (a + b) mod p  is a cyclic convolution of the residue distributions
    (a - b) mod p  is a cyclic cross-correlation
    (a * b) mod p  is residue-wise, one small multiplication table

Three things follow, and none of them is a training result -- they hold by
construction, which is why this module is tested without a model in the loop:

* **Magnitude extrapolation is structural.** A residue does not grow with its
  number. Composition is exact at 6 digits having only ever been trained at 1.
* **Multiplication gets cheaper, not dearer.** In digits, multiplication is where
  length generalization collapses hardest; in residues it is the *easiest* of the
  three, a lookup per modulus with no carries to propagate.
* **The model can check its own arithmetic with no labels.** If the latents live in
  a known algebra, internal consistency is a computable quantity at inference time,
  on problems whose answer nobody knows. That is a correctness signal that survives
  leaving the synthetic distribution -- which is what a verifier cannot do.

Everything here is differentiable on *distributions* over residues, so it composes
with a network that is uncertain about a residue rather than committed to one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Sequence, Tuple

import torch

# Coprime, small, and -- the part that is easy to get wrong -- chosen for the
# *multiplicative order of 10*, not just for size.
#
# The network has to produce ``n mod p`` from digits, which is
# ``sum_i digit_i * (10^i mod p)``: a weighted digit sum whose coefficients repeat
# with period ``ord_p(10)``. That period is exactly how many digit positions have to
# be seen before the pattern is learnable, and everything past it extrapolates for
# free. So a modulus with a long order is nearly useless at short operand widths:
# ord(10) is 16 mod 17, 18 mod 19 and 22 mod 23, so a run trained on 1-3 digits
# learns almost nothing about how position 5 contributes.
#
# These are picked for short orders, and they strictly dominate the obvious
# small-primes choice (7,11,13,17,19,23) on every axis that matters:
#
#     moduli                     units   range      max period
#     (7,11,13,17,19,23)           90    +/-3.7M       22
#     (2,5,9,11,7,13,37)           84    +/-1.67M       6
#
# and three of them are the schoolbook divisibility rules, learnable immediately at
# any width: mod 9 is the digit sum (10 = 1 mod 9), mod 11 the alternating digit
# sum, mod 2 and mod 5 the last digit alone.
DEFAULT_MODULI: Tuple[int, ...] = (2, 5, 9, 11, 7, 13, 37)


def order10(p: int) -> int:
    """Multiplicative order of 10 mod ``p`` -- digit positions before the
    coefficient pattern repeats, hence the operand width a run must reach before
    this modulus starts extrapolating. 1 for moduli dividing a power of 10."""
    if p % 2 == 0 or p % 5 == 0:
        return 1
    k, x = 1, 10 % p
    while x != 1:
        x = x * 10 % p
        k += 1
    return k


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def _inv(a: int, m: int) -> int:
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} is not invertible mod {m}: the moduli must be coprime")
    return x % m


@dataclass(frozen=True)
class ResidueSystem:
    """The algebra: a set of coprime moduli plus exact encode/decode.

    ``decode`` returns a *signed* value in ``(-M/2, M/2]``, so subtraction is closed
    without a separate sign channel -- a negative number is just a residue vector
    like any other, which is what lets the same composition rule serve ``+`` and
    ``-``.
    """

    moduli: Tuple[int, ...] = DEFAULT_MODULI

    def __post_init__(self) -> None:
        for i, p in enumerate(self.moduli):
            if p < 2:
                raise ValueError(f"modulus {p} must be >= 2")
            for q in self.moduli[i + 1:]:
                if _egcd(p, q)[0] != 1:
                    raise ValueError(f"moduli {p} and {q} are not coprime")

    # -- range ------------------------------------------------------------
    @property
    def M(self) -> int:
        """Product of the moduli: the size of the representable ring."""
        return reduce(lambda a, b: a * b, self.moduli, 1)

    @property
    def n_units(self) -> int:
        """Total residue classes -- the width a network head needs."""
        return sum(self.moduli)

    def bounds(self) -> Tuple[int, int]:
        """The signed interval this system represents exactly."""
        half = self.M // 2
        return -half + (1 - self.M % 2), half

    @property
    def max_period(self) -> int:
        """Widest digit-coefficient period across the moduli -- the operand width
        this system needs to see before every modulus extrapolates."""
        return max(order10(p) for p in self.moduli)

    def representable(self, n: int) -> bool:
        lo, hi = self.bounds()
        return lo <= n <= hi

    # -- exact integer <-> residue ---------------------------------------
    def residues(self, n: int) -> List[int]:
        return [n % p for p in self.moduli]

    def crt(self, res: Sequence[int]) -> int:
        """Signed CRT reconstruction. Exact inverse of :meth:`residues` in range."""
        M, x = self.M, 0
        for p, r in zip(self.moduli, res):
            Mi = M // p
            x += (r % p) * Mi * _inv(Mi, p)
        x %= M
        return x - M if x > M // 2 else x

    # -- slicing a flat head into per-modulus blocks ----------------------
    def split(self, flat: torch.Tensor) -> List[torch.Tensor]:
        """``(..., n_units)`` -> one ``(..., p)`` block per modulus."""
        out, i = [], 0
        for p in self.moduli:
            out.append(flat[..., i:i + p])
            i += p
        return out

    def onehot(self, values: Sequence[int], device: str = "cpu") -> torch.Tensor:
        """Exact codes for integers: ``(N, n_units)``, one-hot within each block."""
        out = torch.zeros(len(values), self.n_units, device=device)
        for i, v in enumerate(values):
            off = 0
            for p in self.moduli:
                out[i, off + (int(v) % p)] = 1.0
                off += p
        return out

    def targets(self, values: Sequence[int], device: str = "cpu") -> torch.Tensor:
        """Per-modulus class indices ``(N, n_moduli)`` -- cross-entropy targets."""
        return torch.tensor([[int(v) % p for p in self.moduli] for v in values],
                            dtype=torch.long, device=device)

    def decode(self, blocks: Sequence[torch.Tensor]) -> List[int]:
        """Argmax each block, then CRT. ``blocks[k]`` is ``(N, p_k)``."""
        picks = torch.stack([b.argmax(dim=-1) for b in blocks], dim=-1)  # (N, K)
        return [self.crt(row.tolist()) for row in picks]


class ResidueAlgebra:
    """Differentiable composition of residue *distributions*.

    The tables are built once. ``add``/``sub`` are cyclic convolution and
    cross-correlation; ``mul`` contracts the modular multiplication table. A network
    that is unsure of a residue composes its uncertainty correctly instead of having
    to commit first -- which is what makes this usable as a training signal rather
    than only as a decoder.
    """

    def __init__(self, system: Optional[ResidueSystem] = None, device: str = "cpu"):
        self.sys = system or ResidueSystem()
        self.device = device
        # mul_table[k][i, j] = (i * j) % p  -- as a permutation-index tensor
        self.mul_idx: List[torch.Tensor] = []
        for p in self.sys.moduli:
            i = torch.arange(p, device=device).view(p, 1)
            j = torch.arange(p, device=device).view(1, p)
            self.mul_idx.append((i * j) % p)

    # -- distribution helpers --------------------------------------------
    @staticmethod
    def _probs(block: torch.Tensor, logits: bool) -> torch.Tensor:
        return torch.softmax(block, dim=-1) if logits else block

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cyclic convolution: the distribution of ``(x + y) mod p``."""
        p = a.size(-1)
        # out[c] = sum_i a[i] * b[(c - i) mod p]
        idx = (torch.arange(p, device=a.device).view(p, 1)
               - torch.arange(p, device=a.device).view(1, p)) % p     # (c, i)
        return torch.einsum("...i,...ci->...c", a, b[..., idx])

    def sub(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cyclic cross-correlation: the distribution of ``(x - y) mod p``.

        ``out[c] = sum_i a[i] * b[(i - c) mod p]``: given ``x = i`` and a difference
        of ``c``, the subtrahend must be ``i - c``. Indexing this the other way round
        computes ``y - x``, which agrees with ``x - y`` only when the two are equal --
        so it passes about a tenth of random cases and looks like a near miss rather
        than a sign error.
        """
        p = a.size(-1)
        idx = (torch.arange(p, device=a.device).view(1, p)
               - torch.arange(p, device=a.device).view(p, 1)) % p     # (c, i) -> i - c
        return torch.einsum("...i,...ci->...c", a, b[..., idx])

    def mul(self, a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
        """Residue-wise multiplication -- the case digits find hardest, and here
        the cheapest: one small table, no carries to propagate."""
        p = a.size(-1)
        outer = a.unsqueeze(-1) * b.unsqueeze(-2)                     # (..., i, j)
        out = torch.zeros(outer.shape[:-2] + (p,), device=a.device, dtype=outer.dtype)
        return out.index_add(-1, self.mul_idx[k].reshape(-1),
                             outer.reshape(*outer.shape[:-2], -1))

    # -- the public operation --------------------------------------------
    def compose_blocks(self, A: Sequence[torch.Tensor], B: Sequence[torch.Tensor],
                       op: str) -> List[torch.Tensor]:
        """Compose two already-split distributions, block by block.

        The recursive form: composing a whole tree means feeding one composition's
        output straight into the next, so the flat/split conversion must not sit in
        the middle of the recursion.
        """
        out = []
        for k, (x, y) in enumerate(zip(A, B)):
            if op == "+":
                out.append(self.add(x, y))
            elif op == "-":
                out.append(self.sub(x, y))
            elif op == "*":
                out.append(self.mul(x, y, k))
            else:
                raise ValueError(f"unsupported operator {op!r}")
        return out

    def compose(self, a_flat: torch.Tensor, b_flat: torch.Tensor, op: str,
                logits: bool = True) -> List[torch.Tensor]:
        """Compose two coded values. Returns one distribution block per modulus.

        This is the step the network does *not* perform. Given the codes of two
        sub-results, the value of their combination is fixed by the algebra, so
        nothing about arithmetic has to be learned a second time at a magnitude
        the network has not seen.
        """
        A = [self._probs(x, logits) for x in self.sys.split(a_flat)]
        B = [self._probs(x, logits) for x in self.sys.split(b_flat)]
        return self.compose_blocks(A, B, op)

    def compose_exact(self, a: int, b: int, op: str) -> int:
        """The same composition on exact integers -- the oracle the tests check."""
        enc = lambda v: self.sys.onehot([v], self.device)              # noqa: E731
        return self.sys.decode(self.compose(enc(a), enc(b), op, logits=False))[0]
