"""The latent ALU: the latents hold values, and the algebra does the arithmetic.

:mod:`lamb.algebra` gives an exact, differentiable residue arithmetic. This wires it
to the latent block, which changes what a latent position *is*.

Until now a latent was supervised to predict the **digit tokens** of its
intermediate. Two things follow from that and both are bad. A latent budget has to
grow with the number of *digits* rather than the number of reasoning *steps* --
depth-3 needs 14 token slots for 6 intermediates. And the model has to relearn
composition at every magnitude, because nothing in a digit target says that
``12`` and ``13`` are one apart.

Here a latent position holds one **value**, coded as its residues, and the
combination of two values is not predicted at all -- it is *computed*, by the
algebra, exactly. The network's job shrinks to mapping a sub-expression to a code.
Arithmetic stops being something the network approximates and becomes something the
architecture performs.

The layout, for a balanced expression tree of depth ``D``:

    slots 0 .. T-1   the T = 2^D - 2 intermediates, in the grammar's post-order
    slot  T          the root -- the model's own *direct* guess at the answer

The root slot is what makes this self-checking. The answer is available two ways:
the model states it (slot ``T``), and the algebra composes it from the children.
On a correct trajectory the two agree. Their disagreement is a **residual that
needs no label**, because it compares the model against arithmetic rather than
against a key -- so unlike an exact verifier, it still means something on a problem
nobody has the answer to, and on a distribution the verifier was never built for.
That is the property this is for.

Nothing is decoded on the way: a residue code is not a token, the latents are never
verbalised, and the model stays a blackbox that happens to be checkable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .algebra import ResidueAlgebra, ResidueSystem

# An expression tree: a leaf is ``(op, a, b)`` over literals, a node is
# ``(op, left, right)`` over sub-trees.
Leaf = Tuple[str, int, int]
Node = Tuple[str, "Tree", "Tree"]
Tree = Union[Leaf, Node]


def parse_expr(expr: str) -> Tree:
    """Parse the grammar's output back into its tree.

    The structure comes from the *input*, never from the answer, so using it at
    inference is reading the question rather than peeking at the key. The grammar
    emits either ``a<op>b`` or ``(L)<op>(R)``, and operands are non-negative, so a
    top-level operator is unambiguous.

    ``/`` is accepted here because the grammar emits it (``ops_key=2``). Parsing it
    and composing it are different questions: the integer :class:`ResidueAlgebra`
    has no division at all -- an inverse needs a divisor coprime to every modulus,
    and the short-digit-period moduli are exactly the set that denies that -- so a
    ``/`` tree belongs to :class:`lamb.regmachine.RegisterMachine` with
    ``rational=True``, and reaches :class:`LatentALU` only as an error. Leaving the
    parser unable to read it was worse: it made the grammar's own output
    unparseable, so division data existed and nothing could consume it.
    """
    expr = expr.strip()
    if expr.startswith("("):
        depth = 0
        for i, c in enumerate(expr):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
        op = expr[i + 1]
        return (op, parse_expr(expr[1:i]), parse_expr(expr[i + 3:-1]))
    for j, c in enumerate(expr):
        if c in "+-*/" and j > 0:
            return (c, int(expr[:j]), int(expr[j + 1:]))
    raise ValueError(f"cannot parse {expr!r}")


def is_leaf(t: Tree) -> bool:
    return isinstance(t[1], int)


def n_internal(t: Tree) -> int:
    """Intermediates excluding the root -- the grammar's trace length."""
    return 0 if is_leaf(t) else 2 + n_internal(t[1]) + n_internal(t[2])


def slot_order(t: Tree) -> List[Tree]:
    """Sub-trees in the grammar's post-order, root excluded.

    Must match ``TaskGrammar._build_traced``, which returns
    ``ltrace + [lval] + rtrace + [rval]`` -- so: everything the left child
    computed, the left child, everything the right child computed, the right child.
    """
    if is_leaf(t):
        return []
    _, l, r = t
    return slot_order(l) + [l] + slot_order(r) + [r]


def slot_pairs(t: Tree, base: int = 0) -> List[Tuple[Tree, int]]:
    """``(sub-tree, slot index)`` in the same post-order as :func:`slot_order`.

    Indices are assigned by *position* in the traversal, never looked up by value.
    ``(1+2)-(1+2)`` has two equal-but-distinct children, so a value-keyed lookup
    hands both the same slot and silently reads one latent twice -- a bug that only
    shows on the minority of problems with a repeated sub-expression, which is
    exactly the kind that survives a smoke test.
    """
    if is_leaf(t):
        return []
    _, l, r = t
    out = slot_pairs(l, base)
    li = base + len(out)
    out = out + [(l, li)]
    rs = slot_pairs(r, li + 1)
    out = out + rs
    return out + [(r, li + 1 + len(rs))]


def _slot_of(pairs: List[Tuple[Tree, int]], node: Tree) -> int:
    """Identity lookup. ``is``, not ``==``: see :func:`slot_pairs`."""
    for sub, idx in pairs:
        if sub is node:
            return idx
    raise KeyError("sub-tree has no slot")


class ScalarALU(nn.Module):
    """The control: one *scalar* per latent, composed by ordinary arithmetic.

    The obvious objection to a residue system is that it is unnecessary -- a latent
    could simply regress its value, and ``a + b`` is exact on scalars too, with no
    moduli, no CRT and no brittleness to a single wrong residue.

    The counter-argument is that regression over unbounded integers is badly
    conditioned where classification over a small ring is not: a scalar has to hit
    ``47.0`` rather than pick one of eleven classes, the error does not quantise, and
    multiplication squares whatever error survives. But that is an argument, and this
    is the arm that turns it into a measurement. It is deliberately the *naive*
    version a reviewer would propose -- fixed scale, Huber loss, round at the end --
    because a naive version that wins would mean the residue machinery is not
    earning its place.
    """

    def __init__(self, d_model: int, scale: float = 1.0e4):
        super().__init__()
        self.scale = scale
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.bias)

    def values(self, latent_h: torch.Tensor) -> torch.Tensor:
        """``(B, L, d) -> (B, L)`` predicted values, in units of ``scale``."""
        return self.head(latent_h).squeeze(-1) * self.scale

    def _from_leaves(self, row: torch.Tensor, t: Tree,
                     pairs: List[Tuple[Tree, int]]) -> torch.Tensor:
        if is_leaf(t):
            return row[_slot_of(pairs, t)]
        op, l, r = t
        a = self._from_leaves(row, l, pairs)
        b = self._from_leaves(row, r, pairs)
        if op not in ("+", "-", "*"):
            # ``parse_expr`` accepts ``/`` because the grammar emits it; this arm
            # cannot execute it. A named refusal beats the ``KeyError`` the dict
            # lookup used to raise, which reads like a bug in the ALU rather than an
            # unsupported instruction set. The residue arm says the same thing from
            # ``ResidueAlgebra.compose_blocks``.
            raise ValueError(f"the scalar ALU has no {op!r}; use "
                             f"RegisterMachine(rational=True)")
        return {"+": a + b, "-": a - b, "*": a * b}[op]

    def compose_tree(self, vals: torch.Tensor, trees: Sequence[Tree]) -> torch.Tensor:
        return torch.stack([self._from_leaves(vals[b], t, slot_pairs(t))
                            for b, t in enumerate(trees)])

    def value_loss(self, vals: torch.Tensor, targets: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """Huber on the scaled value -- the standard choice for a regression head."""
        err = torch.nn.functional.huber_loss(
            vals / self.scale, targets / self.scale, reduction="none")
        return (err * mask).sum() / mask.sum().clamp_min(1.0)


@dataclass
class AluConfig:
    moduli: Tuple[int, ...] = (7, 11, 13, 17, 19, 23)
    consistency_coef: float = 0.1     # weight on the label-free agreement term


class LatentALU(nn.Module):
    """Residue head over the latent block, plus algebraic composition.

    The head is the only new parameter: ``d_model -> sum(moduli)`` logits per latent
    position, read as one distribution per modulus.
    """

    def __init__(self, d_model: int, system: Optional[ResidueSystem] = None,
                 device: str = "cpu"):
        super().__init__()
        self.sys = system or ResidueSystem()
        self.alg = ResidueAlgebra(self.sys, device=device)
        self.head = nn.Linear(d_model, self.sys.n_units)
        nn.init.zeros_(self.head.bias)

    def codes(self, latent_h: torch.Tensor) -> torch.Tensor:
        """``(B, L, d) -> (B, L, n_units)`` residue logits per latent position."""
        return self.head(latent_h)

    # -- the arithmetic ---------------------------------------------------
    def _from_leaves(self, row: torch.Tensor, t: Tree,
                     pairs: List[Tuple[Tree, int]]) -> List[torch.Tensor]:
        """Blocks for sub-tree ``t``, built up from the **leaf** slots alone.

        This is where reasoning depth stops costing anything. The network is only
        ever asked for a leaf -- two literals and one operator -- and every internal
        node is *computed*. A depth-5 expression asks the network exactly the same
        question a depth-2 one does, so depth generalisation is not a capability the
        model has to acquire; it is a property of the composition.
        """
        if is_leaf(t):
            return self.sys.split(torch.softmax(row[_slot_of(pairs, t)], dim=-1))
        op, l, r = t
        return self.alg.compose_blocks(self._from_leaves(row, l, pairs),
                                       self._from_leaves(row, r, pairs), op)

    def compose_tree(self, codes: torch.Tensor,
                     trees: Sequence[Tree]) -> List[List[torch.Tensor]]:
        """Each row's answer, composed by the algebra from its leaf slots."""
        return [self._from_leaves(codes[b], t, slot_pairs(t))
                for b, t in enumerate(trees)]

    def internal_checks(self, codes: torch.Tensor, trees: Sequence[Tree]
                        ) -> List[List[Tuple[int, List[torch.Tensor]]]]:
        """``(slot, composed blocks)`` for every internal node the model also states.

        From depth 3 up an internal node is over-determined: the model predicts it
        in its own slot *and* the algebra derives it from that node's leaves. Their
        disagreement needs no label, which is the whole point -- it is a check
        against arithmetic rather than against a key, so it survives leaving the
        distribution the key was written for.
        """
        out = []
        for b, t in enumerate(trees):
            pairs = slot_pairs(t)
            row = []
            for sub, idx in pairs:
                if not is_leaf(sub):
                    row.append((idx, self._from_leaves(codes[b], sub, pairs)))
            out.append(row)
        return out

    def decode(self, blocks_per_row: Sequence[List[torch.Tensor]]) -> List[int]:
        return [self.sys.crt([int(bl.argmax(-1)) for bl in blocks])
                for blocks in blocks_per_row]

    # -- the label-free signal -------------------------------------------
    def consistency(self, codes: torch.Tensor, trees: Sequence[Tree],
                    root_slot: int) -> torch.Tensor:
        """Disagreement between the model's stated answer and the composed one.

        ``(B,)``, a cross-entropy of the direct root distribution against the
        algebraically composed one. No target is involved: it measures the model
        against arithmetic, so it is defined on any problem, answered or not.
        """
        composed = self.compose_tree(codes, trees)
        direct = self.sys.split(codes[:, root_slot])
        per_row = []
        for b, blocks in enumerate(composed):
            tot = codes.new_zeros(())
            for k, comp in enumerate(blocks):
                logp = torch.log_softmax(direct[k][b], dim=-1)
                tot = tot - (comp.detach() * logp).sum()
            per_row.append(tot)
        return torch.stack(per_row)

    # -- supervision ------------------------------------------------------
    def code_loss(self, codes: torch.Tensor, targets: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of each latent's residues against its gold value.

        ``targets`` is ``(B, L, n_moduli)`` class indices, ``mask`` ``(B, L)``. The
        gold values are exact and free -- the evaluator produces them for every
        intermediate of every problem, which is the asset a language model does not
        have and the reason this representation is affordable at all.
        """
        blocks = self.sys.split(codes)
        total = codes.new_zeros(())
        denom = mask.sum().clamp_min(1.0)
        for k, blk in enumerate(blocks):
            lp = torch.log_softmax(blk, dim=-1)
            picked = lp.gather(-1, targets[..., k:k + 1]).squeeze(-1)   # (B, L)
            total = total - (picked * mask).sum() / denom
        return total / len(blocks)
