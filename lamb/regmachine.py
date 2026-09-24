"""A differentiable register machine over the residue algebra.

:mod:`lamb.alu` composes an answer from the latent slots using the expression tree
**read from the input**. That is legitimate for synthetic arithmetic -- the tree is
the question, not the key -- but it is exactly what a natural-language problem does
not come with, and it is therefore the wall between this model and any benchmark
whose input is prose.

So the structure stops being given and starts being *emitted*. The latents become
**registers** holding values, and the model produces a short **program** over them:
at each step, an operation and two pointers to registers it has already written.
The algebra executes the program. Nothing about the execution is learned, and
nothing about the program is decoded into tokens.

    registers 0 .. N-1      the problem's operands, loaded exactly
    instruction t           (op, ptr_a, ptr_b) -> writes register N+t
    answer                  the last register written

Registers are append-only -- instruction ``t`` writes to ``N+t`` and can read only
registers ``< N+t``. That makes the dataflow a DAG by construction, so there are no
write conflicts to resolve and no way to read a register that does not exist yet.
The pointer mask enforces it rather than a loss encouraging it.

Execution is differentiable in the way that matters: a register read is a *mixture*
over registers weighted by the pointer distribution, and an operation is a mixture
over the three composed results. Both are mixtures of probability distributions, so
both are probability distributions, and a model that is unsure which register it
means composes that uncertainty instead of having to commit to one first. Gradients
therefore reach the pointers from the answer, through exact arithmetic.

This is the piece that distinguishes the design from PAL/Program-of-Thought, whose
interpreter is an external Python process: a non-differentiable executor can only be
trained through imitation or RL, and RL on this model's latent block was measured
inert (ROADMAP 3a-ii). It shares its shape with the 2026 differentiable-executor
line -- arXiv:2604.18907 learns programs through a *neural* executor with
Gumbel-Softmax, arXiv:2606.09930 differentiates an interpreter wholesale -- and
differs in that this executor is exact algebra rather than a learned approximation,
so it cannot itself be wrong.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .algebra import RedundantResidueSystem, ResidueAlgebra, ResidueSystem
from .alu import Tree, is_leaf

OPS: Tuple[str, ...] = ("+", "-", "*")
# With rational registers the instruction set closes under division. Kept separate
# from OPS so the integer path -- which carries the depth-2 result in ROADMAP
# 3a-vii -- stays byte-for-byte what produced it.
RATIONAL_OPS: Tuple[str, ...] = ("+", "-", "*", "/")
Instr = Tuple[int, int, int]          # (op index, register a, register b)


def operands(t: Tree) -> List[int]:
    """The tree's literal operands, left to right -- the initial register file."""
    if is_leaf(t):
        _, a, b = t
        return [a, b]
    _, l, r = t
    return operands(l) + operands(r)


def gold_program(t: Tree, ops: Tuple[str, ...] = OPS) -> Tuple[List[Instr], int, int]:
    """``(instructions, n_operands, output register)`` in post-order.

    Post-order is a valid topological order of the dataflow, and it is the order the
    grammar's own trace uses, so instruction ``i`` writes precisely trace entry
    ``i`` and the final instruction writes the answer. That alignment is what lets
    one supervision signal cover the program and the values at once -- and it is
    free, because the generator knows the program it generated.
    """
    n_operands = len(operands(t))
    instrs: List[Instr] = []
    cursor = 0

    def walk(sub: Tree) -> int:
        nonlocal cursor
        if is_leaf(sub):
            op, _a, _b = sub
            ra, rb = cursor, cursor + 1
            cursor += 2
        else:
            op, l, r = sub
            ra, rb = walk(l), walk(r)
        dest = n_operands + len(instrs)
        instrs.append((ops.index(op), ra, rb))
        return dest

    out = walk(t)
    return instrs, n_operands, out


class RegisterFile:
    """Register contents as one residue-distribution block per modulus.

    ``blocks[k]`` is ``(B, R, p_k)``. Kept split rather than flat because execution
    composes block by block and the flat/split conversion has no business sitting
    inside the loop.
    """

    def __init__(self, sysm: ResidueSystem, values: Sequence[Sequence[int]],
                 device: str = "cpu", sharp: float = 30.0,
                 alg: "Optional[ResidueAlgebra]" = None, n_total: Optional[int] = None):
        self.sys = sysm
        self.alg = alg or ResidueAlgebra(sysm, device=device)
        b, n = len(values), len(values[0])
        P = max(sysm.moduli)
        # Registers are held **packed** as (B, R, K, P), one row per modulus padded
        # to the widest. The per-modulus list is the obvious representation and the
        # wrong one on a GPU: it turns every read and every composition into one
        # tiny kernel per modulus, and the arithmetic in a composition is ~0.001 us
        # against a 5-10 us launch. Packing pays ~70% wasted arithmetic and about
        # 5x the intermediate memory -- 10 MB rather than 2 at batch 256, nothing on
        # any real card -- to cut the kernel count by 7-9x along the hot path.
        # The register file is **preallocated** to its final width and never grows.
        #
        # Appending with torch.cat changes the shape on every instruction, and a
        # changing shape is the one thing compilers cannot fuse across: XLA requires
        # static shapes outright and would recompile per length, and TorchInductor
        # traces it but cannot fuse over the boundary. Since the whole workload is
        # launch-bound -- the arithmetic in a composition is ~0.001 us against a
        # 5-10 us launch -- losing fusion is the expensive part, not the allocation.
        #
        # Writes are out-of-place against a slot indicator rather than in-place
        # assignment, which keeps the shape static *and* keeps autograd happy; an
        # in-place write into a tensor on the graph trips the version counter.
        self.n_total = n_total if n_total is not None else n
        self.n_filled = n
        full = torch.full((b, self.n_total, len(sysm.moduli), P), -sharp, device=device)
        for i, row in enumerate(values):
            for j, v in enumerate(row):
                for k, p in enumerate(sysm.moduli):
                    full[i, j, k, int(v) % p] = sharp
        self.packed = torch.softmax(full, dim=-1) * self._valid(device)
        if self.n_total > n:                      # unwritten slots hold nothing
            keep = torch.zeros(self.n_total, 1, 1, device=device)
            keep[:n] = 1.0
            self.packed = self.packed * keep

    def _valid(self, device) -> torch.Tensor:
        return self.alg._tables(torch.device(device))[3]        # (K, P) padding mask

    @property
    def blocks(self) -> List[torch.Tensor]:
        """The per-modulus view, for callers that want it. Narrowing is free."""
        return [self.packed[..., k, :p] for k, p in enumerate(self.sys.moduli)]

    def append(self, new: torch.Tensor) -> None:
        """Write ``new`` ``(B, K, P)`` into the next slot, keeping the shape fixed."""
        slot = torch.zeros(self.n_total, 1, 1, device=new.device, dtype=new.dtype)
        slot[self.n_filled] = 1.0
        self.packed = self.packed + slot * new.unsqueeze(1)
        self.n_filled += 1

    def read(self, ptr: torch.Tensor) -> torch.Tensor:
        """Pointer-weighted mixture over registers. ``ptr`` ``(B, R)`` -> ``(B, K, P)``."""
        return torch.einsum("br,brkp->bkp", ptr, self.packed)

    def decode(self, index: int) -> List[int]:
        picks = self.packed[:, index].argmax(-1)                 # (B, K)
        return [self.sys.crt([int(picks[i, k]) % p
                              for k, p in enumerate(self.sys.moduli)])
                for i in range(picks.size(0))]

    def residues_at(self, index: int) -> List[List[int]]:
        """The argmax residue vector per row -- what a checker needs to see."""
        picks = self.packed[:, index].argmax(-1)                 # (B, K)
        return [[int(picks[i, k]) % p for k, p in enumerate(self.sys.moduli)]
                for i in range(picks.size(0))]

    def decode_checked(self, index: int) -> List[Tuple[Optional[int], Optional[int]]]:
        """Decode through the redundant range check: ``(value, faulty modulus)``.

        Plain :meth:`decode` is a bare CRT, which has no locality -- one wrong
        residue does not give a nearby number, it gives an essentially uniform one,
        so a single flipped residue is a *wildly* wrong answer that looks exactly
        like a right one. Sizing the core moduli so legitimate values occupy part of
        the ring makes that detectable, and dropping each modulus in turn identifies
        and repairs it.

        The same check covers a failure that has nothing to do with residue errors.
        A soft pointer read is not a blend: decoding takes an argmax *per modulus,
        independently*, so the winning residues need not come from the same register,
        and when they disagree the CRT lands somewhere unrelated to either input
        (~30% of 50/50 reads). An incoherent vector is not a legitimate value either,
        so the range check flags it without caring what made it inconsistent.

        ``(None, None)`` is a **refusal**, not a guess. Where the evidence does not
        single out a culprit this returns nothing rather than the most plausible
        candidate, because this representation is used precisely where nothing
        downstream can catch a wrong answer.
        """
        if not isinstance(self.sys, RedundantResidueSystem):
            raise TypeError("decode_checked needs a RedundantResidueSystem; a plain "
                            "system has no spare moduli to check against")
        return [self.sys.correct(res) for res in self.residues_at(index)]


def execute(alg: ResidueAlgebra, regs: RegisterFile, op_w: torch.Tensor,
            ptr_a: torch.Tensor, ptr_b: torch.Tensor,
            ops: Tuple[str, ...] = OPS) -> torch.Tensor:
    """One instruction. ``op_w`` ``(B, n_ops)``; pointers ``(B, R)``, all normalised.

    Returns the packed ``(B, K, P)`` result. The operation is a mixture over the
    composed results rather than a choice between them, so an undecided model still
    produces a well-formed value and the gradient can tell it which way to move. All
    compositions run batched over moduli, so an instruction costs a few kernels'
    worth of algebra rather than a few times seven.

    ``ops`` is threaded through rather than read from the module. Reading the
    module-level ``OPS`` is silently wrong the moment the operation head is any
    width other than three: the head would emit four logits and only the first
    three would ever be composed, so the fourth operation would be trainable,
    selectable, and a no-op.
    """
    a, b = regs.read(ptr_a), regs.read(ptr_b)
    out: Optional[torch.Tensor] = None
    for oi, op in enumerate(ops):
        comp = alg.compose_packed(a, b, op)
        w = op_w[:, oi].view(-1, 1, 1)
        out = w * comp if out is None else out + w * comp
    return out


def causal_mask(n_slots: int, n_operands: int, step: int, device: str = "cpu"
                ) -> torch.Tensor:
    """``(n_slots,)`` -- instruction ``step`` may read registers ``< n_operands+step``.

    Enforced rather than encouraged: a program that reads a register it has not
    written is not a worse program, it is not a program.
    """
    m = torch.full((n_slots,), float("-inf"), device=device)
    m[: n_operands + step] = 0.0
    return m


def row_causal_mask(n_slots: int, n_operands: int, n_instr: int,
                    counts: torch.Tensor) -> torch.Tensor:
    """``(B, n_instr, n_slots)`` -- the same rule, but per row.

    The grammar hands every problem exactly ``2**depth`` operands, so one mask
    serves the batch. A word problem does not: the register file is padded to a
    fixed width and most rows leave part of it empty, and a pointer into an empty
    slot is not a worse program, it is not a program. ``registers_from_quantities``
    has returned the real count since it was written and nothing consumed it.

    Note the readable set is not a prefix: instruction ``t`` may read the row's own
    operands ``[0, count)`` and the results written so far ``[n_operands, n_operands
    + t)``, but *not* the padding in between. Masking a prefix ``[0, count + t)``
    instead would be subtly wrong in the direction that hides itself -- it would
    make the padding readable and the early results unreachable.
    """
    dev = counts.device
    j = torch.arange(n_slots, device=dev).view(1, 1, -1)
    t = torch.arange(n_instr, device=dev).view(1, -1, 1)
    # A row with no real operands has no legal pointer at t=0 and would softmax to
    # NaN, so the floor is one slot rather than an exception; with constants preloaded
    # the count is never actually zero.
    c = counts.clamp_min(1).view(-1, 1, 1)
    ok = (j < c) | ((j >= n_operands) & (j < n_operands + t))
    return torch.where(ok, torch.zeros((), device=dev),
                       torch.full((), float("-inf"), device=dev))


def run_gold(sysm: ResidueSystem, alg: ResidueAlgebra, trees: Sequence[Tree],
             device: str = "cpu", ops: Tuple[str, ...] = OPS) -> List[int]:
    """Execute each tree's own gold program. The oracle the neural version is
    measured against, and the check that the semantics are right at all."""
    progs = [gold_program(t, ops=ops) for t in trees]
    n_op = progs[0][1]
    n_instr = len(progs[0][0])
    n_slots = n_op + n_instr                      # the file's final, fixed width
    regs = RegisterFile(sysm, [operands(t) for t in trees], device, n_total=n_slots)
    b = len(trees)
    for step in range(n_instr):
        op_w = torch.zeros(b, len(ops), device=device)
        # Pointers span the whole file; slots past the written tail are simply zero,
        # which is what the causal mask produces in the learned path too.
        pa = torch.zeros(b, n_slots, device=device)
        pb = torch.zeros(b, n_slots, device=device)
        for i, (instrs, _, _) in enumerate(progs):
            oi, ra, rb = instrs[step]
            op_w[i, oi] = 1.0
            pa[i, ra] = 1.0
            pb[i, rb] = 1.0
        regs.append(execute(alg, regs, op_w, pa, pb, ops))
    return regs.decode(progs[0][2])


class RegisterMachine(nn.Module):
    """Emit a program from latent states and execute it in the algebra.

    One latent position per instruction. Three heads read it: which operation, and
    which two registers to read. Pointers are masked to registers that already
    exist, so an ill-formed program is unrepresentable rather than merely penalised.

    ``tau`` and ``hard`` expose Gumbel-Softmax on the discrete choices
    (arXiv:2604.18907's device for exactly this problem): a straight-through sample
    keeps the forward pass discrete -- so the executor sees a real program rather
    than a blur of three -- while the backward pass stays differentiable. It is off
    by default, because a soft mixture is the easier optimisation and whether the
    discreteness is needed is a question to measure, not to assume.
    """

    def __init__(self, d_model: int, n_operands: int, n_instr: int,
                 system: Optional[ResidueSystem] = None, device: str = "cpu",
                 rational: bool = False):
        super().__init__()
        self.rational = rational
        if rational:
            # Rational registers close the instruction set under division, which
            # plain residues cannot offer: an inverse needs a divisor coprime to
            # every modulus, and the short-digit-period moduli are exactly the set
            # that denies that (ROADMAP 3a-viii). The ring is larger because
            # denominators multiply.
            from .rational import RationalAlgebra

            self.ralg = RationalAlgebra(system, device=device)
            self.sys = self.ralg.sys
            self.alg = self.ralg.alg
        else:
            self.sys = system or ResidueSystem()
            self.alg = ResidueAlgebra(self.sys, device=device)
            self.ralg = None
        self.ops = RATIONAL_OPS if rational else OPS
        self.n_operands = n_operands
        self.n_instr = n_instr
        self.n_slots = n_operands + n_instr
        self.op_head = nn.Linear(d_model, len(self.ops))
        self.ptr_a = nn.Linear(d_model, self.n_slots)
        self.ptr_b = nn.Linear(d_model, self.n_slots)

    def logits(self, latent_h: torch.Tensor,
               counts: Optional[torch.Tensor] = None):
        """``(B, n_instr, d)`` -> op, ptr-a and ptr-b logits, pointers masked.

        ``counts`` is the per-row number of *real* operand registers. Without it the
        whole operand block is readable, which is right when the generator hands
        every problem the same number of operands and wrong for a word problem,
        whose register file is padded.
        """
        h = latent_h[:, : self.n_instr]
        op = self.op_head(h)
        a, b = self.ptr_a(h), self.ptr_b(h)
        dev = latent_h.device
        if counts is None:
            mask = torch.stack([causal_mask(self.n_slots, self.n_operands, t, dev)
                                for t in range(self.n_instr)])      # (n_instr, slots)
        else:
            mask = row_causal_mask(self.n_slots, self.n_operands, self.n_instr,
                                   counts.to(dev))                  # (B, n_instr, slots)
        return op, a + mask, b + mask

    @staticmethod
    def _select(logits: torch.Tensor, tau: float, hard: bool) -> torch.Tensor:
        if tau <= 0:
            return torch.softmax(logits, dim=-1)
        return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

    def run(self, latent_h: torch.Tensor, values: Sequence[Sequence[int]],
            tau: float = 0.0, hard: bool = False,
            counts: Optional[torch.Tensor] = None):
        """Execute the emitted program. Returns ``(register file, logits)``.

        The operands are loaded exactly -- they are digits in the prompt and the
        digit->residue map is a known fixed function (ROADMAP 3a-vi: asking the
        network to induce it leaves it at chance for 1200 steps). So nothing here
        is spent on arithmetic the model should not be learning, and the only thing
        being tested is whether it can emit the right *program*.
        """
        op_l, a_l, b_l = self.logits(latent_h, counts)
        if self.rational:
            vals = [[v if isinstance(v, Fraction) else Fraction(int(v)) for v in row]
                    for row in values]
            regs = RationalRegisterFile(self.ralg, vals, n_total=self.n_slots,
                                        device=str(latent_h.device))
        else:
            regs = RegisterFile(self.sys, values, str(latent_h.device),
                                n_total=self.n_slots)
        for t in range(self.n_instr):
            # Softmax over the *full* register file rather than a slice. The causal
            # mask is already -inf past the written tail, so those entries come out
            # exactly zero -- same result, static shape.
            ow = self._select(op_l[:, t], tau, hard)
            pa = self._select(a_l[:, t], tau, hard)
            pb = self._select(b_l[:, t], tau, hard)
            regs.append(execute_rational(self.ralg, regs, ow, pa, pb) if self.rational
                        else execute(self.alg, regs, ow, pa, pb, self.ops))
        return regs, (op_l, a_l, b_l)

    def program_loss(self, logits, gold: Sequence[Sequence[Instr]],
                     keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-entropy on the emitted program against the generator's own.

        The gold program is free: the grammar knows the expression it built, so
        every problem carries its exact program as well as its exact answer. That is
        what makes supervising a program affordable here and not elsewhere, and it
        is the curriculum that avoids the instability neural program induction is
        known for -- supervise first, relax after.

        ``keep`` is the out-of-ring row mask. The gold program is still *correct* on
        a row whose value the ring cannot hold, so masking it here is not about
        correctness -- it is so that the ``program_coef=1`` and ``program_coef=0``
        arms train on exactly the same rows. Two arms that differ in which problems
        they see are not a controlled comparison, and that is the shape of confound
        this project has already retracted claims over.
        """
        op_l, a_l, b_l = logits
        dev = op_l.device
        o = torch.tensor([[i[0] for i in g] for g in gold], device=dev)
        a = torch.tensor([[i[1] for i in g] for g in gold], device=dev)
        b = torch.tensor([[i[2] for i in g] for g in gold], device=dev)
        ce = torch.nn.functional.cross_entropy

        def term(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            flat = ce(logit.reshape(-1, logit.size(-1)), target.reshape(-1),
                      reduction="none").view(target.shape)          # (B, n_instr)
            if keep is None:
                return flat.mean()
            w = keep.to(flat.dtype).view(-1, 1).expand_as(flat)
            return (flat * w).sum() / w.sum().clamp_min(1.0)

        return (term(op_l, o) + term(a_l, a) + term(b_l, b)) / 3.0


class RegMachineTrainer:
    """Train the latent core to emit programs; the algebra executes them.

    Wraps :class:`lamb.lotus.LotusTrainer` for the latent block, the hash-partitioned
    problem split and the sampling, and replaces what sits on top: instead of
    predicting the answer's tokens, the latents emit a program, and the answer is
    whatever executing it produces.

    Two losses. ``program`` supervises the emitted instructions against the
    generator's own -- free, because the grammar knows the expression it built.
    ``answer`` is the residue cross-entropy of the executed result, which reaches the
    program heads *through the arithmetic*, so a program can be corrected by the
    answer being wrong even where the gold program is not consulted. Weighting them
    is the curriculum: supervise the program first, lean on the answer after.

    ``rational=True`` swaps the register file for ``(numerator, denominator)`` pairs
    and widens the instruction set to include ``/``. It is opt-in because the integer
    path carries the depth-2 result of ROADMAP 3a-vii and should stay byte-for-byte
    what produced it.
    """

    def __init__(self, cfg, tokenizer, model_cfg=None, program_coef: float = 1.0,
                 answer_coef: float = 1.0, tau: float = 0.0, hard: bool = False,
                 rational: bool = False, den_zero_coef: float = 1.0,
                 redundant_moduli: Optional[Sequence[int]] = None,
                 program_anneal: Optional[Tuple[int, int]] = None,
                 program_frac: float = 1.0, n_instr: Optional[int] = None,
                 entropy_coef: float = 0.0, entropy_target: float = 0.1):
        from .lotus import LotusTrainer

        self.inner = LotusTrainer(cfg, tokenizer, model_cfg)
        self.cfg = cfg
        self.device = self.inner.device
        # A balanced tree of depth D needs exactly 2^D - 1 instructions, and deriving
        # that meant this trainer could never express a program *shorter* than its
        # budget -- so the one thing the language bridge does universally (a fixed
        # 8-instruction budget over a median-3 chain, padded with ``x * 1``) was
        # untestable on the only task where ground truth exists.
        #
        # Over-provisioning is not free in principle: each spare instruction widens the
        # pointer softmax, adds a register, and gives the model another way to be wrong
        # on a problem needing none of them. Whether it is free in practice is a
        # measurement, and this is what makes it one.
        self.gold_instr = 2 ** cfg.depth - 1
        self.n_instr = int(n_instr) if n_instr else self.gold_instr
        if self.n_instr < self.gold_instr:
            raise ValueError(f"n_instr={self.n_instr} cannot hold the "
                             f"{self.gold_instr} instructions depth {cfg.depth} needs")
        # ``x * 1`` is the pad, so a spare budget needs a register holding 1. It is
        # *prepended*, not written over an operand -- the gold program's pointers index
        # the tree's own operands, and overwriting one would silently change the problem
        # while leaving the program that references it intact. Prepending shifts every
        # index by exactly one, operands and results alike, because results begin
        # immediately after the operand block in both layouts.
        self.n_const = 1 if self.n_instr > self.gold_instr else 0
        self.n_operands = self.n_const + 2 ** cfg.depth
        if cfg.n_latent < self.n_instr:
            raise ValueError(f"n_latent={cfg.n_latent} < {self.n_instr} instructions "
                             f"needed at depth {cfg.depth}")
        d = self.inner.reasoner.model.cfg.d_model
        self.rational = rational
        # Rationals need their own ring: denominators multiply and never reduce, so
        # the integer ``alu_moduli`` (~1.7e6) is exhausted by a couple of divisions.
        if rational:
            from .rational import RATIONAL_MODULI

            moduli = tuple(getattr(cfg, "rational_moduli", None) or RATIONAL_MODULI)
        else:
            moduli = tuple(cfg.alu_moduli)
        # Redundant moduli are opt-in and explicit: the caller names them, because
        # the sizing is a real trade (three redundant moduli fully correct single
        # errors; two lose 16% of corrections to refusal) and there is no default
        # that is right for every core. ``RedundantResidueSystem`` refuses to
        # guess for the same reason.
        self.redundant_moduli = tuple(redundant_moduli or ())
        system = (RedundantResidueSystem(moduli + self.redundant_moduli,
                                         n_core=len(moduli))
                  if self.redundant_moduli else ResidueSystem(moduli))
        self.machine = RegisterMachine(d, self.n_operands, self.n_instr, system,
                                       device=str(self.device),
                                       rational=rational).to(self.device)
        self.program_coef, self.answer_coef = program_coef, answer_coef
        self.den_zero_coef = den_zero_coef
        # The curriculum this class's own docstring describes -- "supervise the program
        # first, lean on the answer after" -- and which no arm had ever run. Both arms
        # in 3a-xii are constant extremes, 1.0 or 0.0, and the interesting regime is
        # neither: 3a-xii found outcome-only induction fails at 7 instructions, while
        # the supervised arm is perfect, so the question that matters is whether the
        # answer loss can *take over* once the pointers have committed.
        # Penalise an undecided program. The failing seeds in 3a-xii are not wrong, they
        # are *uncommitted*: every seed reaching ptr_sharp 1.0 scored 1.000, every seed
        # below it scored ~0.04, with nothing in between. A blurred pointer is not a
        # blend either -- decoding argmaxes each modulus independently, so a soft read
        # lands in neither register ~30% of the time (3a-x). So indecision is not a
        # smaller version of the right answer, it is its own failure mode, and this is
        # the term that prices it.
        #
        # The obvious risk, which is what the arm measures: pressure to commit is
        # pressure to commit *early*, and an early commitment to the wrong program is
        # worse than an undecided one that the answer loss could still have moved.
        self.entropy_coef = float(entropy_coef)
        self.entropy_target = float(entropy_target)
        self.program_anneal = tuple(program_anneal) if program_anneal else None
        # Partial supervision, which is the language bridge's actual condition rather
        # than a hypothetical: GSM8K's calculator annotations recover a program for 42%
        # of the train split and nothing for the rest (3c). Membership is decided by a
        # hash of the problem, not sampled per step, because that is how a dataset
        # behaves -- a problem either has an annotation or it never does, and per-step
        # dropout would quietly give every problem supervision eventually.
        self.program_frac = float(program_frac)
        self.tau, self.hard = tau, hard
        self.opt = torch.optim.AdamW(
            list(self.inner.reasoner.parameters()) + list(self.machine.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _program_coef_at(self, step: int) -> float:
        """``program_coef``, annealed if a window was given. Linear 1 -> 0 over it."""
        if self.program_anneal is None:
            return self.program_coef
        a, b = self.program_anneal
        if step <= a:
            return self.program_coef
        if step >= b:
            return 0.0
        return self.program_coef * (1.0 - (step - a) / max(1, b - a))

    def _has_program(self, tasks) -> List[bool]:
        """Which rows carry a gold program, decided by the problem and not the step."""
        if self.program_frac >= 1.0:
            return [True] * len(tasks)
        if self.program_frac <= 0.0:
            return [False] * len(tasks)
        out = []
        for expr, _, _ in tasks:
            h = hashlib.blake2b(expr.encode("utf-8"), digest_size=8).digest()
            out.append((int.from_bytes(h, "big") % 1000) < self.program_frac * 1000)
        return out

    # -- the ring ----------------------------------------------------------
    def _in_ring(self, answer: int, trace: Sequence[int], vals: Sequence[int]) -> bool:
        """Is every value this row needs representable?

        **Out-of-range values are masked, never clipped.** In a residue ring a
        wrapped value is a *different* number, not a large one, so
        ``ResidueSystem.targets`` -- which is an unguarded ``int(v) % p`` -- turns an
        out-of-ring answer into a perfectly legal target for the wrong number, and
        the model is then trained on arithmetic that is false. :mod:`lamb.lotus`
        has masked on this since the ALU landed; this trainer did not, and it went
        unnoticed because depth 2 at 1 digit over ``(+,-)`` never leaves the ring.
        Depth 3, or multiplication, leaves it immediately.

        For rationals this is **necessary but not sufficient**: a fraction cannot be
        reduced in residue form, so the *unreduced* numerator and denominator the
        program actually builds are larger than anything checked here and depend on
        the program. :meth:`evaluate` reports ``max_den_magnitude`` as the
        sufficient-side monitor, so a chain approaching the ring is observed rather
        than discovered from a wrong answer.
        """
        sysm = self.machine.sys
        if isinstance(sysm, RedundantResidueSystem):
            # With redundancy the usable range is the *core*'s, not the whole ring:
            # detection works precisely because legitimate values leave the rest of
            # the ring empty, so a value that is merely ``representable`` would be
            # indistinguishable from a corrupted one. Using ``representable`` here
            # would silently disarm the checker rather than fail.
            return all(abs(int(v)) <= sysm.legit for v in (*vals, *trace, answer))
        return all(sysm.representable(int(v))
                   for v in (*vals, *trace, answer))

    # -- one batch ---------------------------------------------------------
    def _prepare(self, tasks):
        from .alu import parse_expr

        trees = [parse_expr(e) for e, _, _ in tasks]
        # The instruction set is the *machine's*, not the module's. Defaulting to the
        # integer ``OPS`` here is what made the grammar's own division output
        # (``ops_key=2``) unconsumable: ``OPS.index("/")`` raises, so the only
        # program trainer in the repo crashed on data that generates correctly.
        golds = [gold_program(t, ops=self.machine.ops)[0] for t in trees]
        vals = [operands(t) for t in trees]
        if self.n_const:
            k = self.n_const
            golds = [[(op, ra + k, rb + k) for op, ra, rb in g] for g in golds]
            vals = [[1] + v for v in vals]
            mul = self.machine.ops.index("*")
            for g in golds:
                while len(g) < self.n_instr:
                    # multiply the last written register through the constant 1
                    g.append((mul, self.n_operands + len(g) - 1, 0))
        answers = [int(a) for _, a, _ in tasks]
        keep = [self._in_ring(int(a), tr, v)
                for (_, a, tr), v in zip(tasks, vals)]
        return trees, golds, vals, answers, keep

    def _latents(self, tasks):
        prompt, *_ = self.inner._collate(tasks)
        m = self.inner.reasoner.model
        x = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                    prompt["value"], prompt["value_mask"])
        _, _, latent_h, _ = self.inner.reasoner.latent_block(x, prompt["pad_mask"])
        return latent_h

    # -- the answer loss ---------------------------------------------------
    def _answer_loss_integer(self, regs, out_reg, answers, keep):
        tgt = self.machine.sys.targets(answers, device=str(self.device))
        per_mod = [torch.nn.functional.nll_loss(
                       torch.log(blk[:, out_reg].clamp_min(1e-9)), tgt[:, k],
                       reduction="none")
                   for k, blk in enumerate(regs.blocks)]
        row = torch.stack(per_mod).mean(0)                       # (B,)
        return (row * keep).sum() / keep.sum().clamp_min(1.0), {}

    def _answer_loss_rational(self, regs, out_reg, answers, keep):
        return rational_answer_loss(self.machine, regs, out_reg,
                                    [Fraction(int(a)) for a in answers], keep,
                                    self.den_zero_coef)

    def _losses(self, tasks):
        trees, golds, vals, answers, keep = self._prepare(tasks)
        latent_h = self._latents(tasks)
        regs, logits = self.machine.run(latent_h, vals, self.tau, self.hard)
        k = torch.tensor(keep, dtype=torch.float32, device=self.device)
        # Rows outside the ring are dropped from *both* losses so the arms see the same
        # data; rows without a gold program are dropped from the program loss only,
        # since the answer loss is exactly what has to carry them.
        sup = torch.tensor([a and b for a, b in zip(keep, self._has_program(tasks))],
                           dtype=torch.float32, device=self.device)
        prog = self.machine.program_loss(logits, golds, sup)
        out_reg = self.n_operands + self.n_instr - 1
        fn = (self._answer_loss_rational if self.rational
              else self._answer_loss_integer)
        ans, extra = fn(regs, out_reg, answers, k)
        if self.entropy_coef:
            # **Floored, because an unfloored version diverges.** Minimising entropy has
            # no lower bound in logit space: driving p toward one-hot drives the logits
            # toward +-inf and they never stop. Measured -- the first version of this
            # term sent *every* logit to inf within 40 steps, after which softmax
            # returns nan and ``ptr_sharp`` reports nan, which is how the arm announced
            # it rather than by failing.
            #
            # Penalising only the excess over a small target removes the gradient once a
            # distribution is committed enough, so the logits stop growing. It also
            # makes the coefficient mean something: without a floor the term's scale is
            # set by how far the logits have already run.
            ent = torch.zeros((), device=self.device)
            for x in logits:                       # op, ptr-a, ptr-b
                # **Clamp, do not nan_to_num.** A masked slot has ``lp = -inf``, so
                # ``p * lp`` is ``0 * -inf = nan``. ``nan_to_num`` repairs the *forward*
                # value and does nothing to the backward, so a nan gradient reached the
                # optimiser and destroyed every weight at step 0 -- while the printed
                # loss stayed finite and the entropy term reported 0.0000, because
                # ``nan_to_num`` was also swallowing the evidence. Clamping keeps both
                # passes finite; the masked terms then contribute ~1e-13 * -30, which is
                # nothing.
                lp = torch.log_softmax(x, dim=-1).clamp_min(-30.0)
                h = -(lp.exp() * lp).sum(-1)
                ent = ent + torch.relu(h - self.entropy_target).mean()
            ans = ans + self.entropy_coef * ent / 3.0
            extra["ptr_entropy"] = float(ent.detach() / 3.0)
        return {"prog": prog, "ans": ans, "extra": extra, "regs": regs,
                "logits": logits, "golds": golds, "answers": answers,
                "out_reg": out_reg, "keep": keep, "vals": vals}

    def train_step(self, step: int):
        self.inner.reasoner.train()
        self.machine.train()
        tasks = self.inner._sample_batch(self.cfg.batch_size)
        for g in self.opt.param_groups:
            g["lr"] = self.inner._lr_at(step)
        b = self._losses(tasks)
        prog, ans = b["prog"], b["ans"]
        pc = self._program_coef_at(step)
        loss = pc * prog + self.answer_coef * ans
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        # Fail loudly rather than train on wreckage. A nan gradient turns every weight
        # nan on the next step, after which the forward still runs, the loss still
        # prints a number, and ``evaluate`` still returns an accuracy -- decoded from a
        # destroyed model. One arm of the commitment study reported a statistically
        # significant +0.013 that way. A silent void result is worse than a crash.
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        for nm, prm in (("reasoner", self.inner.reasoner), ("machine", self.machine)):
            for q in prm.parameters():
                if q.grad is not None and not torch.isfinite(q.grad).all():
                    raise FloatingPointError(
                        f"non-finite gradient in {nm} at step {step}")
        torch.nn.utils.clip_grad_norm_(
            list(self.inner.reasoner.parameters()) + list(self.machine.parameters()),
            self.cfg.grad_clip)
        self.opt.step()
        out = {"loss": float(loss.detach()), "program": float(prog.detach()),
               "answer": float(ans.detach()), "program_coef": pc,
               # Reported, not just applied: a configuration that spends its batch
               # outside the ring should be visible in the log rather than inferred
               # from a bad number later.
               "dropped": 1.0 - sum(b["keep"]) / max(1, len(b["keep"]))}
        out.update(b["extra"])
        return out

    # -- self-knowledge ----------------------------------------------------
    @torch.no_grad()
    def self_consistency(self, n_tasks: int = 256, n_samples: int = 8,
                         tau: float = 1.0) -> Dict[str, float]:
        """Does the model know when its own program is wrong? No labels, no verifier.

        Redundant residues detect an *ill-formed code* and cannot detect a well-formed
        code for the wrong program -- measured in 3a-xiii, where they converted 1.3% of
        errors into refusals and left 44.6% as confident wrong numbers. That gap is the
        whole remaining safety budget of a design that deliberately never exposes its
        reasoning: if you are not going to read the thought, the system has to tell you
        when not to trust it.

        The signal here is **agreement under resampling**. The heads already expose
        Gumbel-Softmax, so ``n_samples`` discrete programs can be drawn from the model's
        own distribution and each executed exactly. A peaked distribution samples the
        same program every time; a flat one does not. This is the per-problem,
        per-emitter version of ``ptr_sharp``, which at the arm level already separates
        the two outcomes perfectly (sharp -> 1.000, blurred -> ~0.04) -- but unlike
        ``ptr_sharp`` it needs no access to the logits, so it transfers to any emitter,
        and unlike the exact verifier it needs no ground truth, so it survives leaving
        the distribution the verifier was written for.

        Returns the numbers that decide whether it is usable. ``acc_agreed`` against
        ``acc_split`` is the separation; ``coverage`` is what fraction you keep if you
        refuse whenever the samples disagree. A signal that refuses everything is not a
        signal, so both have to be read together.
        """
        self.inner.reasoner.eval()
        self.machine.eval()
        tasks = self.inner._eval_set(n_tasks)
        trees, golds, vals, answers, keep = self._prepare(tasks)
        latent_h = self._latents(tasks)
        out_reg = self.n_operands + self.n_instr - 1

        # ``hard=True`` so each draw is a real program rather than a blur of several;
        # a soft mixture would agree with itself trivially and measure nothing.
        votes: List[List[Optional[Fraction]]] = []
        for _ in range(n_samples):
            regs, _ = self.machine.run(latent_h, vals, tau=tau, hard=True)
            got, _ = self._decode_answers(regs, out_reg)
            votes.append(got)

        n_ok = n_agree = n_ok_agree = n_split = n_ok_split = 0
        for i, (a, k) in enumerate(zip(answers, keep)):
            if not k:
                continue
            col = [v[i] for v in votes]
            modal = max(set(map(str, col)), key=lambda x: list(map(str, col)).count(x))
            share = list(map(str, col)).count(modal) / len(col)
            correct = col[0] is not None and col[0] == Fraction(a)
            n_ok += int(correct)
            if share == 1.0:
                n_agree += 1
                n_ok_agree += int(correct)
            else:
                n_split += 1
                n_ok_split += int(correct)
        n = max(1, n_agree + n_split)
        # **The counts are returned because the separation is meaningless without them.**
        # ``max(1, n_split)`` makes an *empty* disagreeing set report ``acc_split = 0``,
        # so a model that agrees with itself on every single problem scores a perfect
        # separation of 1.000 while having demonstrated nothing at all. Measured: a
        # supervised depth-3 model goes 0.004 -> 1.000 between steps 40 and 80, and every
        # row after that printed separation 1.000 over zero split rows. A confidence
        # signal can only be evaluated where the model is competent *and* fallible, and
        # ``n_split`` is what says whether it was.
        return {
            "acc": n_ok / n,
            "coverage": n_agree / n,                       # kept if you refuse on split
            "acc_agreed": n_ok_agree / max(1, n_agree),    # precision of the kept set
            "acc_split": n_ok_split / max(1, n_split),     # what refusal throws away
            "separation": (n_ok_agree / max(1, n_agree)) - (n_ok_split / max(1, n_split)),
            "n_agree": float(n_agree), "n_split": float(n_split),
            "valid": float(n_split >= 8 and n_agree >= 8),  # enough of both to mean it
            "n_samples": float(n_samples), "tau": tau,
        }

    # -- decoding ----------------------------------------------------------
    def _decode_answers(self, regs, out_reg):
        """``(values, n_zero_denominator)``. ``None`` where the pair cannot decode.

        With redundant moduli this is the **refusal** path: rather than a bare CRT,
        which has no locality and turns one wrong residue into a wildly wrong number
        that looks like a right one, the reconstruction is range-checked, single
        errors are repaired, and ambiguous evidence returns nothing. An admitted
        failure beats a confident number wherever nothing downstream can catch it --
        which is the entire regime this representation exists for.
        """
        if self.redundant_moduli:
            got = (regs.decode_checked(out_reg) if self.rational
                   else [None if v is None else Fraction(v)
                         for v, _ in regs.decode_checked(out_reg)])
            return got, sum(1 for g in got if g is None)
        if not self.rational:
            return [Fraction(v) for v in regs.decode(out_reg)], 0
        alg, sysm = self.machine.alg, self.machine.sys
        nums = sysm.decode(alg.unpack(regs.num[:, out_reg]))
        dens = sysm.decode(alg.unpack(regs.den[:, out_reg]))
        # Decoded row by row rather than through ``RationalAlgebra.decode``: that
        # builds the whole list at once, so a single zero denominator raises and
        # takes the other 255 rows with it. A zero denominator is a wrong answer,
        # not a crash.
        got = [None if d == 0 else Fraction(n, d) for n, d in zip(nums, dens)]
        return got, sum(1 for g in got if g is None)

    @torch.no_grad()
    def evaluate(self, n_tasks: int = 256):
        """Answer accuracy, plus how far the emitted program matches the generator's.

        **``answer_acc`` is the correctness measure; the program metrics are not.**
        The program that computes a function is not unique -- 48 distinct
        three-instruction programs compute ``(a+b)+(c+d)`` exactly, of which the
        grammar emits one -- so ``canonical_acc`` measures *conformity to the
        generator's form*, not competence. A model that induces one of the other 47
        scores zero on it while being perfectly correct, which is exactly what the
        answer-only arm does.

        Both are reported because the pair is diagnostic in the direction that still
        matters: a wrong answer from a canonical program means the operands or the
        ring failed rather than the reasoning, and high canonical agreement with a
        wrong answer would mean the executor is broken. It is only the reverse
        inference -- low canonical agreement implying a wrong program -- that does
        not hold.

        ``dropped`` is the share of the evaluation set that left the ring, and it is
        reported rather than silently excluded: an accuracy measured on 60% of a
        held-out set is not the same number as one measured on all of it, and which
        it is should not have to be reconstructed by the reader.
        """
        self.inner.reasoner.eval()
        self.machine.eval()
        tasks = self.inner._eval_set(n_tasks)
        b = self._losses(tasks)
        regs, logits, golds = b["regs"], b["logits"], b["golds"]
        answers, out_reg, keep = b["answers"], b["out_reg"], b["keep"]
        got, n_zero_den = self._decode_answers(regs, out_reg)
        n_kept = max(1, sum(keep))
        acc = sum(g == Fraction(a) for g, a, k in zip(got, answers, keep)
                  if k and g is not None) / n_kept
        op_l, a_l, b_l = logits
        dev = op_l.device
        o = torch.tensor([[i[0] for i in g] for g in golds], device=dev)
        ra = torch.tensor([[i[1] for i in g] for g in golds], device=dev)
        rb = torch.tensor([[i[2] for i in g] for g in golds], device=dev)
        ok_op = (op_l.argmax(-1) == o).float()
        ok_a = (a_l.argmax(-1) == ra).float()
        ok_b = (b_l.argmax(-1) == rb).float()
        whole = (ok_op * ok_a * ok_b)
        # How committed the program is. This is not a quality measure -- it is what
        # makes several of the other numbers *readable*. A soft read is not a blend:
        # decoding argmaxes each modulus independently, so with blurred pointers the
        # winning residues come from different registers and the CRT lands nowhere
        # near either. Every decoded quantity below inherits that.
        sharp = torch.stack([torch.softmax(x, -1).max(-1).values
                             for x in (op_l, a_l, b_l)]).mean()
        # The three outcomes partition the kept rows: right, refused, or silently
        # wrong. Reporting them separately is the point of the redundancy -- an
        # accuracy alone cannot tell "it did not answer" from "it answered wrongly",
        # and those are not the same failure. ``mis_answered`` is the column that
        # matters: under-provisioned redundancy loses corrections, it never invents
        # one, so this should stay at zero and a non-zero value means the ring, not
        # the model, is the thing to look at.
        # **The same weights, read as a decided program.** ``answer_acc`` above executes
        # the *soft* program -- a mixture over operations and a mixture over registers
        # -- because that is what the loss reads and what training shapes. But a
        # mixture is not what a model would emit at inference, and worse, decoding one
        # is incoherent by construction: the argmax is taken per modulus
        # independently, so ~30% of blurred reads decode to a value in neither
        # register (3a-x). An undecided model is therefore scored on something it
        # never actually computes.
        #
        # So the argmax program is executed as a *real* program and scored too. The
        # gap between the two is not a detail: if a seed's soft score is 0.25 and its
        # hard score is 1.000, then nothing was wrong with what it learned and
        # everything was wrong with how it was read. Both are reported because only
        # the pair distinguishes "did not learn a program" from "learned one and the
        # eval blurred it", and this project has retracted five claims to
        # measurement artefacts of precisely that shape.
        op_i, a_i, b_i = op_l.argmax(-1), a_l.argmax(-1), b_l.argmax(-1)
        progs = [[(int(op_i[i, t]), int(a_i[i, t]), int(b_i[i, t]))
                  for t in range(self.n_instr)] for i in range(len(answers))]
        hard_regs = run_program(self.machine, b["vals"], progs, str(self.device))
        if self.rational:
            hn = self.machine.sys.decode(
                self.machine.alg.unpack(hard_regs.num[:, out_reg]))
            hd = self.machine.sys.decode(
                self.machine.alg.unpack(hard_regs.den[:, out_reg]))
            got_hard = [None if d == 0 else Fraction(n, d) for n, d in zip(hn, hd)]
        else:
            got_hard = [Fraction(v) for v in hard_regs.decode(out_reg)]
        acc_hard = sum(1 for g, a, k in zip(got_hard, answers, keep)
                       if k and g is not None and g == Fraction(a)) / n_kept

        refused = sum(1 for g, k in zip(got, keep) if k and g is None) / n_kept
        mis = sum(1 for g, a, k in zip(got, answers, keep)
                  if k and g is not None and g != Fraction(a)) / n_kept
        out = {"answer_acc": acc,                        # the soft-program measure
               # the same weights read as a decided program -- see above
               "answer_acc_hard": acc_hard,
               "refused": refused,
               "mis_answered": mis,
               "instr_acc": float(whole.mean()),          # per-instruction conformity
               "canonical_acc": float(whole.min(dim=1).values.mean()),
               "program_acc": float(whole.min(dim=1).values.mean()),  # kept: old name
               "op_acc": float(ok_op.mean()), "ptr_acc": float((ok_a * ok_b).mean()),
               "program_loss": float(b["prog"]), "answer_loss": float(b["ans"]),
               "ptr_sharp": float(sharp),
               "dropped": 1.0 - sum(keep) / max(1, len(keep))}
        if self.rational:
            # The sufficient-side ring monitor: denominators multiply and never
            # reduce, so on a *sharp* program this only grows, and it is how a long
            # chain announces that it is approaching the ring instead of producing a
            # wrong answer.
            #
            # **Read it with ``ptr_sharp``, or do not read it.** It is an argmax over
            # each modulus independently, so an undecided model decodes to an
            # essentially uniform value in the ring whatever the arithmetic did:
            # measured at 4 training steps it reports ~1.3e15 against a 4.5e15 ring,
            # which is not denominator growth, it is incoherence. A blurred program
            # has no denominator to report, and pretending otherwise would turn a
            # noise figure into an alarm about the ring size.
            dens = self.machine.sys.decode(
                self.machine.alg.unpack(regs.den[:, out_reg]))
            out["max_den_magnitude"] = float(max(abs(d) for d in dens))
            if not self.redundant_moduli:
                # With redundancy a ``None`` can be a refusal rather than a zero
                # denominator, and conflating the two would report a detection as a
                # defect. ``refused`` already covers that case.
                out["zero_den"] = n_zero_den / max(1, len(got))
        return out


class RationalRegisterFile:
    """Registers holding rationals: a numerator and a denominator, both packed.

    Division is the operation grade-school word problems are built on -- "half as
    many", "split among four" -- and it is not available on plain residues: an
    inverse exists only for divisors coprime to every modulus, and the moduli were
    chosen for short digit-periods, which is exactly the set that makes small
    divisors non-invertible (ROADMAP 3a-viii). Carrying ``(num, den)`` makes
    division multiplication with the operands swapped, so the instruction set closes
    without a single new primitive.

    **One caveat, and it is not the one it looks like.** A soft pointer read does
    not produce a *blend* of the registers it mixes. Decoding takes an argmax per
    modulus independently, so the winning residues need not all come from the same
    register; when they disagree the CRT lands somewhere unrelated to either input.
    Measured over random pairs, ~30% of soft reads decode to a value that was in
    neither register, and the error is one of **incoherence**, not averaging.

    This is not specific to rationals -- the integer register file has it too, for
    the same reason. Two things contain it. Sharp pointers are exact, and training
    drives pointers sharp (the depth-2 arms reached fully determined pointers, and
    both scored 1.000). And with redundant moduli an incoherent residue vector is
    not a legitimate value, so the range check of
    :class:`lamb.algebra.RedundantResidueSystem` flags it: measured, **100% of
    incoherent reads detected, none silently wrong**. The redundancy built for the
    model's own residue errors turns out to cover this as well, because the range
    check does not care what made the vector inconsistent.

    Training is unaffected either way, since the losses read the distributions
    rather than the argmax.
    """

    def __init__(self, ralg, values: Sequence[Sequence[Fraction]],
                 n_total: Optional[int] = None, device: str = "cpu",
                 sharp: float = 30.0):
        self.r = ralg
        self.sys = ralg.sys
        b, n = len(values), len(values[0])
        self.n_total = n_total if n_total is not None else n
        self.n_filled = n
        flat = [v for row in values for v in row]
        num, den = self.r.encode(flat, device)                 # (b*n, K, P)
        K, P = num.shape[-2], num.shape[-1]
        pad = self.n_total - n
        def lay(x):
            x = x.view(b, n, K, P)
            if pad:
                x = torch.cat([x, torch.zeros(b, pad, K, P, device=x.device)], dim=1)
            return x
        self.num, self.den = lay(num), lay(den)

    def append(self, new) -> None:
        slot = torch.zeros(self.n_total, 1, 1, device=new[0].device, dtype=new[0].dtype)
        slot[self.n_filled] = 1.0
        self.num = self.num + slot * new[0].unsqueeze(1)
        self.den = self.den + slot * new[1].unsqueeze(1)
        self.n_filled += 1

    def read(self, ptr: torch.Tensor):
        return (torch.einsum("br,brkp->bkp", ptr, self.num),
                torch.einsum("br,brkp->bkp", ptr, self.den))

    def decode(self, index: int) -> List[Fraction]:
        return self.r.decode((self.num[:, index], self.den[:, index]))

    def decode_checked(self, index: int) -> List[Optional[Fraction]]:
        """Both components through the redundant range check; ``None`` on refusal.

        A rational is only as trustworthy as its worse half, so a refusal on either
        the numerator or the denominator refuses the pair. A zero denominator also
        returns ``None`` rather than raising -- it is a wrong answer, not a crash,
        and it is the one thing the ring genuinely cannot detect for itself.
        """
        if not isinstance(self.sys, RedundantResidueSystem):
            raise TypeError("decode_checked needs a RedundantResidueSystem")
        alg = self.r.alg
        out: List[Optional[Fraction]] = []
        for part in ("num", "den"):
            picks = torch.stack([b.argmax(-1) for b in
                                 alg.unpack(getattr(self, part)[:, index])], dim=-1)
            vals = [self.sys.correct([int(picks[i, k]) % p
                                      for k, p in enumerate(self.sys.moduli)])[0]
                    for i in range(picks.size(0))]
            out.append(vals)                                  # type: ignore[arg-type]
        nums, dens = out                                      # type: ignore[misc]
        return [None if (n is None or d is None or d == 0) else Fraction(n, d)
                for n, d in zip(nums, dens)]


def execute_rational(ralg, regs: RationalRegisterFile, op_w: torch.Tensor,
                     ptr_a: torch.Tensor, ptr_b: torch.Tensor):
    """One instruction over rationals. ``op_w`` is ``(B, 4)`` across ``RATIONAL_OPS``."""
    a, b = regs.read(ptr_a), regs.read(ptr_b)
    out = None
    for oi, op in enumerate(RATIONAL_OPS):
        n, d = ralg.compose(a, b, op)
        w = op_w[:, oi].view(-1, 1, 1)
        out = (w * n, w * d) if out is None else (out[0] + w * n, out[1] + w * d)
    return out


def run_program(machine: "RegisterMachine", values: Sequence[Sequence[object]],
                programs: Sequence[Sequence[Instr]], device: str = "cpu"):
    """Execute *known* programs through a machine's own file and algebra.

    :func:`run_gold` is the same idea for the integer path only, and built its own
    ``RegisterFile``; the rational path had no oracle at all. That gap is part of
    why the division data path could break without a test noticing -- every
    rational test hand-fed ``Fraction`` values through hand-built pointers, so
    nothing ever executed a program the *grammar* produced.

    It is also what the language bridge needs: given a program recovered from a
    dataset's own annotations, the first question is whether executing it actually
    yields that dataset's answer, and that question has nothing to do with a model.
    """
    b, n_instr = len(programs), len(programs[0])
    if machine.rational:
        vals = [[v if isinstance(v, Fraction) else Fraction(int(v)) for v in row]
                for row in values]
        regs = RationalRegisterFile(machine.ralg, vals, n_total=machine.n_slots,
                                    device=device)
    else:
        regs = RegisterFile(machine.sys, values, device, n_total=machine.n_slots)
    for step in range(n_instr):
        op_w = torch.zeros(b, len(machine.ops), device=device)
        pa = torch.zeros(b, machine.n_slots, device=device)
        pb = torch.zeros(b, machine.n_slots, device=device)
        for i, prog in enumerate(programs):
            oi, ra, rb = prog[step]
            op_w[i, oi] = 1.0
            pa[i, ra] = 1.0
            pb[i, rb] = 1.0
        regs.append(execute_rational(machine.ralg, regs, op_w, pa, pb)
                    if machine.rational
                    else execute(machine.alg, regs, op_w, pa, pb, machine.ops))
    return regs


def rational_answer_loss(machine: "RegisterMachine", regs, out_reg: int,
                         targets: Sequence[Fraction], keep: torch.Tensor,
                         den_zero_coef: float = 1.0):
    """Supervise ``N*q - D*p = 0`` rather than ``N`` and ``D`` separately.

    A fraction cannot be reduced in residue form, so the pair a program builds is
    *not* the gold pair: ``(48/2) + 13/4`` lands on some unreduced ``(N, D)`` with
    ``N/D = 109/4``, and supervising ``N`` against ``109`` would be supervising the
    wrong number. The cross-product residual is the exact statement of "these two
    fractions are equal", it is one target -- zero -- in every modulus, and it is
    built from the ``+``, ``-``, ``*`` the ring already has.

    **It has exactly one degenerate, and it is narrower than it looks.** ``(N, D) =
    (0, 0)`` satisfies the residual for any target. A zero *divisor* alone does not
    get there: ``5 / 0`` gives ``(5, 0)``, whose residual is ``5`` and which the
    criterion rejects at full cost. Since ``a/b / c/d -> (ad, bc)``, reaching ``(0,
    0)`` needs a zero numerator *and* a zero divisor -- ``0 / 0``. That is reachable
    (one-digit operands include ``0``, and a pointer pair may address the same
    register twice), so ``den_zero_coef`` is load-bearing, but for one case rather
    than as a general necessity. ROADMAP 3a-viii's *"division by zero is undetectable
    in the ring"* met from the loss side, where the ring turns out to do better than
    expected.

    Shared by the synthetic trainer and the language bridge deliberately: two copies
    of a loss with a known degenerate is how one of them quietly loses the guard.
    """
    alg, sysm = machine.alg, machine.sys
    dev = str(keep.device)
    num, den = regs.num[:, out_reg], regs.den[:, out_reg]         # (B, K, P)
    pk = alg.pack(sysm.split(sysm.onehot([t.numerator for t in targets], dev)))
    qk = alg.pack(sysm.split(sysm.onehot([t.denominator for t in targets], dev)))
    resid = alg.compose_packed(alg.compose_packed(num, qk, "*"),
                               alg.compose_packed(den, pk, "*"), "-")
    zero = torch.zeros(len(targets), dtype=torch.long, device=keep.device)
    per_mod = [torch.nn.functional.nll_loss(
                   torch.log(blk.clamp_min(1e-9)), zero, reduction="none")
               for blk in alg.unpack(resid)]
    row = torch.stack(per_mod).mean(0)                            # (B,)
    ans = (row * keep).sum() / keep.sum().clamp_min(1.0)

    # P(the denominator is zero in *every* modulus) -- the only way D == 0.
    p_zero = torch.ones(len(targets), device=keep.device)
    for blk in alg.unpack(den):
        p_zero = p_zero * blk[:, 0]
    guard = -torch.log((1.0 - p_zero).clamp_min(1e-9))
    guard = (guard * keep).sum() / keep.sum().clamp_min(1.0)
    return ans + den_zero_coef * guard, {"den_zero": float(guard.detach())}
