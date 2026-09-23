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

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .algebra import ResidueAlgebra, ResidueSystem
from .alu import Tree, is_leaf

OPS: Tuple[str, ...] = ("+", "-", "*")
Instr = Tuple[int, int, int]          # (op index, register a, register b)


def operands(t: Tree) -> List[int]:
    """The tree's literal operands, left to right -- the initial register file."""
    if is_leaf(t):
        _, a, b = t
        return [a, b]
    _, l, r = t
    return operands(l) + operands(r)


def gold_program(t: Tree) -> Tuple[List[Instr], int, int]:
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
        instrs.append((OPS.index(op), ra, rb))
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


def execute(alg: ResidueAlgebra, regs: RegisterFile, op_w: torch.Tensor,
            ptr_a: torch.Tensor, ptr_b: torch.Tensor) -> torch.Tensor:
    """One instruction. ``op_w`` ``(B, n_ops)``; pointers ``(B, R)``, all normalised.

    Returns the packed ``(B, K, P)`` result. The operation is a mixture over the
    three composed results rather than a choice between them, so an undecided model
    still produces a well-formed value and the gradient can tell it which way to
    move. All three compositions run batched over moduli, so an instruction costs
    three kernels' worth of algebra rather than three times seven.
    """
    a, b = regs.read(ptr_a), regs.read(ptr_b)
    out: Optional[torch.Tensor] = None
    for oi, op in enumerate(OPS):
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


def run_gold(sysm: ResidueSystem, alg: ResidueAlgebra, trees: Sequence[Tree],
             device: str = "cpu") -> List[int]:
    """Execute each tree's own gold program. The oracle the neural version is
    measured against, and the check that the semantics are right at all."""
    progs = [gold_program(t) for t in trees]
    n_op = progs[0][1]
    n_instr = len(progs[0][0])
    n_slots = n_op + n_instr                      # the file's final, fixed width
    regs = RegisterFile(sysm, [operands(t) for t in trees], device, n_total=n_slots)
    b = len(trees)
    for step in range(n_instr):
        op_w = torch.zeros(b, len(OPS), device=device)
        # Pointers span the whole file; slots past the written tail are simply zero,
        # which is what the causal mask produces in the learned path too.
        pa = torch.zeros(b, n_slots, device=device)
        pb = torch.zeros(b, n_slots, device=device)
        for i, (instrs, _, _) in enumerate(progs):
            oi, ra, rb = instrs[step]
            op_w[i, oi] = 1.0
            pa[i, ra] = 1.0
            pb[i, rb] = 1.0
        regs.append(execute(alg, regs, op_w, pa, pb))
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
                 system: Optional[ResidueSystem] = None, device: str = "cpu"):
        super().__init__()
        self.sys = system or ResidueSystem()
        self.alg = ResidueAlgebra(self.sys, device=device)
        self.n_operands = n_operands
        self.n_instr = n_instr
        self.n_slots = n_operands + n_instr
        self.op_head = nn.Linear(d_model, len(OPS))
        self.ptr_a = nn.Linear(d_model, self.n_slots)
        self.ptr_b = nn.Linear(d_model, self.n_slots)

    def logits(self, latent_h: torch.Tensor):
        """``(B, n_instr, d)`` -> op, ptr-a and ptr-b logits, pointers masked."""
        h = latent_h[:, : self.n_instr]
        op = self.op_head(h)
        a, b = self.ptr_a(h), self.ptr_b(h)
        dev = latent_h.device
        mask = torch.stack([causal_mask(self.n_slots, self.n_operands, t, dev)
                            for t in range(self.n_instr)])          # (n_instr, slots)
        return op, a + mask, b + mask

    @staticmethod
    def _select(logits: torch.Tensor, tau: float, hard: bool) -> torch.Tensor:
        if tau <= 0:
            return torch.softmax(logits, dim=-1)
        return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

    def run(self, latent_h: torch.Tensor, values: Sequence[Sequence[int]],
            tau: float = 0.0, hard: bool = False):
        """Execute the emitted program. Returns ``(register file, logits)``.

        The operands are loaded exactly -- they are digits in the prompt and the
        digit->residue map is a known fixed function (ROADMAP 3a-vi: asking the
        network to induce it leaves it at chance for 1200 steps). So nothing here
        is spent on arithmetic the model should not be learning, and the only thing
        being tested is whether it can emit the right *program*.
        """
        op_l, a_l, b_l = self.logits(latent_h)
        regs = RegisterFile(self.sys, values, str(latent_h.device),
                            n_total=self.n_slots)
        for t in range(self.n_instr):
            # Softmax over the *full* register file rather than a slice. The causal
            # mask is already -inf past the written tail, so those entries come out
            # exactly zero -- same result, static shape.
            ow = self._select(op_l[:, t], tau, hard)
            pa = self._select(a_l[:, t], tau, hard)
            pb = self._select(b_l[:, t], tau, hard)
            regs.append(execute(self.alg, regs, ow, pa, pb))
        return regs, (op_l, a_l, b_l)

    def program_loss(self, logits, gold: Sequence[Sequence[Instr]]) -> torch.Tensor:
        """Cross-entropy on the emitted program against the generator's own.

        The gold program is free: the grammar knows the expression it built, so
        every problem carries its exact program as well as its exact answer. That is
        what makes supervising a program affordable here and not elsewhere, and it
        is the curriculum that avoids the instability neural program induction is
        known for -- supervise first, relax after.
        """
        op_l, a_l, b_l = logits
        dev = op_l.device
        o = torch.tensor([[i[0] for i in g] for g in gold], device=dev)
        a = torch.tensor([[i[1] for i in g] for g in gold], device=dev)
        b = torch.tensor([[i[2] for i in g] for g in gold], device=dev)
        ce = torch.nn.functional.cross_entropy
        return (ce(op_l.reshape(-1, op_l.size(-1)), o.reshape(-1))
                + ce(a_l.reshape(-1, a_l.size(-1)), a.reshape(-1))
                + ce(b_l.reshape(-1, b_l.size(-1)), b.reshape(-1))) / 3.0


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
    """

    def __init__(self, cfg, tokenizer, model_cfg=None, program_coef: float = 1.0,
                 answer_coef: float = 1.0, tau: float = 0.0, hard: bool = False):
        from .lotus import LotusTrainer

        self.inner = LotusTrainer(cfg, tokenizer, model_cfg)
        self.cfg = cfg
        self.device = self.inner.device
        self.n_operands = 2 ** cfg.depth
        self.n_instr = 2 ** cfg.depth - 1
        if cfg.n_latent < self.n_instr:
            raise ValueError(f"n_latent={cfg.n_latent} < {self.n_instr} instructions "
                             f"needed at depth {cfg.depth}")
        d = self.inner.reasoner.model.cfg.d_model
        self.machine = RegisterMachine(d, self.n_operands, self.n_instr,
                                       ResidueSystem(tuple(cfg.alu_moduli)),
                                       device=str(self.device)).to(self.device)
        self.program_coef, self.answer_coef = program_coef, answer_coef
        self.tau, self.hard = tau, hard
        self.opt = torch.optim.AdamW(
            list(self.inner.reasoner.parameters()) + list(self.machine.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay)

    # -- one batch ---------------------------------------------------------
    def _prepare(self, tasks):
        from .alu import parse_expr

        trees = [parse_expr(e) for e, _, _ in tasks]
        golds = [gold_program(t)[0] for t in trees]
        vals = [operands(t) for t in trees]
        answers = [int(a) for _, a, _ in tasks]
        return trees, golds, vals, answers

    def _latents(self, tasks):
        prompt, *_ = self.inner._collate(tasks)
        m = self.inner.reasoner.model
        x = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                    prompt["value"], prompt["value_mask"])
        _, _, latent_h, _ = self.inner.reasoner.latent_block(x, prompt["pad_mask"])
        return latent_h

    def _losses(self, tasks):
        trees, golds, vals, answers = self._prepare(tasks)
        latent_h = self._latents(tasks)
        regs, logits = self.machine.run(latent_h, vals, self.tau, self.hard)
        prog = self.machine.program_loss(logits, golds)
        out_reg = self.n_operands + self.n_instr - 1
        tgt = self.machine.sys.targets(answers, device=str(self.device))
        ans = sum(torch.nn.functional.nll_loss(
                      torch.log(blk[:, out_reg].clamp_min(1e-9)), tgt[:, k])
                  for k, blk in enumerate(regs.blocks)) / len(regs.blocks)
        return prog, ans, regs, logits, golds, answers, out_reg

    def train_step(self, step: int):
        self.inner.reasoner.train()
        self.machine.train()
        tasks = self.inner._sample_batch(self.cfg.batch_size)
        for g in self.opt.param_groups:
            g["lr"] = self.inner._lr_at(step)
        prog, ans, *_ = self._losses(tasks)
        loss = self.program_coef * prog + self.answer_coef * ans
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.inner.reasoner.parameters()) + list(self.machine.parameters()),
            self.cfg.grad_clip)
        self.opt.step()
        return {"loss": float(loss.detach()), "program": float(prog.detach()),
                "answer": float(ans.detach())}

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
        """
        self.inner.reasoner.eval()
        self.machine.eval()
        tasks = self.inner._eval_set(n_tasks)
        prog, ans, regs, logits, golds, answers, out_reg = self._losses(tasks)
        got = regs.decode(out_reg)
        acc = sum(g == a for g, a in zip(got, answers)) / len(answers)
        op_l, a_l, b_l = logits
        dev = op_l.device
        o = torch.tensor([[i[0] for i in g] for g in golds], device=dev)
        ra = torch.tensor([[i[1] for i in g] for g in golds], device=dev)
        rb = torch.tensor([[i[2] for i in g] for g in golds], device=dev)
        ok_op = (op_l.argmax(-1) == o).float()
        ok_a = (a_l.argmax(-1) == ra).float()
        ok_b = (b_l.argmax(-1) == rb).float()
        whole = (ok_op * ok_a * ok_b)
        return {"answer_acc": acc,                       # the correctness measure
                "instr_acc": float(whole.mean()),         # per-instruction conformity
                "canonical_acc": float(whole.min(dim=1).values.mean()),
                "program_acc": float(whole.min(dim=1).values.mean()),  # kept: old name
                "op_acc": float(ok_op.mean()), "ptr_acc": float((ok_a * ok_b).mean()),
                "program_loss": float(prog), "answer_loss": float(ans)}
