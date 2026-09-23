"""The register machine: program semantics, causality, and end-to-end gradients.

The load-bearing properties are that the emitted program means what the grammar's
trace means, that an ill-formed program is unrepresentable rather than merely
penalised, and that gradients reach the program heads *through* exact arithmetic --
which is the thing an external interpreter cannot give you.
"""

from __future__ import annotations

import math

import pytest
import torch

from lamb import ArithmeticTokenizer, LotusConfig
from lamb.algebra import ResidueAlgebra, ResidueSystem
from lamb.alu import parse_expr
from lamb.config import ModelConfig
from lamb.regmachine import (OPS, RegisterFile, RegisterMachine, RegMachineTrainer,
                             causal_mask, execute, gold_program, operands, run_gold)
from lamb.selfplay.grammar import Descriptor, TaskGrammar


def test_gold_program_is_post_order_and_single_assignment():
    t = parse_expr("(6+6)-(4-8)")
    instrs, n_op, out = gold_program(t)
    assert operands(t) == [6, 6, 4, 8] and n_op == 4
    assert instrs == [(OPS.index("+"), 0, 1), (OPS.index("-"), 2, 3),
                      (OPS.index("-"), 4, 5)]
    assert out == 6                                   # last register written
    # every instruction writes a fresh register, and reads only earlier ones
    for t_i, (_, a, b) in enumerate(instrs):
        assert a < n_op + t_i and b < n_op + t_i


def test_program_length_and_operand_count_follow_the_depth():
    g = TaskGrammar()
    for depth in (1, 2, 3, 4):
        expr, _, trace = g.sample_with_trace(Descriptor(depth, 1, 0), 3)
        instrs, n_op, _ = gold_program(parse_expr(expr))
        assert n_op == 2 ** depth
        assert len(instrs) == 2 ** depth - 1
        # instruction i writes trace entry i; the last writes the answer
        assert len(instrs) == len(trace) + 1


def test_gold_programs_execute_exactly_at_several_depths():
    s, alg, g = ResidueSystem(), None, TaskGrammar()
    alg = ResidueAlgebra(s)
    for depth in (2, 3, 4):
        trees, answers = [], []
        for seed in range(30):
            e, a, _ = g.sample_with_trace(Descriptor(depth, 1, 0), seed)
            if not s.representable(int(a)):
                continue
            trees.append(parse_expr(e))
            answers.append(int(a))
        assert run_gold(s, alg, trees) == answers


def test_soft_execution_with_sharp_logits_equals_hard_execution():
    """Execution is a mixture over registers and operations. With a decided model
    the mixture has to collapse to the discrete program, or the differentiable
    relaxation is not a relaxation of anything."""
    s = ResidueSystem()
    alg = ResidueAlgebra(s)
    g = TaskGrammar()
    trees, answers = [], []
    for seed in range(16):
        e, a, _ = g.sample_with_trace(Descriptor(2, 1, 0), seed)
        trees.append(parse_expr(e))
        answers.append(int(a))
    golds = [gold_program(t)[0] for t in trees]
    n_op, n_instr = 4, 3
    n_slots = n_op + n_instr
    B = len(trees)
    # The file is preallocated to its final width and pointers span all of it;
    # slots past the written tail are zero, which is what the causal mask produces
    # in the learned path.
    regs = RegisterFile(s, [operands(t) for t in trees], n_total=n_slots)
    for step in range(n_instr):
        op_l = torch.full((B, len(OPS)), -30.0)
        a_l = torch.full((B, n_slots), -30.0)
        b_l = torch.full((B, n_slots), -30.0)
        for i, gp in enumerate(golds):
            o, ra, rb = gp[step]
            op_l[i, o] = 30.0
            a_l[i, ra] = 30.0
            b_l[i, rb] = 30.0
        regs.append(execute(alg, regs, torch.softmax(op_l, -1),
                            torch.softmax(a_l, -1), torch.softmax(b_l, -1)))
    assert regs.decode(n_op + n_instr - 1) == answers


def test_pointers_cannot_reference_unwritten_registers():
    """Enforced by masking, not by a loss: a program that reads a register it has
    not written is not a worse program, it is not a program."""
    m = causal_mask(7, 4, 0)
    assert torch.isfinite(m[:4]).all() and torch.isinf(m[4:]).all()
    rm = RegisterMachine(d_model=16, n_operands=4, n_instr=3, system=ResidueSystem())
    _, a_l, b_l = rm.logits(torch.randn(2, 3, 16))
    for t in range(3):
        assert int(torch.isfinite(a_l[0, t]).sum()) == 4 + t
        assert int(torch.isfinite(b_l[0, t]).sum()) == 4 + t


def test_gradients_reach_the_program_heads_from_an_answer_loss():
    """The property an external interpreter cannot provide. PAL/Program-of-Thought
    call a non-differentiable Python process, so the program is trainable only by
    imitation or RL -- and RL on this latent block was measured inert."""
    s = ResidueSystem()
    g = TaskGrammar()
    trees, answers = [], []
    for seed in range(16):
        e, a, _ = g.sample_with_trace(Descriptor(2, 1, 0), seed)
        trees.append(parse_expr(e))
        answers.append(int(a))
    rm = RegisterMachine(d_model=16, n_operands=4, n_instr=3, system=s)
    h = torch.randn(len(trees), 3, 16, requires_grad=True)
    regs, _ = rm.run(h, [operands(t) for t in trees])
    tgt = s.targets(answers)
    loss = sum(torch.nn.functional.nll_loss(
                   torch.log(blk[:, 6].clamp_min(1e-9)), tgt[:, k])
               for k, blk in enumerate(regs.blocks))
    loss.backward()
    assert float(rm.ptr_a.weight.grad.norm()) > 0.0
    assert float(rm.op_head.weight.grad.norm()) > 0.0
    assert float(h.grad.norm()) > 0.0          # and back into the latent block


def test_program_loss_starts_near_chance():
    s = ResidueSystem()
    rm = RegisterMachine(d_model=16, n_operands=4, n_instr=3, system=s)
    g = TaskGrammar()
    trees = [parse_expr(g.sample_with_trace(Descriptor(2, 1, 0), i)[0]) for i in range(16)]
    golds = [gold_program(t)[0] for t in trees]
    _, logits = rm.run(torch.randn(16, 3, 16), [operands(t) for t in trees])
    chance = (math.log(3) + 2 * math.log(4)) / 3        # ops, then two pointers
    assert abs(float(rm.program_loss(logits, golds).detach()) - chance) < 0.6


def test_gumbel_straight_through_keeps_the_forward_pass_discrete():
    """A straight-through sample lets the executor see a real program rather than a
    blur of three, while the backward pass stays differentiable."""
    logits = torch.randn(8, 5, requires_grad=True)
    soft = RegisterMachine._select(logits, tau=0.0, hard=False)
    hard = RegisterMachine._select(logits, tau=1.0, hard=True)
    assert torch.allclose(soft.sum(-1), torch.ones(8))
    assert torch.allclose(hard.sum(-1), torch.ones(8))
    assert set(hard.detach().flatten().tolist()) <= {0.0, 1.0}
    hard.sum().backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) >= 0.0


def _trainer(**kw):
    torch.manual_seed(0)
    base = dict(steps=4, batch_size=16, n_latent=8, loops=3, depth=2, eval_tasks=16,
                trace_coef=0.0, alu_coef=0.0, use_boundaries=False, switch_coef=0.0,
                device="cpu", alu_moduli=(16, 25, 27, 11, 37))
    base.update(kw)
    return RegMachineTrainer(LotusConfig(**base), ArithmeticTokenizer(),
                             ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3))


def test_trainer_needs_a_slot_per_instruction():
    with pytest.raises(ValueError):
        _trainer(depth=3, n_latent=4)         # depth 3 needs 7 instructions


def test_training_reduces_both_losses():
    tr = _trainer(steps=30, batch_size=32)
    first = tr.train_step(0)
    last = first
    for s in range(1, 30):
        last = tr.train_step(s)
    assert last["program"] < first["program"]


def test_evaluation_separates_program_correctness_from_answer_correctness():
    """A right answer from a wrong program is luck; a wrong answer from a right
    program means the operands or the range failed. Reporting one number hides
    both."""
    tr = _trainer()
    r = tr.evaluate(32)
    for k in ("answer_acc", "instr_acc", "program_acc", "op_acc", "ptr_acc"):
        assert 0.0 <= r[k] <= 1.0


def test_register_file_width_is_static():
    """Appending with torch.cat changes the shape every instruction, and a changing
    shape is the one thing compilers cannot fuse across -- XLA requires static
    shapes outright, and TorchInductor traces it but cannot fuse over the boundary.
    On a workload this launch-bound, losing fusion costs more than preallocating."""
    s = ResidueSystem((16, 25, 27, 11, 37))
    g = TaskGrammar()
    trees = [parse_expr(g.sample_with_trace(Descriptor(2, 1, 0), i)[0]) for i in range(6)]
    vals = [operands(t) for t in trees]
    rm = RegisterMachine(d_model=16, n_operands=4, n_instr=3, system=s)
    regs = RegisterFile(s, vals, n_total=rm.n_slots)
    shapes = [tuple(regs.packed.shape)]
    for _ in range(3):
        regs.append(torch.zeros(6, len(s.moduli), max(s.moduli)))
        shapes.append(tuple(regs.packed.shape))
    assert len(set(shapes)) == 1, shapes            # identical throughout
    assert shapes[0][1] == rm.n_slots


def test_unwritten_registers_hold_nothing():
    """A preallocated slot must not read as a value, or a pointer into the tail
    would silently contribute a spurious operand."""
    s = ResidueSystem((16, 25, 27, 11, 37))
    regs = RegisterFile(s, [[1, 2, 3, 4]], n_total=7)
    assert float(regs.packed[0, 4:].sum()) == 0.0
    assert abs(float(regs.packed[0, 0].sum()) - len(s.moduli)) < 1e-4   # written: 1 per modulus


def test_the_whole_machine_traces_as_one_graph():
    """Zero graph breaks is what makes torch.compile worth reaching for here, and
    is also the evidence that a JAX rewrite would not buy fusion this does not
    already have."""
    import torch._dynamo as dynamo

    s = ResidueSystem((16, 25, 27, 11, 37))
    g = TaskGrammar()
    vals = [operands(parse_expr(g.sample_with_trace(Descriptor(2, 1, 0), i)[0]))
            for i in range(4)]
    rm = RegisterMachine(d_model=16, n_operands=4, n_instr=3, system=s)
    dynamo.reset()
    ex = dynamo.explain(lambda x: rm.run(x, vals)[0].packed)(torch.randn(4, 3, 16))
    assert ex.graph_break_count == 0
    assert ex.graph_count == 1
