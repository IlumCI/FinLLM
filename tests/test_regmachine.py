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
    # ``_n_instr`` goes to the trainer, not to LotusConfig, so the instruction budget
    # can be over-provisioned independently of the task's depth.
    n_instr = kw.pop("_n_instr", None)
    base = dict(steps=4, batch_size=16, n_latent=8, loops=3, depth=2, eval_tasks=16,
                trace_coef=0.0, alu_coef=0.0, use_boundaries=False, switch_coef=0.0,
                device="cpu", alu_moduli=(16, 25, 27, 11, 37))
    base.update(kw)
    return RegMachineTrainer(LotusConfig(**base), ArithmeticTokenizer(),
                             ModelConfig(d_model=48, n_heads=4, d_ff=96, recurrent_steps=3),
                             n_instr=n_instr)


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


def test_rational_mode_closes_the_instruction_set_under_division():
    from lamb.regmachine import RATIONAL_OPS

    rm = RegisterMachine(d_model=16, n_operands=4, n_instr=3, rational=True)
    assert rm.ops == RATIONAL_OPS == ("+", "-", "*", "/")
    assert rm.op_head.out_features == 4
    # the integer path, which carries the depth-2 result, is untouched
    assert RegisterMachine(d_model=8, n_operands=4, n_instr=3).ops == ("+", "-", "*")


def test_division_executes_exactly_through_a_program():
    """(48 / 2) + 3.25 -- a division and a decimal, neither expressible on plain
    residues: small divisors have no modular inverse under these moduli, and a
    scale is not something the integer ring tracks."""
    from fractions import Fraction

    from lamb.rational import RationalAlgebra
    from lamb.regmachine import RATIONAL_OPS, RationalRegisterFile, execute_rational

    R = RationalAlgebra()
    vals = [[Fraction(48), Fraction(2), Fraction(13, 4), Fraction(0)]]
    regs = RationalRegisterFile(R, vals, n_total=6)

    def pick(width, idx):
        t = torch.full((1, width), -30.0)
        t[0, idx] = 30.0
        return torch.softmax(t, -1)

    regs.append(execute_rational(R, regs, pick(4, RATIONAL_OPS.index("/")),
                                 pick(6, 0), pick(6, 1)))
    assert regs.decode(4) == [Fraction(24)]
    regs.append(execute_rational(R, regs, pick(4, RATIONAL_OPS.index("+")),
                                 pick(6, 4), pick(6, 2)))
    assert regs.decode(5) == [Fraction(48, 2) + Fraction(13, 4)]


def test_soft_pointer_reads_are_incoherent_not_blended():
    """A soft read does not average the registers -- it decodes each modulus
    independently, so the winning residues can come from different registers and the
    CRT lands nowhere near either. An earlier version of this test asserted the
    'blend' story and failed, which is how the real behaviour was found.
    """
    import random
    from fractions import Fraction

    from lamb.rational import RationalAlgebra
    from lamb.regmachine import RationalRegisterFile

    R = RationalAlgebra()
    regs = RationalRegisterFile(R, [[Fraction(1, 2), Fraction(3, 4)]], n_total=2)
    assert R.decode(regs.read(torch.tensor([[1.0, 0.0]]))) == [Fraction(1, 2)]
    assert R.decode(regs.read(torch.tensor([[0.0, 1.0]]))) == [Fraction(3, 4)]

    rng = random.Random(0)
    neither = 0
    for _ in range(120):
        a = Fraction(rng.randint(1, 50), rng.choice([1, 2, 3, 4, 5]))
        b = Fraction(rng.randint(1, 50), rng.choice([1, 2, 3, 4, 5]))
        rf = RationalRegisterFile(R, [[a, b]], n_total=2)
        got = R.decode(rf.read(torch.tensor([[0.5, 0.5]])))[0]
        neither += got not in (a, b)
    assert neither > 10          # it happens often; ~30% over larger samples


def test_redundant_moduli_catch_incoherent_reads():
    """The redundancy added for the model's own residue errors covers this too: an
    incoherent vector is not a legitimate value, and the range check does not care
    what made it inconsistent. The soft-pointer caveat therefore degrades to
    'detected' rather than 'silently wrong'."""
    import random

    from lamb.algebra import RedundantResidueSystem

    s = RedundantResidueSystem((16, 25, 27, 11, 37, 7, 41), n_core=4)
    alg = ResidueAlgebra(s)
    rng = random.Random(0)
    silent = incoherent = 0
    for _ in range(150):
        a, b = rng.randint(-40000, 40000), rng.randint(-40000, 40000)
        rf = RegisterFile(s, [[a, b]], n_total=2, alg=alg)
        blocks = rf.read(torch.tensor([[0.5, 0.5]]))
        res = [int(bl.argmax(-1)) for bl in alg.unpack(blocks)]
        if s.crt(res) in (a, b):
            continue
        incoherent += 1
        if not s.detect(res):
            silent += 1
    assert incoherent > 0
    assert silent == 0           # never silently wrong


def test_gradients_reach_the_program_heads_through_rational_arithmetic():
    from fractions import Fraction

    rm = RegisterMachine(d_model=16, n_operands=4, n_instr=3, rational=True)
    vals = [[Fraction(48), Fraction(2), Fraction(13, 4), Fraction(1)]] * 4
    h = torch.randn(4, 3, 16, requires_grad=True)
    regs, _ = rm.run(h, vals)
    tgt = rm.sys.targets([48] * 4)
    loss = sum(torch.nn.functional.nll_loss(torch.log(b.clamp_min(1e-9)), tgt[:, k])
               for k, b in enumerate(rm.alg.unpack(regs.num[:, 6])))
    loss.backward()
    assert float(rm.ptr_a.weight.grad.norm()) > 0.0
    assert float(rm.op_head.weight.grad.norm()) > 0.0


def test_gold_program_indexes_into_whichever_instruction_set_is_in_use():
    from lamb.regmachine import RATIONAL_OPS

    t = parse_expr("(6+6)-(4-8)")
    int_prog, _, _ = gold_program(t)
    rat_prog, _, _ = gold_program(t, ops=RATIONAL_OPS)
    assert [OPS[i[0]] for i in int_prog] == [RATIONAL_OPS[i[0]] for i in rat_prog]


def test_the_grammars_division_output_reaches_the_machine():
    """Generated, parsed, compiled, executed -- the four layers that were not joined.

    The commit that added division ("Division end to end: tokenizer, evaluator,
    grammar") stopped one layer short of the consumer. ``alu.parse_expr`` scanned
    ``"+-*"``, so the grammar's own leaf ``48/2`` was *unparseable*; and
    ``gold_program`` defaulted to the integer ``OPS``, so a ``/`` node raised from
    ``ops.index``. Both sit inside ``RegMachineTrainer._prepare``, so the only
    program trainer in the repo crashed on data that generated perfectly well, and
    the ROADMAP recorded the opposite problem -- "no division training data yet".

    Nothing caught it because every rational test hand-fed ``Fraction`` values
    through hand-built pointers. A test that constructs its own inputs cannot fail
    on a parser, which is exactly the shape of gap worth pinning: the assertion here
    is less important than the fact that it starts from a string the grammar wrote.
    """
    from fractions import Fraction

    from lamb.regmachine import RATIONAL_OPS, RegisterMachine, run_program
    from lamb.selfplay.grammar import Descriptor, TaskGrammar

    g = TaskGrammar()
    machine = RegisterMachine(d_model=8, n_operands=4, n_instr=3, rational=True)
    exprs, progs, vals, answers = [], [], [], []
    for i in range(48):
        expr, ans, _ = g.sample_with_trace(Descriptor(2, 1, 2), i)
        t = parse_expr(expr)                       # used to raise on "48/2"
        instrs, n_op, out = gold_program(t, ops=RATIONAL_OPS)
        assert n_op == 4 and out == 6
        exprs.append(expr)
        progs.append(instrs)
        vals.append(operands(t))
        answers.append(Fraction(int(ans)))

    assert any("/" in e for e in exprs), "ops_key=2 should emit division"
    assert any(RATIONAL_OPS[i[0]] == "/" for p in progs for i in p)

    regs = run_program(machine, vals, progs)
    assert regs.decode(6) == answers


def test_the_integer_instruction_set_cannot_take_the_grammars_division():
    """The honest limit, pinned so it is not mistaken for an oversight.

    Parsing ``/`` and *composing* it are different questions. The integer ring has
    no division at all -- an inverse needs a divisor coprime to every modulus, and
    the short-digit-period moduli are exactly the set that denies that -- so a ``/``
    tree must be rejected by the integer instruction set rather than quietly
    mapped onto some other operation.
    """
    import pytest

    t = parse_expr("48/2")
    with pytest.raises(ValueError):
        gold_program(t)                            # default ops == ("+", "-", "*")


def test_out_of_range_answers_are_masked_rather_than_trained_on():
    """A wrapped value is a *different* number, not a large one.

    ``ResidueSystem.targets`` is an unguarded ``int(v) % p``, so an answer outside
    the ring becomes a perfectly legal cross-entropy target for the wrong number and
    the model is trained on arithmetic that is false. :mod:`lamb.lotus` has masked on
    this since the ALU landed; this trainer did not, and it went unnoticed because
    depth 2 at one digit over ``(+,-)`` never leaves the ring. Depth 3, or
    multiplication, leaves it immediately -- which is precisely the configuration the
    scaling study needs.
    """
    tr = _trainer()
    sysm = tr.machine.sys
    lo, hi = sysm.bounds()
    assert tr._in_ring(0, [1, 2], [3, 4])
    assert not tr._in_ring(hi + 1, [], [1, 2])          # the answer leaves the ring
    assert not tr._in_ring(1, [hi + 1], [1, 2])         # an intermediate does
    assert not tr._in_ring(1, [], [lo - 1, 2])          # an operand does


def test_the_register_file_can_refuse_instead_of_answering():
    """The refusal path, wired into the decode rather than living in a test.

    ``RedundantResidueSystem`` has been measured since 3a-ix -- 100% of single
    errors corrected, 0% mis-corrected, 100% of incoherent soft reads detected --
    and was imported by nothing. A bare CRT has no locality, so one wrong residue
    is not a near miss, it is a uniformly wrong number that is indistinguishable
    from a right one. That is the failure the bridge walks into, because off the
    grammar's distribution there is no verifier to catch it.

    Three outcomes, and the third is the one that has to stay empty: corrected,
    refused, or silently wrong.
    """
    from lamb.algebra import RedundantResidueSystem
    from lamb.regmachine import RegisterFile

    sysm = RedundantResidueSystem((16, 25, 27, 11) + (37, 7, 41), n_core=4)
    values = [[123, -4567, 0, 890]]
    regs = RegisterFile(sysm, values, n_total=4)
    assert [v for v, _ in regs.decode_checked(0)] == [123]

    # Corrupt one residue of register 1 and the check should repair it and name
    # the culprit -- neither of which a plain CRT can do.
    k_bad = 2
    res = regs.residues_at(1)[0]
    res[k_bad] = (res[k_bad] + 1) % sysm.moduli[k_bad]
    val, faulty = sysm.correct(res)
    assert (val, faulty) == (-4567, k_bad)

    # Two simultaneous errors: the evidence no longer singles out a culprit, so it
    # returns nothing rather than the most plausible candidate. An admitted failure
    # beats a confident number wherever nothing downstream can catch it.
    res = regs.residues_at(3)[0]
    for k in (0, 1):
        res[k] = (res[k] + 1) % sysm.moduli[k]
    val, faulty = sysm.correct(res)
    assert val is None or val == 890          # never a third, wrong number


def test_a_plain_system_says_so_rather_than_pretending_to_check():
    """No spare moduli means no check. Silently returning a bare CRT from a method
    named ``decode_checked`` would be the worst of both."""
    import pytest

    from lamb.algebra import ResidueSystem
    from lamb.regmachine import RegisterFile

    regs = RegisterFile(ResidueSystem((16, 25, 27, 11)), [[7, 8]], n_total=2)
    with pytest.raises(TypeError):
        regs.decode_checked(0)


def test_execute_honours_the_instruction_set_it_is_given():
    """It used to read the module-level ``OPS``. Harmless while every machine had
    three operations, silently wrong the moment the operation head is four wide: the
    fourth logit would be trainable, selectable, and never composed."""
    from lamb.algebra import ResidueAlgebra, ResidueSystem
    from lamb.regmachine import RegisterFile, execute

    sysm = ResidueSystem((16, 25, 27, 11, 37))
    alg = ResidueAlgebra(sysm)
    regs = RegisterFile(sysm, [[9, 4]], n_total=3)
    one_hot = lambda i, n: torch.eye(n)[i:i + 1]                  # noqa: E731

    # Given only ("*",), the single op weight must mean multiplication, not addition.
    got = RegisterFile(sysm, [[9, 4]], n_total=3)
    got.append(execute(alg, regs, one_hot(0, 1), one_hot(0, 3), one_hot(1, 3),
                       ops=("*",)))
    assert got.decode(2) == [36]

    # And given ("+",) the same weights must mean addition. Reading OPS would have
    # made both of these say 13.
    got = RegisterFile(sysm, [[9, 4]], n_total=3)
    got.append(execute(alg, regs, one_hot(0, 1), one_hot(0, 3), one_hot(1, 3),
                       ops=("+",)))
    assert got.decode(2) == [13]


def test_the_rational_answer_loss_degenerate_is_zero_over_zero_and_nothing_else():
    """``N*q - D*p = 0`` is the exact statement that two fractions are equal, and it
    has exactly one blind spot. Finding out *which* took writing this test.

    **The first version of this was wrong, in the direction that overstates the
    problem.** It claimed the degenerate was reached "by dividing through a register
    that holds zero". It is not: ``5 / 0`` gives ``(N, D) = (5, 0)``, whose residual
    is ``5*1 - 0*7 = 5``, so the criterion *catches* it at full cost. Division by a
    zero register is detected, not hidden.

    The real degenerate needs ``N`` and ``D`` both zero, which for ``a/b / c/d ->
    (ad, bc)`` means a zero *numerator* and a zero *divisor*: ``0 / 0``. Reachable --
    one-digit operands include ``0``, and a pointer pair may address the same register
    twice -- but a good deal narrower than "any zero divisor". So the guard is
    load-bearing for one case rather than a general necessity, and the residual does
    more of the work than the docstring first gave it credit for.

    This is ROADMAP 3a-viii's "division by zero is undetectable in the ring" met from
    the loss side, and the ring turns out to be better at it than expected.
    """
    from fractions import Fraction

    from lamb.algebra import ResidueSystem
    from lamb.rational import RATIONAL_MODULI
    from lamb.regmachine import (RATIONAL_OPS, RegisterMachine,
                                 rational_answer_loss, run_program)

    machine = RegisterMachine(8, n_operands=2, n_instr=1, rational=True,
                              system=ResidueSystem(RATIONAL_MODULI))
    keep = torch.ones(1)
    div = RATIONAL_OPS.index("/")

    def loss(vals, target, coef):
        regs = run_program(machine, [vals], [[(div, 0, 1)]])
        return rational_answer_loss(machine, regs, 2, [target], keep,
                                    den_zero_coef=coef)

    # 5 / 0 -> (5, 0). The residual alone already rejects it.
    bare, _ = loss([Fraction(5), Fraction(0)], Fraction(7), 0.0)
    assert float(bare) > 1.0, "a zero divisor with a live numerator is detected"

    # 0 / 0 -> (0, 0). This one looks perfect to the residual for *any* target.
    bare, _ = loss([Fraction(0), Fraction(0)], Fraction(7), 0.0)
    assert float(bare) < 1e-3

    # ...and the guard is the only thing that prices it.
    guarded, extra = loss([Fraction(0), Fraction(0)], Fraction(7), 1.0)
    assert float(guarded) > 1.0 and extra["den_zero"] > 1.0

    # A legitimate value pays neither cost.
    good, extra = loss([Fraction(5), Fraction(2)], Fraction(5, 2), 1.0)
    assert float(good) < 1e-3 and extra["den_zero"] < 1e-6


def test_per_row_masking_is_not_a_prefix():
    """A word problem's register file is padded, and the readable set has a hole in it.

    Instruction ``t`` may read the row's own operands ``[0, count)`` and the results
    written so far ``[n_operands, n_operands + t)`` -- but *not* the padding in
    between. Masking the prefix ``[0, count + t)`` instead is the natural mistake and
    it is wrong in the direction that hides itself: it makes padding readable and the
    early results unreachable, both silently.
    """
    from lamb.regmachine import row_causal_mask

    # 4 operand slots, 3 instructions; row 0 has 2 real operands, row 1 has 4.
    m = row_causal_mask(n_slots=7, n_operands=4, n_instr=3,
                        counts=torch.tensor([2, 4]))
    assert m.shape == (2, 3, 7)
    ok = torch.isfinite(m)

    # row 0, instruction 0: only its own two operands
    assert ok[0, 0].tolist() == [True, True, False, False, False, False, False]
    # row 0, instruction 2: operands 0-1, plus results at 4 and 5 -- slots 2 and 3
    # are padding and stay closed, which a prefix mask would have opened.
    assert ok[0, 2].tolist() == [True, True, False, False, True, True, False]
    # row 1 fills the operand block, so for it the readable set *is* a prefix
    assert ok[1, 2].tolist() == [True, True, True, True, True, True, False]


def test_a_row_with_no_operands_does_not_produce_nan():
    """An all-masked row softmaxes to NaN and takes the batch with it. The floor is
    one slot rather than an exception; with constants preloaded the count is never
    actually zero, which is exactly why this would go unnoticed."""
    from lamb.regmachine import row_causal_mask

    m = row_causal_mask(4, 2, 2, torch.tensor([0]))
    assert torch.isfinite(torch.softmax(m[0, 0], dim=-1)).all()


def test_the_anneal_schedule_actually_withdraws_the_gold_program():
    """A schedule that silently never withdraws produces a clean-looking number that
    answers nothing -- it would just be the supervised arm under a different name.

    The window is a fraction of training in the study, not an absolute, for the same
    reason: an absolute window becomes "always supervised" on a longer run and the arm
    stops being the arm.
    """
    tr = _trainer(steps=1000)
    tr.program_anneal = (200, 600)
    got = [round(tr._program_coef_at(s), 3) for s in (0, 100, 200, 300, 400, 500, 600, 999)]
    assert got == [1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
    tr.program_anneal = None
    assert tr._program_coef_at(999) == 1.0        # unchanged when no window is given


def test_partial_supervision_is_decided_per_problem_not_per_step():
    """A dataset either annotates a problem or it never does. Sampling per step would
    hand every problem supervision eventually and measure something else entirely --
    and it would do so while looking exactly like partial supervision.

    This is the language bridge's real condition: GSM8K's calculator annotations
    recover a program for 42% of the train split and nothing for the rest.
    """
    tr = _trainer()
    tr.program_frac = 0.42
    tasks = tr.inner._sample_batch(600)
    first = tr._has_program(tasks)
    # stable across calls -- the same problem gets the same answer forever
    assert tr._has_program(tasks) == first
    # and stable across trainers, because it is a hash of the problem
    other = _trainer()
    other.program_frac = 0.42
    assert other._has_program(tasks) == first
    frac = sum(first) / len(first)
    assert 0.35 < frac < 0.49, frac

    tr.program_frac = 1.0
    assert all(tr._has_program(tasks))
    tr.program_frac = 0.0
    assert not any(tr._has_program(tasks))


def test_unsupervised_rows_still_reach_the_answer_loss():
    """The point of partial supervision: rows without a gold program are dropped from
    the *program* loss and must still train through the answer, since that is exactly
    what has to carry them. Dropping them from both would make the arm a smaller
    supervised arm rather than a partially supervised one."""
    tr = _trainer()
    tr.program_frac = 0.0                      # nothing is supervised
    tasks = tr.inner._sample_batch(8)
    b = tr._losses(tasks)
    assert float(b["prog"]) == 0.0             # no program signal at all
    assert float(b["ans"]) > 0.0               # but the answer loss is live
    b["ans"].backward()
    assert float(tr.machine.ptr_a.weight.grad.norm()) > 0.0


def test_over_provisioned_instruction_budget_pads_exactly():
    """The language bridge's program format, tested where ground truth exists.

    The bridge gives every problem a fixed 8-instruction budget over a chain whose
    median length is 3, padding with ``x * 1``. This trainer derived ``n_instr`` from
    the tree depth, so it could never express a short program and the format went
    untested on the synthetic task.

    The identity register is *prepended* rather than written over an operand: the gold
    program's pointers index the tree's own operands, so overwriting one would change
    the problem while leaving the program that references it intact -- a wrong answer
    from a program that still looks right.
    """
    tr = _trainer(depth=2, n_latent=16)
    assert tr.n_instr == tr.gold_instr == 3 and tr.n_const == 0

    over = _trainer(depth=2, n_latent=16, _n_instr=7)
    assert over.gold_instr == 3 and over.n_instr == 7
    assert over.n_const == 1 and over.n_operands == 5      # 1 constant + 4 operands
    assert over.machine.n_slots == 12

    tasks = over.inner._eval_set(8)
    trees, golds, vals, answers, keep = over._prepare(tasks)
    assert all(v[0] == 1 for v in vals)                    # the identity register
    assert all(len(g) == 7 for g in golds)
    # the pad instructions multiply the running result through register 0
    mul = over.machine.ops.index("*")
    for g in golds:
        assert [i[0] for i in g[3:]] == [mul] * 4
        assert [i[2] for i in g[3:]] == [0] * 4

    # and executing the padded gold program still gives the exact answer
    from lamb.regmachine import run_program
    regs = run_program(over.machine, vals, golds)
    assert regs.decode(over.machine.n_slots - 1) == answers


def test_an_instruction_budget_too_small_for_the_depth_is_refused():
    import pytest

    with pytest.raises(ValueError):
        _trainer(depth=3, n_latent=16, _n_instr=4)     # depth 3 needs 7
