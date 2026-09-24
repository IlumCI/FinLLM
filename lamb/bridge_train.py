"""GSM8K through the language bridge: text in, a program out, the algebra executes it.

:mod:`lamb.bridge` had the pieces -- a quantity parser, a resampler, a register
loader -- and no path between them. Nothing in the repo turned a sentence into a
number. This is that path, and it is deliberately thin, because almost none of it
is learned:

===========================================  ==========
text -> frozen encoder -> cached embeddings  **no**
text -> quantities -> registers              **no** (regex + a fixed rational map)
embeddings -> K latents (resampler)          yes
latents -> program over registers            yes
program -> answer                            **no** (exact rational algebra)
===========================================  ==========

**On the gold program.** The roadmap's premise was that a natural-language problem
does not come with one, which is why the bridge looked like it rested entirely on
the outcome-only induction result of ROADMAP 3a-vii -- measured once, at depth 2,
over three instructions. That premise is true of prose in general and **false of
GSM8K**, whose train solutions carry inline calculator annotations of the form
``<<48/2=24>>``: an operation, its operands and its result, in evaluation order.
That is very nearly the emitted program.

So the bridge is not a leap, it is a measurement with a control. The supervised arm
recovers a program from the dataset's own annotations; the answer-only arm
(``program_coef=0``) removes it entirely and asks the same question 3a-vii asked,
on real text. Neither arm is assumed.

**What is not claimed.** The encoder is pretrained, so it has seen these benchmarks,
and "frozen" means its weights do not move, not that the information is absent. The
claim that this design has a zero GSM8K->GSM1K gap *by construction* does not survive
a pretrained encoder and is not made here. What survives is an experiment, which
``--encoder`` makes a one-flag arm: run the same core on a pretrained encoder and on
one that never saw the benchmark, and the difference prices the encoder's prior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .algebra import ResidueSystem
from .bridge import (DEFAULT_CONSTANTS, DEFAULT_ENCODER, LanguageFront,
                     all_quantities, encode_dataset, load_encoded,
                     registers_from_quantities)
from .config import ModelConfig
from .device import resolve_device
from .model.lamb import build_model
from .rational import GSM8K_MODULI
from .regmachine import (RATIONAL_OPS, Instr, RegisterMachine,
                         rational_answer_loss, run_program)
from .tokenizer import ArithmeticTokenizer

# ---------------------------------------------------------------------------
# 1. The data
# ---------------------------------------------------------------------------

GSM8K_BASE = ("https://raw.githubusercontent.com/openai/grade-school-math/"
              "master/grade_school_math/data")


DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "lamb", "gsm8k")


def fetch_gsm8k(split: str = "train", cache_dir: str = DEFAULT_CACHE
                ) -> List[Dict[str, str]]:
    """GSM8K as two JSONL files, fetched once and cached on disk.

    Deliberately not via ``datasets``. The official split ships as plain JSONL in
    the paper's own repository, so one URL and a cache file replaces a dependency
    tree -- and pins the provenance rather than resolving it. ``uv.lock`` is part of
    the comparison in this project, and the environment behind every arithmetic
    number in it should not move to load a text file.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be train or test, not {split!r}")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{GSM8K_BASE}/{split}.jsonl", path)
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


_FINAL = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_ANNOT = re.compile(r"<<([^<>]*?)=([^<>]*?)>>")
_BINOP = re.compile(r"^\s*(-?[\d,]*\.?\d+)\s*([+\-*/])\s*(-?[\d,]*\.?\d+)\s*$")


def _frac(text: str) -> Optional[Fraction]:
    """A literal as an exact rational. ``3.25`` is ``325/100``, never a float."""
    t = text.strip().replace(",", "").rstrip("%")
    if not t or t in ("-", "."):
        return None
    try:
        if "." in t:
            whole, _, frac = t.partition(".")
            neg = whole.startswith("-")
            digits = (whole.lstrip("-") or "0") + frac
            v = int(digits) * (-1 if neg else 1)
            return Fraction(v, 10 ** len(frac))
        return Fraction(int(t))
    except ValueError:
        return None


def parse_final_answer(solution: str) -> Optional[Fraction]:
    """The number after ``####``, exactly."""
    m = _FINAL.search(solution)
    return _frac(m.group(1)) if m else None


Step = Tuple[Fraction, str, Fraction, Fraction]        # (a, op, b, result)


# No ``-?`` on the number: the sign is lexed as its own token and resolved in
# ``factor`` as a unary minus. Allowing a signed literal here makes the lexer
# greedily swallow a *binary* minus -- ``5-2*2`` tokenised to ``["5", "-2", "*", "2"]``,
# which then fails to parse and rejected a perfectly good annotation. It went unnoticed
# for exactly one test round because the obvious cases (``48*3+5``, ``100*0.2``) have no
# subtraction in them.
_LHS_TOK = re.compile(r"\s*([\d,]*\.?\d+|[-+*/()])")


def _lhs_steps(lhs: str) -> Optional[Tuple[List[Step], Fraction]]:
    """Decompose an annotation's left side into binary steps, in evaluation order.

    ``48*3+5`` becomes ``[(48,*,3,144), (144,+,5,149)]``. Standard precedence,
    left-associative, parentheses honoured -- a recursive-descent parser rather than
    a regex, because the whole point is the *structure*.

    **This is worth a parser, and the measurement says how much.** Compound
    annotations were rejected outright, and rejecting them cascades: the result of a
    rejected step is never written to a register, so every later step that reads it
    fails to align too. Categorising the 39% of chains that failed alignment found
    **52.1% of them traceable to exactly this** -- far and away the largest single
    cause, and larger than every component measured in the ablation above combined.
    """
    toks = []
    i = 0
    while i < len(lhs):
        m = _LHS_TOK.match(lhs, i)
        if not m:
            return None
        toks.append(m.group(1))
        i = m.end()
    if not toks:
        return None

    steps: List[Step] = []
    pos = 0

    def peek() -> Optional[str]:
        return toks[pos] if pos < len(toks) else None

    def expr() -> Optional[Fraction]:
        nonlocal pos
        left = term()
        if left is None:
            return None
        while peek() in ("+", "-"):
            op = toks[pos]
            pos += 1
            right = term()
            if right is None:
                return None
            val = left + right if op == "+" else left - right
            steps.append((left, op, right, val))
            left = val
        return left

    def term() -> Optional[Fraction]:
        nonlocal pos
        left = factor()
        if left is None:
            return None
        while peek() in ("*", "/"):
            op = toks[pos]
            pos += 1
            right = factor()
            if right is None:
                return None
            if op == "/" and right == 0:
                return None            # undetectable in the ring; reject here
            val = left * right if op == "*" else left / right
            steps.append((left, op, right, val))
            left = val
        return left

    def factor() -> Optional[Fraction]:
        nonlocal pos
        t = peek()
        if t is None:
            return None
        if t == "(":
            pos += 1
            v = expr()
            if v is None or peek() != ")":
                return None
            pos += 1
            return v
        # A leading '-' is a sign, not an operator, only at the start of a factor.
        if t == "-":
            pos += 1
            v = factor()
            return None if v is None else -v
        if t in ("+", "*", "/", ")"):
            return None
        pos += 1
        return _frac(t)

    value = expr()
    if value is None or pos != len(toks):
        return None
    return steps, value


def parse_annotation_steps(solution: str) -> Tuple[List[Step], int]:
    """``(steps, n_rejected)`` from GSM8K's ``<<a op b=c>>`` calculator annotations.

    A compound left side is decomposed into its binary steps by :func:`_lhs_steps`,
    so ``<<48*3+5=149>>`` contributes two instructions rather than being thrown away.
    Rejecting them was the single largest cause of alignment failure (52.1% of the
    39%), because the result of a rejected step is never written to a register and
    every later step reading it fails too.

    An annotation that does not compute its own stated result is still dropped --
    GSM8K rounds in places, and supervision toward arithmetic the dataset itself does
    not do would be learned perfectly and be wrong. The rejection count is returned
    rather than logged, so coverage is a number in the output rather than a claim.
    """
    steps: List[Step] = []
    rejected = 0
    for lhs, rhs in _ANNOT.findall(solution):
        res = _frac(rhs)
        parsed = _lhs_steps(lhs) if res is not None else None
        if parsed is None:
            rejected += 1
            continue
        sub, value = parsed
        # The annotation is the dataset's arithmetic, not ours.
        if value != res or not sub:
            rejected += 1
            continue
        steps.extend(sub)
    return steps, rejected


def align_program(reg_values: Sequence[Fraction], steps: Sequence[Step],
                  n_operands: int, n_instr: int, count: Optional[int] = None,
                  ops: Tuple[str, ...] = RATIONAL_OPS,
                  identity_slot: int = 0) -> Optional[List[Instr]]:
    """Turn annotation steps into instructions over a concrete register file.

    An annotation names its operands by *value*, an instruction names them by
    *address*, so alignment is a lookup: every operand must already be in the file,
    either as a quantity the parser found, a preloaded constant, or the result of an
    earlier instruction. ``None`` means it is not -- which is a fact about extraction
    coverage, not about the model, and the reason recall is measured before training.

    **The search is over *readable* slots, not over the list.** The register file is
    padded to a fixed width with zeros, and those slots are masked out of the pointer
    distribution (``row_causal_mask``), so resolving an operand of ``0`` to a padding
    index would emit a program that points somewhere it is not allowed to read --
    unrepresentable at execution, and invisible here because the value matches. The
    readable set is also not a prefix: the row's own operands ``[0, count)`` and the
    results written so far ``[n_operands, ...)``, with the padding in between excluded.

    Shorter programs are padded with multiplication by the constant ``1``. The
    register file is a fixed shape, so a program has to be exactly ``n_instr`` long
    and the answer has to land in the *last* register; ``x * 1`` is an exact identity
    in the ring -- it multiplies numerator and denominator by one, so it does not
    even grow the denominator -- and it carries the value forward one slot.
    """
    if not steps or len(steps) > n_instr:
        return None
    if len(reg_values) != n_operands:
        raise ValueError(f"expected {n_operands} operand slots, "
                         f"got {len(reg_values)}")
    n_real = n_operands if count is None else min(int(count), n_operands)
    readable: List[Tuple[int, Fraction]] = list(enumerate(reg_values[:n_real]))

    def find(v: Fraction) -> Optional[int]:
        for i, r in readable:
            if r == v:
                return i
        return None

    instrs: List[Instr] = []
    for a, op, b, res in steps:
        ia, ib = find(a), find(b)
        if ia is None or ib is None:
            return None
        instrs.append((ops.index(op), ia, ib))
        readable.append((n_operands + len(instrs) - 1, res))
    if len(instrs) < n_instr:
        # The padding instruction is ``x * 1``, so the identity slot has to actually
        # hold one and has to be readable -- checked rather than assumed. With
        # ``constants=()`` slot 0 is the first *quantity*, and padding by multiplying
        # through whatever that happens to be is exactly the silently wrong program
        # this layer exists to make unrepresentable. Only programs that need padding
        # are subject to it.
        if identity_slot >= n_real or reg_values[identity_slot] != 1:
            return None
        while len(instrs) < n_instr:
            last = n_operands + len(instrs) - 1
            instrs.append((ops.index("*"), last, identity_slot))
    return instrs


@dataclass
class Example:
    """One problem, already reduced to everything the machine needs."""

    text: str
    answer: Fraction
    values: List[int]
    scales: List[int]
    count: int
    program: Optional[List[Instr]] = None

    @property
    def fractions(self) -> List[Fraction]:
        return [Fraction(v, 10 ** s) for v, s in zip(self.values, self.scales)]


def build_examples(rows: Sequence[Dict[str, str]], n_operands: int, n_instr: int,
                   constants: Sequence[int] = DEFAULT_CONSTANTS,
                   lexical: bool = True) -> Tuple[List[Example], Dict[str, float]]:
    """Problems to examples, with the coverage numbers that decide what is trainable.

    Three failure rates, reported rather than absorbed, because every one of them
    caps what any model on top could possibly reach:

    ``no_answer``    the ``####`` line did not parse -- should be ~0.
    ``no_program``   no usable annotation chain, for any reason.
    ``answer_not_in_chain``
                     of the rows with usable steps, the share whose chain never
                     produces the final answer -- the step that did was compound (so
                     rejected) or unannotated. These are dropped: a chain that ends
                     somewhere else is supervision toward the wrong number, and the
                     model would learn it perfectly.
    ``operand_miss`` of the rows that had usable steps, the share that failed purely
                     on alignment. **If the numbers are not in the registers, no
                     program can be right**, so this is the ceiling on the supervised
                     arm and it has nothing to do with the network.
    """
    texts = [r["question"] for r in rows]
    quants = [all_quantities(t, lexical=lexical) for t in texts]
    vals, scales, counts = registers_from_quantities(quants, n_operands, constants)

    out: List[Example] = []
    n_no_answer = n_no_program = n_operand_miss = n_had_steps = 0
    n_answer_not_in_chain = 0
    for row, text, v, sc, c in zip(rows, texts, vals, scales, counts):
        ans = parse_final_answer(row.get("answer", ""))
        if ans is None:
            n_no_answer += 1
            continue
        ex = Example(text=text, answer=ans, values=v, scales=sc, count=c)
        steps, _ = parse_annotation_steps(row.get("answer", ""))
        # **The chain has to end on the answer, or it is not a program for this
        # problem.** Measured on GSM8K: 16.6% of usable chains end somewhere else,
        # almost always because the step that produced the answer was a compound
        # annotation this parser rejects, or was never annotated at all. Keeping those
        # meant 17.6% of "recovered" programs executed to the wrong number, which is
        # worse than having no program: it is supervision toward a wrong answer, and
        # the model would learn it perfectly. Truncating at the *last* step whose
        # result is the answer keeps the whole chain the dataset wrote and ends where
        # it has to; a chain that never reaches the answer is dropped.
        if steps:
            n_had_steps += 1
            cut = max((i for i, st in enumerate(steps) if st[3] == ans), default=None)
            if cut is None:
                n_answer_not_in_chain += 1
                n_no_program += 1
            else:
                prog = align_program(ex.fractions, steps[:cut + 1], n_operands,
                                     n_instr, count=c)
                if prog is None:
                    n_operand_miss += 1
                    n_no_program += 1
                else:
                    ex.program = prog
        else:
            n_no_program += 1
        out.append(ex)

    n = max(1, len(rows))
    stats = {
        "n": float(len(out)),
        "no_answer": n_no_answer / n,
        "no_program": n_no_program / n,
        "answer_not_in_chain": n_answer_not_in_chain / max(1, n_had_steps),
        "operand_miss": n_operand_miss / max(1, n_had_steps),
        "program_coverage": sum(1 for e in out if e.program is not None) / n,
    }
    return out, stats


def verify_alignment(examples: Sequence[Example], machine: RegisterMachine,
                     limit: int = 256) -> float:
    """Share of recovered programs that actually execute to the stated answer.

    The question that has to be asked before a single gradient step, and has nothing
    to do with a model: a program recovered from someone else's annotations is a
    hypothesis about what the dataset did. If executing it in this ring does not
    reproduce the dataset's own answer, the supervision is wrong and training on it
    would teach the wrong program perfectly.
    """
    have = [e for e in examples if e.program is not None][:limit]
    if not have:
        return 0.0
    regs = run_program(machine, [e.fractions for e in have],
                       [e.program for e in have])          # type: ignore[misc]
    # Decoded component-wise rather than through ``RationalAlgebra.decode``: that
    # builds the whole list at once, so one zero denominator raises and takes every
    # other row with it. A zero denominator here is a rejected alignment, not a
    # crash -- and finding out which is the entire point of this function.
    alg, sysm = machine.alg, machine.sys
    idx = machine.n_slots - 1
    nums = sysm.decode(alg.unpack(regs.num[:, idx]))
    dens = sysm.decode(alg.unpack(regs.den[:, idx]))
    ok = sum(1 for n, d, e in zip(nums, dens, have)
             if d != 0 and Fraction(n, d) == e.answer)
    return ok / len(have)


# ---------------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------------

@dataclass
class BridgeConfig:
    # data
    encoder: str = DEFAULT_ENCODER
    max_len: int = 192
    # Outside the source tree on purpose. The encoder cache is 1.3 GB of binary for
    # GSM8K at max_len=192, and putting build artefacts in the repo invites a desktop
    # file indexer to read all of it: measured on this box, KDE's Baloo sat at 77%
    # CPU and 3.7 GB RSS with the cache in `./.cache`, starving the study's workers to
    # 3.3% CPU each and making one 19-minute arm take ten hours. `.gitignore` does
    # not help -- an indexer does not read it.
    cache_dir: str = DEFAULT_CACHE
    lexical: bool = True
    constants: Tuple[int, ...] = DEFAULT_CONSTANTS

    # register file. Independent of each other, unlike the synthetic path where a
    # balanced tree of depth D forces 2^D operands and 2^D-1 instructions: a word
    # problem's arithmetic is a *chain*, not a tree, so it needs many operands and
    # few instructions.
    # 20, not 12, and the two are coupled to ``DEFAULT_CONSTANTS``: ten constants in a
    # 12-wide file fill 97% of rows and push real quantities out, halving coverage to
    # 0.249. At 20 the file binds on 1.2% of rows and coverage is 0.680 train / 0.691
    # test. Measured, because the naive version of this change is actively harmful.
    n_operands: int = 20
    # 8 is measured as sufficient: chain length is median 3, p90 6, p99 9, max 15, and
    # raising it to 12 buys +1.0 point of coverage for 50% more execution per step.
    n_instr: int = 8

    # model
    d_model: int = 256
    n_heads: int = 4
    n_resample: int = 32          # K vectors the problem is compressed to
    n_latent: int = 8             # latent positions; must be >= n_instr
    loops: int = 3
    recurrent_steps: int = 4

    # optimisation
    steps: int = 2000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    program_coef: float = 1.0     # 0.0 is the answer-only arm
    answer_coef: float = 1.0
    den_zero_coef: float = 1.0
    device: str = "auto"
    # Sized from the measured unreduced growth over GSM8K's own recovered programs
    # (worst case 2.91e9 against this ring's 4.04e11), not from a worst-case argument.
    # See lamb.rational.GSM8K_MODULI -- it is 5.9x cheaper per composition than
    # RATIONAL_MODULI and still has a 139x margin.
    rational_moduli: Tuple[int, ...] = GSM8K_MODULI


class BridgeReasoner(nn.Module):
    """Resampler -> shared latent core -> program heads.

    The core is the *same* class the arithmetic path uses, driven from embeddings
    instead of token ids. Its token embedding and LM head go unused, which is a real
    if small waste, and the alternative -- a forked core for language -- would give
    up the one property this design is for: there is no language model here, only a
    peripheral feeding the reasoner that already exists.
    """

    def __init__(self, cfg: BridgeConfig, d_enc: int, device: str = "cpu"):
        super().__init__()
        from .lotus import LotusReasoner

        if cfg.n_latent < cfg.n_instr:
            raise ValueError(f"n_latent={cfg.n_latent} < {cfg.n_instr} instructions")
        mcfg = ModelConfig(d_model=cfg.d_model, n_heads=cfg.n_heads,
                           d_ff=2 * cfg.d_model, n_prelude=1, n_recurrent=1,
                           n_coda=1, recurrent_steps=cfg.recurrent_steps)
        # ``build_model`` rather than ``LAMb(cfg)`` directly: it is what sets
        # ``cfg.vocab_size`` from the tokenizer, and every other entry point goes
        # through it. The vocabulary is unused on this path -- the bridge never
        # embeds a token or decodes one -- but the core is the *same* class the
        # arithmetic path uses, and forking it to save an unused embedding would give
        # up the one property this design is for.
        self.front = LanguageFront(d_enc, cfg.d_model, n_latents=cfg.n_resample)
        self.core = LotusReasoner(build_model(mcfg, ArithmeticTokenizer()),
                                  cfg.n_latent, cfg.loops)
        self.machine = RegisterMachine(cfg.d_model, cfg.n_operands, cfg.n_instr,
                                       ResidueSystem(tuple(cfg.rational_moduli)),
                                       device=device, rational=True)

    def forward(self, enc: torch.Tensor, pad_mask: torch.Tensor,
                values: Sequence[Sequence[Fraction]], counts: torch.Tensor,
                tau: float = 0.0, hard: bool = False):
        x = self.front(enc, pad_mask)                    # (B, K, d) -- length-invariant
        # The resampler's output has no padding by construction: K learned queries
        # attend over whatever was there, so a paragraph and a sentence both arrive
        # as exactly K vectors. That is the whole reason it is here.
        pad = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        _, _, latent_h, _ = self.core.latent_block(x, pad)
        return self.machine.run(latent_h, values, tau, hard, counts=counts)


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

class BridgeTrainer:
    """Train the resampler and the program heads. Nothing else moves."""

    def __init__(self, cfg: BridgeConfig, train: Sequence[Example],
                 enc_train: Dict[str, torch.Tensor],
                 test: Optional[Sequence[Example]] = None,
                 enc_test: Optional[Dict[str, torch.Tensor]] = None):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.train_ex, self.test_ex = list(train), list(test or [])
        self.enc_train, self.enc_test = enc_train, enc_test
        d_enc = int(enc_train["enc"].shape[-1])
        self.model = BridgeReasoner(cfg, d_enc, str(self.device)).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self._rng = torch.Generator().manual_seed(cfg.seed)
        self.out_reg = cfg.n_operands + cfg.n_instr - 1

    # -- one batch ---------------------------------------------------------
    def _batch(self, idx: Sequence[int], examples, enc):
        rows = [examples[i] for i in idx]
        e = enc["enc"][list(idx)].to(self.device).float()
        m = enc["pad_mask"][list(idx)].to(self.device)
        counts = torch.tensor([r.count for r in rows], device=self.device)
        return rows, e, m, counts

    def _losses(self, idx: Sequence[int], examples, enc):
        rows, e, m, counts = self._batch(idx, examples, enc)
        regs, logits = self.model(e, m, [r.fractions for r in rows], counts)
        # Every row is in the ring or it is not a row: unlike the grammar, the ring
        # here is the 4.5e15 rational one and a grade-school quantity does not come
        # close, but the check is kept rather than assumed because the denominators
        # are what grow and ``max_den_magnitude`` is the thing that would say so.
        keep = torch.ones(len(rows), device=self.device)
        ans, extra = rational_answer_loss(self.model.machine, regs, self.out_reg,
                                          [r.answer for r in rows], keep,
                                          self.cfg.den_zero_coef)
        # Only a minority of rows carry a recoverable program, so the program loss
        # is masked to those. The rows without one still need *a* gold tensor of the
        # right shape to index into, so a placeholder is substituted and then
        # weighted to zero -- it never reaches the gradient, and the answer loss
        # covers those rows on its own, which is precisely what the answer-only arm
        # relies on.
        have = [i for i, r in enumerate(rows) if r.program is not None]
        if have:
            pm = torch.zeros(len(rows), device=self.device)
            pm[have] = 1.0
            filler = rows[have[0]].program
            golds = [r.program if r.program is not None else filler for r in rows]
            prog = self.model.machine.program_loss(logits, golds, pm)
        else:
            prog = torch.zeros((), device=self.device)
        return prog, ans, extra, regs, rows, logits

    def train_step(self, step: int) -> Dict[str, float]:
        self.model.train()
        n = len(self.train_ex)
        idx = torch.randint(0, n, (self.cfg.batch_size,),
                            generator=self._rng).tolist()
        prog, ans, extra, *_ = self._losses(idx, self.train_ex, self.enc_train)
        loss = self.cfg.program_coef * prog + self.cfg.answer_coef * ans
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        out = {"loss": float(loss.detach()), "program": float(prog.detach()),
               "answer": float(ans.detach())}
        out.update(extra)
        return out

    @torch.no_grad()
    def evaluate(self, n: int = 512) -> Dict[str, float]:
        """Exact-match on the official test split, plus the ring's own diagnostics."""
        if not self.test_ex or self.enc_test is None:
            return {}
        self.model.eval()
        idx = list(range(min(n, len(self.test_ex))))
        prog, ans, _, regs, rows, _ = self._losses(idx, self.test_ex, self.enc_test)
        alg, sysm = self.model.machine.alg, self.model.machine.sys
        nums = sysm.decode(alg.unpack(regs.num[:, self.out_reg]))
        dens = sysm.decode(alg.unpack(regs.den[:, self.out_reg]))
        got = [None if d == 0 else Fraction(nn_, d) for nn_, d in zip(nums, dens)]
        ok = sum(1 for g, r in zip(got, rows) if g is not None and g == r.answer)
        return {"exact_match": ok / max(1, len(rows)),
                "zero_den": sum(1 for g in got if g is None) / max(1, len(got)),
                "max_den_magnitude": float(max(abs(d) for d in dens)),
                "program_loss": float(prog), "answer_loss": float(ans)}


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def _encode_split(split: str, cfg: BridgeConfig, out_dir: str) -> str:
    rows = fetch_gsm8k(split, cfg.cache_dir)
    tag = cfg.encoder.replace("/", "_")
    path = os.path.join(out_dir, f"gsm8k-{split}-{tag}-{cfg.max_len}.pt")
    if os.path.exists(path):
        print(f"{split}: cache already at {path}")
        return path
    os.makedirs(out_dir, exist_ok=True)
    out = encode_dataset([r["question"] for r in rows], cfg.encoder, path,
                         max_len=cfg.max_len)
    print(f"{split}: {out['meta']['n']} problems, d_enc={out['meta']['d_enc']}, "
          f"truncated {out['meta']['truncated']} -> {path}")
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    p.add_argument("--encode", action="store_true",
                   help="run the frozen encoder over GSM8K and cache it, then exit")
    p.add_argument("--coverage", action="store_true",
                   help="report extraction recall and program alignment, then exit")
    p.add_argument("--train", action="store_true")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-operands", type=int, default=12)
    p.add_argument("--n-instr", type=int, default=8)
    p.add_argument("--n-latent", type=int, default=8)
    p.add_argument("--program-coef", type=float, default=1.0,
                   help="0.0 is the answer-only arm: no gold program anywhere")
    p.add_argument("--no-lexical", action="store_true",
                   help="digits only; the ablation that prices written numerals")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-every", type=int, default=250)
    a = p.parse_args(argv)

    cfg = BridgeConfig(encoder=a.encoder, cache_dir=a.cache, steps=a.steps,
                       batch_size=a.batch, d_model=a.d_model,
                       n_operands=a.n_operands, n_instr=a.n_instr,
                       n_latent=a.n_latent, program_coef=a.program_coef,
                       lexical=not a.no_lexical, seed=a.seed, device=a.device)

    if a.encode:
        for split in ("train", "test"):
            _encode_split(split, cfg, a.cache)
        return

    train_rows = fetch_gsm8k("train", a.cache)
    test_rows = fetch_gsm8k("test", a.cache)
    tr_ex, tr_stats = build_examples(train_rows, cfg.n_operands, cfg.n_instr,
                                     cfg.constants, cfg.lexical)
    te_ex, te_stats = build_examples(test_rows, cfg.n_operands, cfg.n_instr,
                                     cfg.constants, cfg.lexical)
    machine = RegisterMachine(8, cfg.n_operands, cfg.n_instr,
                              ResidueSystem(tuple(cfg.rational_moduli)),
                              rational=True)
    exec_ok = verify_alignment(tr_ex, machine)

    print("coverage -- the ceiling on anything downstream")
    for name, st in (("train", tr_stats), ("test", te_stats)):
        print(f"  {name:>5}  n={int(st['n'])}  no_answer {st['no_answer']:.3f}  "
              f"no_program {st['no_program']:.3f}  "
              f"answer_not_in_chain {st['answer_not_in_chain']:.3f}  "
              f"operand_miss {st['operand_miss']:.3f}  "
              f"program_coverage {st['program_coverage']:.3f}")
    print(f"  recovered programs that execute to the stated answer: {exec_ok:.3f}")
    if a.coverage:
        return

    if not a.train:
        p.error("nothing to do: pass --encode, --coverage or --train")

    tag = cfg.encoder.replace("/", "_")
    paths = {s: os.path.join(a.cache, f"gsm8k-{s}-{tag}-{cfg.max_len}.pt")
             for s in ("train", "test")}
    missing = [s for s, path in paths.items() if not os.path.exists(path)]
    if missing:
        p.error(f"no encoder cache for {missing}; run --encode first")
    tr = BridgeTrainer(cfg, tr_ex, load_encoded(paths["train"]),
                       te_ex, load_encoded(paths["test"]))
    for s in range(cfg.steps):
        m = tr.train_step(s)
        if (s + 1) % a.eval_every == 0 or s == 0:
            e = tr.evaluate(512)
            print(f"step {s + 1:>5}  loss {m['loss']:.4f}  prog {m['program']:.4f}  "
                  f"ans {m['answer']:.4f}  exact_match {e.get('exact_match', 0):.4f}")


if __name__ == "__main__":
    main()
