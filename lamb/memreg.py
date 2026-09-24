"""infContext meets the register machine: compute over a value you saw long ago.

The two pillars did not touch. `memory_bench` and `ruler_bench` measure *retrieval* and
compute nothing over what they retrieve; `regmachine` loads its operands exactly from the
prompt, because ROADMAP 3a-vi measured that a network cannot induce the digit->residue map
and so must be handed it. That left the claim a latent frontier model actually needs --
**reason over what you remembered, with O(1) state** -- untested anywhere.

The design here keeps the exactness and moves the difficulty to where it belongs:

    @3 = 47 ; @1 = 82 ; @6 = 19 ; ... ; @3 + @1 =

Every value in the context is loaded into a register **exactly**, by the same fixed map
the bridge uses. So no arithmetic is learned. What must be learned is *which register a key
refers to*, across however much context separates the binding from the query. That is
retrieval expressed as pointer selection, and the answer is then composed by exact algebra.

What this is and is not. It is not unbounded context: a file of R registers holds R values.
It is the composition that was missing, and the quantity it measures is the one that
matters, namely whether accuracy survives distance. The neural memory (`use_memory`) is
what has to carry the key->position association across the filler; with it off, the core
must do it with attention alone, which is the control.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .tokenizer import ArithmeticTokenizer, Encoded

OPS: Tuple[str, ...] = ("+", "-")


@dataclass
class MemRegConfig:
    n_keys: int = 8              # bindings in the context, and register-file width
    digits: int = 2
    n_query: int = 2             # operands the query names
    ops: Tuple[str, ...] = OPS
    seed: int = 0


class MemRegTask:
    """Bindings, then a query over two of them. Exact answer, exact gold program."""

    def __init__(self, cfg: MemRegConfig):
        self.cfg = cfg
        self.tok = ArithmeticTokenizer(n_keys=cfg.n_keys)

    def _value(self, rng: random.Random) -> int:
        d = max(1, self.cfg.digits)
        return rng.randint(0, 9) if d == 1 else rng.randint(10 ** (d - 1), 10 ** d - 1)

    def sample(self, seed: int, n_binding: Optional[int] = None):
        """One episode.

        ``n_binding`` is the context length in bindings, and the queried keys sit at
        **random** positions within it.

        The first version placed them first so that distance would be a clean constant.
        That made the correct pointer the constant ``(0, 1)``, and `ptr_acc` reached 1.000
        in three training steps: the model was not retrieving anything, it was emitting a
        fixed address. It is the same degeneracy that made every earlier induction claim
        in this repo a constant-recall claim (ROADMAP 3a-xv), reintroduced here by an
        attempt to make the measurement tidy.

        So the positions are drawn, and distance is reported per episode as how far the
        *farther* queried binding sits from the query. The length curve is then built by
        bucketing on it rather than by fixing it.
        """
        cfg = self.cfg
        rng = random.Random(seed)
        n = n_binding if n_binding is not None else cfg.n_keys
        n = max(cfg.n_query, min(n, cfg.n_keys))
        keys = list(range(cfg.n_keys))
        rng.shuffle(keys)
        keys = keys[:n]
        vals = [self._value(rng) for _ in keys]
        qi = rng.sample(range(n), cfg.n_query)        # random positions, not the first
        op = rng.choice(cfg.ops)
        a, b = vals[qi[0]], vals[qi[1]]
        answer = a + b if op == "+" else a - b
        return {
            "keys": keys, "vals": vals, "query": (keys[qi[0]], op, keys[qi[1]]),
            "ptr": (qi[0], qi[1]), "op": cfg.ops.index(op), "answer": answer,
            # how far back the *farther* of the two bindings is from the query
            "distance": n - min(qi),
        }

    # -- tokenisation ------------------------------------------------------
    def encode(self, ep: Dict) -> Encoded:
        """``[BOS] @k = v ; ... ; @i op @j =``, built with the tokenizer's own emitters.

        The value channel is populated on every binding's digits, so magnitude is
        available wherever a number appears; it is what the register loader reads.
        """
        t = self.tok
        enc = Encoded(ids=[], abacus=[], value=[], value_mask=[], ans_start=0)
        t._emit_symbol(enc, t.BOS)
        for k, v in zip(ep["keys"], ep["vals"]):
            t._emit_symbol(enc, t.KEY0 + k)
            t._emit_symbol(enc, t.EQ)
            t._emit_number(enc, str(v), on_problem_side=True)
            t._emit_symbol(enc, t.SEP)
        ki, op, kj = ep["query"]
        t._emit_symbol(enc, t.KEY0 + ki)
        t._emit_symbol(enc, t._op_ids[op])
        t._emit_symbol(enc, t.KEY0 + kj)
        t._emit_symbol(enc, t.EQ)
        enc.ans_start = len(enc.ids)
        return enc

    def collate(self, eps: Sequence[Dict], device: str = "cpu"):
        """Left-padded prompts, so the query sits at a common column for every row."""
        encs = [self.encode(e) for e in eps]
        w = max(len(e) for e in encs)
        b = len(encs)
        ids = torch.zeros(b, w, dtype=torch.long, device=device)
        ab = torch.zeros(b, w, dtype=torch.long, device=device)
        val = torch.zeros(b, w, device=device)
        vm = torch.zeros(b, w, device=device)
        pad = torch.ones(b, w, dtype=torch.bool, device=device)
        for i, e in enumerate(encs):
            off = w - len(e)
            ids[i, off:] = torch.tensor(e.ids, device=device)
            ab[i, off:] = torch.tensor(e.abacus, device=device)
            val[i, off:] = torch.tensor(e.value, device=device)
            vm[i, off:] = torch.tensor(e.value_mask, device=device)
            pad[i, off:] = False
        return {"input_ids": ids, "abacus_ids": ab, "value": val,
                "value_mask": vm, "pad_mask": pad}

    # -- what the register machine needs -----------------------------------
    def registers(self, eps: Sequence[Dict]) -> Tuple[List[List[int]], List[int]]:
        """Values in presentation order, padded to ``n_keys``, plus the real count.

        Loaded exactly. Nothing about arithmetic is learned here; the only thing the
        model has to produce is a pointer to the register a key named.
        """
        n = self.cfg.n_keys
        vals, counts = [], []
        for e in eps:
            v = list(e["vals"])[:n]
            counts.append(len(v))
            vals.append(v + [0] * (n - len(v)))
        return vals, counts

    def gold(self, eps: Sequence[Dict]) -> List[List[Tuple[int, int, int]]]:
        """One instruction: the op, and the two registers the query's keys landed in."""
        return [[(e["op"], e["ptr"][0], e["ptr"][1])] for e in eps]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class MemRegReasoner(torch.nn.Module):
    """Latent block over the context, register machine over the values it names."""

    def __init__(self, cfg: MemRegConfig, model_cfg, n_latent: int = 4,
                 loops: int = 3, moduli=(16, 25, 27, 11, 37), device: str = "cpu"):
        super().__init__()
        from .algebra import ResidueSystem
        from .lotus import LotusReasoner
        from .model.lamb import build_model
        from .regmachine import RegisterMachine

        tok = ArithmeticTokenizer(n_keys=cfg.n_keys)
        model_cfg.vocab_size = tok.vocab_size
        self.core = LotusReasoner(build_model(model_cfg, tok), n_latent, loops)
        self.machine = RegisterMachine(model_cfg.d_model, cfg.n_keys, 1,
                                       ResidueSystem(tuple(moduli)), device=device)

    def forward(self, prompt, values, counts, tau: float = 0.0, hard: bool = False):
        m = self.core.model
        x = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                    prompt["value"], prompt["value_mask"])
        _, _, latent_h, _ = self.core.latent_block(x, prompt["pad_mask"])
        return self.machine.run(latent_h, values, tau, hard, counts=counts)


class MemRegTrainer:
    """Train it, then read accuracy as a function of distance.

    The only measurement that matters here is the length curve: a model that resolves a
    key one binding back and fails ten back has not composed the two pillars, it has
    learned to look at the previous token.
    """

    def __init__(self, cfg: MemRegConfig, model_cfg, n_latent: int = 4, loops: int = 3,
                 lr: float = 3e-4, device: str = "cpu", program_coef: float = 1.0):
        torch.manual_seed(cfg.seed)
        self.cfg, self.device = cfg, device
        self.task = MemRegTask(cfg)
        self.model = MemRegReasoner(cfg, model_cfg, n_latent, loops,
                                    device=device).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.program_coef = program_coef
        self._seed = cfg.seed * 10_000

    def _batch(self, n: int, n_binding: Optional[int] = None, train: bool = True):
        eps = []
        for _ in range(n):
            self._seed += 1
            # Training draws every context length so the model cannot specialise on one
            # distance; evaluation fixes it, which is what makes the curve a curve.
            nb = n_binding or torch.randint(self.cfg.n_query, self.cfg.n_keys + 1,
                                            (1,)).item()
            eps.append(self.task.sample(self._seed, nb))
        return eps

    def _loss(self, eps):
        prompt = self.task.collate(eps, self.device)
        vals, counts = self.task.registers(eps)
        cnt = torch.tensor(counts, device=self.device)
        regs, logits = self.model(prompt, vals, cnt)
        golds = self.task.gold(eps)
        prog = self.model.machine.program_loss(logits, golds)
        tgt = self.model.machine.sys.targets([e["answer"] for e in eps], device=self.device)
        out_reg = self.cfg.n_keys          # n_operands + n_instr - 1
        ans = sum(torch.nn.functional.nll_loss(
                      torch.log(b[:, out_reg].clamp_min(1e-9)), tgt[:, k])
                  for k, b in enumerate(regs.blocks)) / len(regs.blocks)
        return prog, ans, regs, logits

    def train_step(self, batch: int = 64) -> Dict[str, float]:
        self.model.train()
        prog, ans, *_ = self._loss(self._batch(batch))
        loss = self.program_coef * prog + ans
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {float(loss)}")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return {"loss": float(loss.detach()), "program": float(prog.detach()),
                "answer": float(ans.detach())}

    @torch.no_grad()
    def accuracy(self, n: int = 256, n_binding: Optional[int] = None) -> Dict[str, float]:
        from .regmachine import run_program

        self.model.eval()
        eps = self._batch(n, n_binding=n_binding)
        prompt = self.task.collate(eps, self.device)
        vals, counts = self.task.registers(eps)
        cnt = torch.tensor(counts, device=self.device)
        regs, logits = self.model(prompt, vals, cnt)
        out_reg = self.cfg.n_keys
        oi, ai, bi = (x.argmax(-1) for x in logits)
        progs = [[(int(oi[i, 0]), int(ai[i, 0]), int(bi[i, 0]))] for i in range(len(eps))]
        got = run_program(self.model.machine, vals, progs, self.device).decode(out_reg)
        answers = [e["answer"] for e in eps]
        acc = sum(1 for g, a in zip(got, answers) if g == a) / len(answers)
        ptr = sum(1 for i, e in enumerate(eps)
                  if (int(ai[i, 0]), int(bi[i, 0])) == e["ptr"]) / len(eps)
        # The refusal signal of 3a-xix, which needs no labels and should transfer here.
        conf = torch.stack([b[:, out_reg].max(-1).values for b in regs.blocks]).mean(0)
        keep = [i for i in range(len(eps)) if float(conf[i]) >= 0.9]
        out = {"acc": acc, "ptr_acc": ptr,
               "conf_cover": len(keep) / len(eps),
               "acc_conf": (sum(1 for i in keep if got[i] == answers[i]) / len(keep))
                           if keep else float("nan")}
        # Accuracy bucketed by how far back the binding was. A model that resolves one
        # binding back and fails ten has not composed the pillars, it has learned to look
        # at the previous token, and only the curve distinguishes the two.
        by_d: Dict[int, List[int]] = {}
        for i, e in enumerate(eps):
            by_d.setdefault(e["distance"], []).append(int(got[i] == answers[i]))
        out["by_distance"] = {d: sum(v) / len(v) for d, v in sorted(by_d.items())}
        return out


# ---------------------------------------------------------------------------
# Design 2: more bindings than registers, so something must choose
# ---------------------------------------------------------------------------

class SelectiveLoader(torch.nn.Module):
    """Choose which `R` of `N` bindings occupy the register file.

    Design 1 (`MemRegReasoner`) put every binding in a register, which left no retrieval
    to do: order-indexed registers made the pointer an uncountable function of position,
    and key-indexed registers made it a constant. Retrieval only exists when the file is
    too small for the context.

    The wall this has to avoid is ROADMAP 3a-vi: a network cannot learn the digit->residue
    map, so nothing learned may sit between a value and its code. It is avoided by
    selecting **positions, not values**. The head emits one distribution per register slot
    over the `N` candidate bindings; the candidates' codes are the exact ones the fixed map
    produced; and the register is their mixture. Nothing about arithmetic or encoding is
    learned, only *which digits to look at*.

    That makes it the same object as a pointer read (`RegisterFile.read`): a mixture of
    residue distributions is a residue distribution, so it is differentiable, and a sharp
    selection is exact. It inherits the same caveat as every soft read here -- decoding
    argmaxes each modulus independently, so a blurred selection can decode to a value in
    no candidate at all (3a-x) -- which is contained by selections going sharp and by the
    refusal signal of 3a-xix.
    """

    def __init__(self, d_model: int, n_registers: int, n_candidates: int):
        super().__init__()
        self.n_registers = n_registers
        self.n_candidates = n_candidates
        self.head = torch.nn.Linear(d_model, n_candidates)

    def logits(self, latent_h: torch.Tensor) -> torch.Tensor:
        """``(B, R, N)``: for each register slot, where in the context to read from."""
        return self.head(latent_h[:, : self.n_registers])

    def load(self, latent_h: torch.Tensor, codes: torch.Tensor,
             tau: float = 0.0, hard: bool = False):
        """``codes`` is ``(B, N, K, P)``, the exact code of every candidate binding.

        Returns ``((B, R, K, P)`` register contents, selection logits).
        """
        lg = self.logits(latent_h)
        w = (torch.softmax(lg, dim=-1) if tau <= 0 else
             torch.nn.functional.gumbel_softmax(lg, tau=tau, hard=hard, dim=-1))
        return torch.einsum("brn,bnkp->brkp", w, codes), lg

    def select_loss(self, lg: torch.Tensor, gold: Sequence[Sequence[int]]) -> torch.Tensor:
        """Cross-entropy on which binding each register should have taken.

        Free here for the same reason the gold program is free elsewhere: the generator
        knows which binding holds the queried key.
        """
        tgt = torch.tensor(gold, device=lg.device)
        return torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.size(-1)), tgt.reshape(-1), ignore_index=-100)


@dataclass
class MemRegConfig2:
    """Design 2: ``n_binding`` values in the context, only ``n_registers`` slots."""

    n_keys: int = 16             # bindings in the context (the candidate set)
    n_registers: int = 4         # slots available, deliberately fewer
    digits: int = 2
    ops: Tuple[str, ...] = OPS
    seed: int = 0


class MemRegTask2(MemRegTask):
    """Same context, but the queried bindings must be *selected* into a small file.

    The register slots the queried values land in are **drawn**, not fixed. Assigning them
    to slots 0 and 1 would make the program's pointer the constant ``(0, 1)`` and the whole
    thing would measure selection only, with the program degenerate. That is the third time
    this degeneracy has had to be designed out (ROADMAP 3a-xv, and design 1 above).
    """

    def __init__(self, cfg2: MemRegConfig2):
        self.cfg2 = cfg2
        super().__init__(MemRegConfig(n_keys=cfg2.n_keys, digits=cfg2.digits,
                                      n_query=2, ops=cfg2.ops, seed=cfg2.seed))

    def sample2(self, seed: int, n_binding: Optional[int] = None) -> Dict:
        c2 = self.cfg2
        rng = random.Random(seed ^ 0x5EED)
        n = n_binding or c2.n_keys
        n = max(2, min(n, c2.n_keys))
        keys = list(range(c2.n_keys))
        rng.shuffle(keys)
        keys = keys[:n]
        vals = [self._value(rng) for _ in keys]
        qi = rng.sample(range(n), 2)                  # positions in the context
        slots = rng.sample(range(c2.n_registers), 2)  # where they must be loaded
        op = rng.choice(c2.ops)
        a, b = vals[qi[0]], vals[qi[1]]
        # Unqueried slots carry ``-100``, the ignore index: supervising them toward a
        # drawn binding injects irreducible noise into the loss for no reason, since
        # which filler a spare slot holds is genuinely arbitrary.
        sel = [-100] * c2.n_registers
        sel[slots[0]], sel[slots[1]] = qi[0], qi[1]
        return {
            "keys": keys, "vals": vals, "query": (keys[qi[0]], op, keys[qi[1]]),
            "sel": sel, "ptr": (slots[0], slots[1]), "op": c2.ops.index(op),
            "answer": a + b if op == "+" else a - b,
            "distance": n - min(qi), "n": n,
        }

    def candidate_codes(self, eps: Sequence[Dict], alg, sysm, device: str = "cpu"):
        """``(B, N, K, P)`` exact codes for every binding, padded over short contexts."""
        n = self.cfg2.n_keys
        flat, = [[v for e in eps for v in (list(e["vals"]) + [0] * (n - len(e["vals"])))]]
        codes = alg.pack(sysm.split(sysm.onehot(flat, device)))
        return codes.view(len(eps), n, codes.shape[-2], codes.shape[-1])

    def gold2(self, eps: Sequence[Dict]):
        return ([e["sel"] for e in eps],
                [[(e["op"], e["ptr"][0], e["ptr"][1])] for e in eps])


class MemRegTrainer2:
    """Design 2 end to end: select into a small file, then compute exactly.

    Three losses, all free from the generator: which binding each slot should take, the
    program over the slots, and the answer's residues. The answer loss reaches both heads
    through exact arithmetic, so the selection is correctable by the result being wrong.
    """

    def __init__(self, cfg2: MemRegConfig2, model_cfg, n_latent: int = 8, loops: int = 3,
                 lr: float = 3e-4, device: str = "cpu", moduli=(16, 25, 27, 11, 37),
                 select_coef: float = 1.0, program_coef: float = 1.0):
        from .algebra import ResidueAlgebra, ResidueSystem
        from .lotus import LotusReasoner
        from .model.lamb import build_model
        from .regmachine import RegisterFile, RegisterMachine

        torch.manual_seed(cfg2.seed)
        self.cfg2, self.device = cfg2, device
        self.task = MemRegTask2(cfg2)
        self.sys = ResidueSystem(tuple(moduli))
        self.alg = ResidueAlgebra(self.sys, device=device)
        tok = ArithmeticTokenizer(n_keys=cfg2.n_keys)
        model_cfg.vocab_size = tok.vocab_size
        if n_latent < cfg2.n_registers + 1:
            raise ValueError(f"n_latent={n_latent} must hold {cfg2.n_registers} selection "
                             f"slots plus 1 instruction")
        self.core = LotusReasoner(build_model(model_cfg, tok), n_latent, loops).to(device)
        self.loader = SelectiveLoader(model_cfg.d_model, cfg2.n_registers,
                                      cfg2.n_keys).to(device)
        self.machine = RegisterMachine(model_cfg.d_model, cfg2.n_registers, 1,
                                       self.sys, device=device).to(device)
        self._RF = RegisterFile
        self.opt = torch.optim.AdamW(
            list(self.core.parameters()) + list(self.loader.parameters())
            + list(self.machine.parameters()), lr=lr, weight_decay=0.01)
        self.select_coef, self.program_coef = select_coef, program_coef
        self._seed = cfg2.seed * 10_000

    def _batch(self, n: int, n_binding: Optional[int] = None):
        eps = []
        for _ in range(n):
            self._seed += 1
            eps.append(self.task.sample2(self._seed, n_binding))
        return eps

    def _forward(self, eps, tau: float = 0.0, hard: bool = False):
        from .regmachine import execute

        prompt = self.task.collate(eps, self.device)
        m = self.core.model
        x = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                    prompt["value"], prompt["value_mask"])
        _, _, latent_h, _ = self.core.latent_block(x, prompt["pad_mask"])
        codes = self.task.candidate_codes(eps, self.alg, self.sys, self.device)
        loaded, sel_lg = self.loader.load(latent_h, codes, tau, hard)

        # The loaded registers *are* the file. Built by hand rather than through
        # RegisterFile's integer constructor, because their contents are distributions
        # over selected candidates and not exact integers.
        R = self.cfg2.n_registers
        regs = self._RF(self.sys, [[0] * R] * len(eps), self.device, n_total=R + 1)
        regs.packed = torch.cat(
            [loaded, torch.zeros_like(loaded[:, :1])], dim=1)
        regs.n_filled = R
        h_prog = latent_h[:, R:]
        op_l, a_l, b_l = self.machine.logits(h_prog)
        # ``[:, 0]`` because ``execute`` takes pointers as ``(B, R)`` for one instruction;
        # the logits carry an instruction axis that has to be indexed away first.
        ow, pa, pb = (self.machine._select(z[:, 0], tau, hard)
                      for z in (op_l, a_l, b_l))
        regs.append(execute(self.machine.alg, regs, ow, pa, pb, self.machine.ops))
        return regs, (op_l, a_l, b_l), sel_lg

    def train_step(self, batch: int = 64) -> Dict[str, float]:
        self.core.train(); self.loader.train(); self.machine.train()
        eps = self._batch(batch)
        regs, logits, sel_lg = self._forward(eps)
        sel_gold, prog_gold = self.task.gold2(eps)
        sel = self.loader.select_loss(sel_lg, sel_gold)
        prog = self.machine.program_loss(logits, prog_gold)
        tgt = self.sys.targets([e["answer"] for e in eps], device=self.device)
        out_reg = self.cfg2.n_registers
        ans = sum(torch.nn.functional.nll_loss(
                      torch.log(b[:, out_reg].clamp_min(1e-9)), tgt[:, k])
                  for k, b in enumerate(regs.blocks)) / len(regs.blocks)
        loss = self.select_coef * sel + self.program_coef * prog + ans
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {float(loss)}")
        torch.nn.utils.clip_grad_norm_(
            list(self.core.parameters()) + list(self.loader.parameters())
            + list(self.machine.parameters()), 1.0)
        self.opt.step()
        return {"loss": float(loss.detach()), "select": float(sel.detach()),
                "program": float(prog.detach()), "answer": float(ans.detach())}

    @torch.no_grad()
    def accuracy(self, n: int = 256, n_binding: Optional[int] = None) -> Dict:
        self.core.eval(); self.loader.eval(); self.machine.eval()
        eps = self._batch(n, n_binding)
        regs, logits, sel_lg = self._forward(eps)
        out_reg = self.cfg2.n_registers
        got = regs.decode(out_reg)
        answers = [e["answer"] for e in eps]
        sel_gold, _ = self.task.gold2(eps)
        # Selection accuracy on the two slots that matter; the filler slots are noise.
        hit = 0
        pick = sel_lg.argmax(-1)
        for i, e in enumerate(eps):
            s0, s1 = e["ptr"]
            hit += int(int(pick[i, s0]) == e["sel"][s0]
                       and int(pick[i, s1]) == e["sel"][s1])
        conf = torch.stack([b[:, out_reg].max(-1).values for b in regs.blocks]).mean(0)
        keep = [i for i in range(len(eps)) if float(conf[i]) >= 0.9]
        by_d: Dict[int, List[int]] = {}
        for i, e in enumerate(eps):
            by_d.setdefault(e["distance"], []).append(int(got[i] == answers[i]))
        return {"acc": sum(1 for g, a in zip(got, answers) if g == a) / len(answers),
                "select_acc": hit / len(eps),
                "conf_cover": len(keep) / len(eps),
                "acc_conf": (sum(1 for i in keep if got[i] == answers[i]) / len(keep))
                            if keep else float("nan"),
                "by_distance": {d: sum(v) / len(v) for d, v in sorted(by_d.items())}}
