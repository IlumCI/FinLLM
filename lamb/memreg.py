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
