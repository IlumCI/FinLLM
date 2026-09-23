"""Stage A, restructured -- LOTUS-style parallel supervised latent reasoning.

Stage A's first design followed Coconut: continuous thoughts generated
**autoregressively**, one latent at a time, each conditioned on the last, with only
the final answer supervised. It worked at tiny scale -- and the literature says
that is exactly where it works. Measured across backbones (arXiv:2606.31779), the
sequential continuous-thought family's gap to explicit chain-of-thought *widens*
with scale (-0.1 pts at 124M, -2.3 at 1B, **-9.2 at 3B**), while the looped,
parallel-supervised family stays flat (-1.5 at 3B). If the point is to scale, the
parallel family is the one to build.

The restructure, then:

* **Parallel, not autoregressive.** ``n_latent`` latent positions are appended
  after the prompt *all at once* and refined by ``loops`` passes through the shared
  core. Cost is ``O(loops)`` forwards **regardless of how many latents there are** --
  where Coconut needed one sequential forward per thought. That is the scalability
  argument: the latent budget grows for free.
* **Every latent position is supervised**, directly through the LM head, instead of
  hoping one answer gradient reaches back through a chain of continuous states.

LOTUS supervises latents against gold chain-of-thought *tokens*. LAMb has no
language -- but it has an exact evaluator, so it generates a gold **numeric** trace
(the intermediate sub-expression values) for free, with no language, no human
annotation and no external data. The self-play verifier supplies precisely the
signal the method needs.

Crucially this costs nothing at inference: the latent positions are **never
decoded**. Only the answer is emitted. The trace is a training signal, not an
output, so the model remains a blackbox that computes its intermediates latently.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .alu import LatentALU, ScalarALU, parse_expr, slot_order
from .algebra import ResidueSystem
from .coconut import _masked_ce, coconut_collate
from .comm import _decode_answer
from .config import LotusConfig, ModelConfig
from .holdout import is_heldout
from .device import Amp, add_hardware_args, device_report, resolve_device, resolve_hardware
from .model.lamb import LAMb, build_model
from .model.transformer import RMSNorm
from .selfplay.grammar import Descriptor, TaskGrammar
from .selfplay.verifier import Verifier
from .tokenizer import ArithmeticTokenizer

# (expression, exact answer, intermediate trace values)
Task = Tuple[str, str, List[int]]


def trace_targets(tok: ArithmeticTokenizer, trace: List[int]) -> List[int]:
    """Tokenise the numeric trace into supervision targets (ids only).

    The trace is a *target*, never an input, so no Abacus/value channels are
    needed. Each intermediate is emitted exactly as an answer would be (LSB-first
    digits, a leading ``-`` for negatives); consecutive intermediates are simply
    concatenated -- the Abacus reset at each number boundary is what separates them.
    """
    ids: List[int] = []
    for v in trace:
        e = tok.encode("0", str(v))       # dummy problem; take only the answer span
        ids += e.ids[e.ans_start:-1]      # drop the trailing EOS
    return ids


class LotusReasoner(nn.Module):
    """A LAMb core plus a parallel block of supervised latent positions."""

    def __init__(self, model: LAMb, n_latent: int, loops: int,
                 bot_id: Optional[int] = None, eot_id: Optional[int] = None,
                 trace_compress: int = 1, space_dim: int = 0,
                 alu_moduli: Optional[Tuple[int, ...]] = None,
                 alu_mode: str = "residue"):
        super().__init__()
        self.model = model
        self.n_latent = int(n_latent)
        self.loops = max(1, int(loops))
        self.trace_compress = max(1, int(trace_compress))
        # SWITCH boundaries; None on both disables them (the ablation).
        self.bot_id = bot_id
        self.eot_id = eot_id
        d = model.cfg.d_model
        # Learned initial content for each latent slot. They become input-dependent
        # through attention over the prompt, which precedes them causally.
        self.latent_emb = nn.Embedding(self.n_latent, d)
        nn.init.normal_(self.latent_emb.weight, mean=0.0, std=0.02)
        # Feedback interface, mirroring the Coconut path: bounded magnitude plus a
        # learned tag marking a position as latent rather than a token.
        self.latent_norm = RMSNorm(d)
        self.latent_marker = nn.Parameter(torch.zeros(d))
        # Multi-token prediction over the latent block (arXiv:2404.19737's heads,
        # applied to latents rather than tokens). With ``trace_compress = c`` each
        # latent position is supervised against ``c`` consecutive trace tokens, so
        # the latent budget stops being a copy of the trace length: a trace of T
        # tokens needs ceil(T / c) positions instead of T. That matters because
        # arXiv:2607.16972 finds both continuous-CoT training regimes collapse to
        # about a third of explicit-CoT accuracy on *long* traces -- and a design
        # that needs one latent per trace token can never reach a long trace at all.
        #
        # The compression is generative, not geometric. C-MTP compresses by making
        # a latent the *average* of the token embeddings it stands for; the
        # information-theoretic analysis in arXiv:2606.20075 finds that kind of
        # rigid geometric compression collapses the reasoning space, while
        # generative reconstruction preserves its capacity. So each head decodes
        # its own token through the shared LM head and nothing is averaged.
        # Head 0 is the identity, which makes ``trace_compress=1`` bit-for-bit the
        # uncompressed model rather than merely equivalent to it.
        self.mtp_heads = nn.ModuleList()
        for _ in range(self.trace_compress - 1):
            lin = nn.Linear(d, d)
            nn.init.eye_(lin.weight)           # start each head near the identity
            nn.init.zeros_(lin.bias)
            self.mtp_heads.append(nn.Sequential(RMSNorm(d), lin))
        # Space supervision head (arXiv:2606.20075). That analysis decomposes
        # process supervision into *trajectory* supervision -- dense stepwise
        # signal, which the trace loss already supplies -- and *space* supervision,
        # which preserves the semantic structure of the latent manifold.
        # Trajectory supervision alone leaves the latents free to drift and
        # collapse in every direction the readout does not constrain, which is the
        # "dual collapse" that analysis identifies. Nothing here supplied the
        # second term, so this head adds it: the projection the contrastive
        # objective in :meth:`LotusTrainer._space_loss` operates on.
        # ``space_dim = 0`` builds no head at all, so it is off, not merely zeroed.
        self.space_head = (nn.Sequential(RMSNorm(d), nn.Linear(d, space_dim))
                           if space_dim > 0 else None)
        # The latent ALU (:mod:`lamb.alu`). Built only when asked for, so the
        # parameter count is untouched when it is off.
        if alu_moduli is None:
            self.alu = None
        elif alu_mode == "scalar":
            self.alu = ScalarALU(d)
        else:
            self.alu = LatentALU(d, ResidueSystem(tuple(alu_moduli)))

    def _marker(self, token_id: int, b: int, device) -> torch.Tensor:
        """Embed a boundary token as an ordinary token (B, 1, d)."""
        ids = torch.full((b, 1), token_id, dtype=torch.long, device=device)
        zl = torch.zeros((b, 1), dtype=torch.long, device=device)
        zf = torch.zeros((b, 1), dtype=torch.float32, device=device)
        return self.model.embed(ids, zl, zf, zf)

    def latent_block(self, x_prompt: torch.Tensor, pad_prompt: torch.Tensor,
                     n_steps: Optional[int] = None):
        """Wrap L latent positions in boundaries and refine them with ``loops`` passes.

        Layout: ``[prompt] [BOT] [latent x L] [EOT]``. Returns
        ``(x, pad, latent_hidden, boundary_hidden)``. One core forward per loop
        iteration -- independent of ``n_latent``, because the positions are refined
        together. ``boundary_hidden`` is the state at the BOT position, where
        arXiv:2606.13106 finds the latent computation concentrates; it is the handle
        a probe attaches to.
        """
        b, p, _ = x_prompt.shape
        dev = x_prompt.device
        head = [x_prompt]
        if self.bot_id is not None:
            head.append(self._marker(self.bot_id, b, dev))
        head_x = torch.cat(head, dim=1)
        l0 = head_x.size(1)                       # index where the latents begin
        tail_x = self._marker(self.eot_id, b, dev) if self.eot_id is not None else None

        idx = torch.arange(self.n_latent, device=dev)
        lat = self.latent_emb(idx).unsqueeze(0).expand(b, -1, -1)
        n_extra = (l0 - p) + (0 if tail_x is None else 1) + self.n_latent
        pad = torch.cat(
            [pad_prompt, torch.zeros(b, n_extra, dtype=torch.bool, device=dev)], dim=1)

        def assemble(latents):
            parts = [head_x, latents] + ([] if tail_x is None else [tail_x])
            return torch.cat(parts, dim=1)

        x = assemble(lat)
        h = None
        for _ in range(self.loops):
            h, _ = self.model.core(x, pad, n_steps)
            lat = self.latent_norm(h[:, l0:l0 + self.n_latent, :]) + self.latent_marker
            x = assemble(lat)                     # prompt/boundaries stay as embedded input
        latent_h = h[:, l0:l0 + self.n_latent, :]
        boundary_h = h[:, l0 - 1, :] if self.bot_id is not None else None
        return x, pad, latent_h, boundary_h

    def trace_logits(self, latent_h: torch.Tensor) -> torch.Tensor:
        """Per-latent trace logits, ``(B, L * trace_compress, V)``.

        Head ``j`` predicts the ``j``-th trace token owned by each latent position,
        so the flattened index is ``i * c + j`` -- the same order ``_collate`` lays
        the targets out in. At ``c = 1`` this is exactly ``_readout(latent_h)``.
        """
        outs = [self.model._readout(latent_h)]
        for head in self.mtp_heads:
            outs.append(self.model._readout(head(latent_h)))
        if len(outs) == 1:
            return outs[0]
        # (B, L, c, V) -> (B, L*c, V), offset-major within each position
        stacked = torch.stack(outs, dim=2)
        b, l, c, v = stacked.shape
        return stacked.reshape(b, l * c, v)

    def forward(self, prompt: Dict[str, torch.Tensor], answer_ids: torch.Tensor,
                answer_abacus: torch.Tensor, answer_pad: torch.Tensor,
                n_steps: Optional[int] = None):
        """Answer logits, per-latent-position logits, and the entry-switch logits.

        ``switch_logits`` is the distribution at the last prompt position, which
        predicts the BOT boundary. That is the well-defined probability the latent
        segment otherwise lacks -- the hook for on-policy RL.
        """
        m = self.model
        x_prompt = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                           prompt["value"], prompt["value_mask"])
        p = x_prompt.size(1)
        x, pad, latent_h, _ = self.latent_block(x_prompt, prompt["pad_mask"], n_steps)
        latent_logits = self.trace_logits(latent_h)              # (B, L*c, V) -> trace supervision

        z = torch.zeros_like(answer_abacus, dtype=torch.float32)
        x = torch.cat([x, m.embed(answer_ids, answer_abacus, z, z)], dim=1)
        pad = torch.cat([pad, answer_pad], dim=1)
        h, aux = m.core(x, pad, n_steps)
        logits = m._readout(h)
        ans_logits = logits[:, -(answer_ids.size(1) + 1):, :]
        # Prompts are left-padded, so the last real prompt token is at p-1 for every
        # row; that position is where "enter latent reasoning" is predicted.
        switch_logits = logits[:, p - 1, :] if self.bot_id is not None else None
        aux = dict(aux)                     # copy: never mutate the core's own aux
        aux["latent_h"] = latent_h          # the space loss and collapse diagnostic
        return ans_logits, latent_logits, switch_logits, aux

    @torch.no_grad()
    def boundary_states(self, prompt: Dict[str, torch.Tensor],
                        n_steps: Optional[int] = None) -> torch.Tensor:
        """Hidden state at the entry boundary (B, d) -- the probe attachment point."""
        m = self.model
        x_prompt = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                           prompt["value"], prompt["value_mask"])
        _, _, _, boundary_h = self.latent_block(x_prompt, prompt["pad_mask"], n_steps)
        return boundary_h

    @torch.no_grad()
    def solve(self, problems: List[str], tok: ArithmeticTokenizer, max_answer_len: int = 20,
              n_steps: Optional[int] = None, device: str = "cpu") -> List[Optional[str]]:
        """Greedy-decode answers. The latent positions are **never decoded**."""
        self.eval()
        prompt, *_ = coconut_collate([(p, "0") for p in problems], tok, device)
        m = self.model
        x_prompt = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                           prompt["value"], prompt["value_mask"])
        x, pad, _, _ = self.latent_block(x_prompt, prompt["pad_mask"], n_steps)
        return _decode_answer(m, x, pad, tok, max_answer_len, device)


class LotusTrainer:
    def __init__(self, cfg: LotusConfig, tokenizer: ArithmeticTokenizer,
                 model_cfg: Optional[ModelConfig] = None):
        self.cfg = cfg
        self.tok = tokenizer
        self.device = resolve_device(cfg.device)
        self.amp = Amp(self.device, cfg.amp)
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        mcfg = model_cfg or ModelConfig(d_model=96, n_heads=4, d_ff=192,
                                        n_prelude=1, n_recurrent=1, n_coda=1,
                                        recurrent_steps=4)
        model = build_model(mcfg, tokenizer).to(self.device)
        bot = tokenizer.BOT if cfg.use_boundaries else None
        eot = tokenizer.EOT if (cfg.use_boundaries and cfg.use_exit_boundary) else None
        self.reasoner = LotusReasoner(
            model, cfg.n_latent, cfg.loops, bot, eot, cfg.trace_compress,
            cfg.space_dim if cfg.space_coef > 0 else 0,
            tuple(cfg.alu_moduli) if cfg.alu_coef > 0 else None,
            cfg.alu_mode).to(self.device)
        # ALU slot layout: slots 0..T-1 hold the T = 2^depth - 2 intermediates in
        # the grammar's post-order; the root goes in the **last** slot. One slot per
        # *value*, so the budget tracks reasoning steps rather than digits --
        # depth-3 wants 7 slots here against 14 for the token trace.
        #
        # The root is pinned to the last index rather than to T because T moves with
        # depth (2, 6, 14, ...). Indexing it by T would put the answer in a different
        # place at every depth, so a model trained at one depth would be reading an
        # untrained slot at another -- which silently breaks the depth-transfer
        # experiment this layout exists to make possible.
        self.n_inter = 2 ** cfg.depth - 2
        self.root_slot = cfg.n_latent - 1
        if cfg.alu_coef > 0 and cfg.trace_coef > 0:
            raise ValueError(
                "alu_coef and trace_coef both target the latent slots but mean "
                "different things by them -- one value per slot versus one digit "
                "token per slot. Running both puts contradictory targets on the "
                "same hidden state. Set trace_coef=0 for the ALU arm.")
        if cfg.alu_coef > 0 and cfg.n_latent < self.n_inter + 1:  # +1 for the root
            raise ValueError(
                f"n_latent={cfg.n_latent} cannot hold {self.n_inter} intermediates "
                f"plus a root slot at depth {cfg.depth}")
        self.opt = torch.optim.AdamW(self.reasoner.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.grammar = TaskGrammar()
        self.verifier = Verifier()
        self.descriptor = Descriptor(depth=cfg.depth, digits=cfg.digits, ops_key=cfg.ops_key)
        self.max_ans = cfg.max_answer_len()
        self._seed = cfg.seed * 1_000_003 + 1
        self.truncated = 0   # traces that did not fit in n_latent (diagnostic)

    # -- data -------------------------------------------------------------
    def _widths(self) -> List[int]:
        c = self.cfg
        if c.digits_min and c.digits_max:
            return list(range(c.digits_min, c.digits_max + 1))
        return [c.digits]

    def _sample_batch(self, n: int) -> List[Task]:
        out: List[Task] = []
        widths = self._widths()
        for _ in range(n):
            self._seed += 1
            d = Descriptor(self.cfg.depth, widths[self._seed % len(widths)],
                           self.cfg.ops_key)
            out.append(self.grammar.sample_with_trace(d, self._seed,
                                                      exclude_heldout=True))
        return out

    def _eval_set(self, n: int, digits: Optional[int] = None) -> List[Task]:
        """Held out by *problem*, not by seed: only the evaluation partition.

        Seed separation is not enough -- the task space is small enough that a long
        run trains on most of it, so a seed-separated eval set is over half
        memorised. :mod:`lamb.holdout` partitions by a hash of the problem itself,
        which training rejects, so overlap is zero however long training runs.
        """
        rng = random.Random(self.cfg.seed * 7 + 12345 + 101 * (digits or 0))
        desc = (self.descriptor if digits is None
                else Descriptor(self.cfg.depth, digits, self.cfg.ops_key))
        out: List[Task] = []
        seen = set()
        for _ in range(400 * max(1, n)):          # bounded: small spaces may exhaust
            if len(out) >= n:
                break
            t = self.grammar.sample_with_trace(desc, rng.randint(0, 2 ** 31 - 1))
            if is_heldout(t[0]) and t[0] not in seen:
                seen.add(t[0])
                out.append(t)
        while len(out) < n and out:               # tiny space: allow repeats
            out.append(out[len(out) % len(seen)])
        return out

    def _collate(self, tasks: List[Task]):
        prompt, aids, aab, apad, targets, tmask = coconut_collate(
            [(e, a) for e, a, _ in tasks], self.tok, self.device)
        # Capacity is n_latent * trace_compress: each latent position owns that
        # many consecutive trace tokens, so the block holds a longer trace without
        # growing.
        L = self.cfg.n_latent * self.reasoner.trace_compress
        b = len(tasks)
        tr_ids = torch.full((b, L), self.tok.PAD, dtype=torch.long, device=self.device)
        tr_mask = torch.zeros((b, L), dtype=torch.float32, device=self.device)
        for i, (_, _, trace) in enumerate(tasks):
            toks = trace_targets(self.tok, trace)
            if len(toks) > L:
                self.truncated += 1
                toks = toks[:L]
            if toks:
                tr_ids[i, :len(toks)] = torch.tensor(toks, device=self.device)
                tr_mask[i, :len(toks)] = 1.0
        return prompt, aids, aab, apad, targets, tmask, tr_ids, tr_mask

    def _alu_batch(self, tasks: List[Task]):
        """Residue targets per latent slot, plus each row's parsed tree.

        Slot ``i`` carries the ``i``-th post-order sub-result and the last slot the
        answer, so a slot holds one *value* whatever its digit count. Rows whose
        answer or trace leaves the representable ring are masked rather than
        clipped -- a wrapped value is not a small error in a residue system, it is a
        different number, and training on one teaches arithmetic that is wrong.
        """
        alu = self.reasoner.alu
        scalar = isinstance(alu, ScalarALU)
        K = 1 if scalar else len(alu.sys.moduli)
        L, b = self.cfg.n_latent, len(tasks)
        tgt = torch.zeros((b, L, K), dtype=torch.long, device=self.device)
        raw = torch.zeros((b, L), dtype=torch.float32, device=self.device)
        mask = torch.zeros((b, L), dtype=torch.float32, device=self.device)
        trees, keep = [], []
        for i, (expr, ans, trace) in enumerate(tasks):
            vals = list(trace) + [int(ans)]
            # A scalar has no ring to leave, so only the residue arm can go
            # out of range -- and there a wrapped value is a *different number*,
            # not a big one, so those rows are masked rather than clipped.
            if not scalar and not all(alu.sys.representable(v) for v in vals):
                trees.append(parse_expr(expr))
                keep.append(False)
                continue
            slots = list(range(len(trace))) + [self.root_slot]
            t = None if scalar else alu.sys.targets(vals, device=self.device)
            for j, sl in enumerate(slots):
                if t is not None:
                    tgt[i, sl] = t[j]
                raw[i, sl] = float(vals[j])
                mask[i, sl] = 1.0
            trees.append(parse_expr(expr))
            keep.append(True)
        return tgt, mask, trees, torch.tensor(keep, device=self.device), raw

    @torch.no_grad()
    def algebraic_accuracy(self, n_tasks: Optional[int] = None,
                           digits: Optional[int] = None) -> Dict[str, float]:
        """Two answers from one model: the network's readout and the algebra's.

        The readout is the ordinary decoded answer. The algebraic one is composed
        from the **leaf** slots by exact arithmetic, so it never runs a decoder at
        all -- one forward pass, no sampling, no EOS. Reporting both on the same
        weights is the only honest way to ask whether the algebra is doing work.
        ``residue_acc`` is the diagnostic that actually matters: CRT has no
        locality, so one wrong residue is a wildly wrong answer, and end-to-end
        accuracy is roughly the per-residue accuracy raised to the number of
        residues the problem needs.
        """
        self.reasoner.eval()
        tasks = self._eval_set(n_tasks or self.cfg.eval_tasks, digits)
        alu = self.reasoner.alu
        prompt, aids, aab, apad, *_ = self._collate(tasks)
        tgt, mask, trees, keep, raw = self._alu_batch(tasks)
        m = self.reasoner.model
        x_prompt = m.embed(prompt["input_ids"], prompt["abacus_ids"],
                           prompt["value"], prompt["value_mask"])
        _, _, latent_h, _ = self.reasoner.latent_block(x_prompt, prompt["pad_mask"])
        readout = self.reasoner.solve([e for e, _, _ in tasks], self.tok, self.max_ans,
                                      device=self.device)
        if isinstance(alu, ScalarALU):
            # The control: compose the regressed values by ordinary arithmetic and
            # round. Nothing quantises on the way, so an error anywhere in the tree
            # arrives at the answer intact.
            vals = alu.values(latent_h)
            got = [int(round(float(v))) for v in alu.compose_tree(vals, trees)]
            n_ok = sum(1 for (_, a, _), g in zip(tasks, got) if g == int(a))
            err = ((vals - raw).abs() * mask).sum() / mask.sum().clamp_min(1.0)
            r_ok = sum(self.verifier.check(e, r) for (e, _, _), r in zip(tasks, readout))
            return {"algebraic": n_ok / len(tasks), "readout": r_ok / len(tasks),
                    "leaf_residue": float("nan"), "leaf_residue_min": float("nan"),
                    "root_residue": float("nan"), "leaf_by_modulus": {},
                    "mean_abs_value_error": float(err), "in_range": 1.0}
        codes = alu.codes(latent_h)
        composed = alu.compose_tree(codes, trees)
        got = [alu.sys.crt([int(bl.argmax(-1)) for bl in blocks]) for blocks in composed]
        n_ok = sum(1 for (_, a, _), g, k in zip(tasks, got, keep.tolist())
                   if k and g == int(a))
        r_ok = sum(self.verifier.check(e, r) for (e, _, _), r in zip(tasks, readout))
        # Per-modulus accuracy, split by slot role. Averaging the two hides the
        # result: the algebra composes from *leaves*, so leaf accuracy is what
        # predicts the answer (roughly leaf_acc ** (n_moduli * n_leaves)), while the
        # root slot is only ever needed for the consistency check. The model has no
        # reason to learn the root -- composition is not its job any more -- so the
        # two numbers come apart hard, and the average is a number about nothing.
        blocks = alu.sys.split(codes)
        leaf_mask = mask.clone()
        leaf_mask[:, self.root_slot] = 0.0
        root_mask = torch.zeros_like(mask)
        root_mask[:, self.root_slot] = mask[:, self.root_slot]
        leaf, root = [], []
        for k_i, blk in enumerate(blocks):
            hit = (blk.argmax(-1) == tgt[..., k_i]).float()
            leaf.append(float((hit * leaf_mask).sum() / leaf_mask.sum().clamp_min(1.0)))
            root.append(float((hit * root_mask).sum() / root_mask.sum().clamp_min(1.0)))
        # Per-modulus, not just the mean. At an untrained operand width there are
        # two quite different failure modes and only this separates them: if the
        # leaf residues hold up and the answer still fails, the problem is range or
        # composition; if they collapse, the encoder cannot read the wider number at
        # all, or that modulus needed digit positions the training widths never
        # showed it. The per-modulus profile names which.
        by_mod = {p_: round(v, 4) for p_, v in zip(alu.sys.moduli, leaf)}
        return {"algebraic": n_ok / len(tasks), "readout": r_ok / len(tasks),
                "leaf_residue": sum(leaf) / len(leaf), "leaf_residue_min": min(leaf),
                "root_residue": sum(root) / len(root),
                "leaf_by_modulus": by_mod,
                "in_range": float(keep.float().mean())}

    # -- space supervision ------------------------------------------------
    def _space_loss(self, latent_h: torch.Tensor, tr_ids: torch.Tensor,
                    tr_mask: torch.Tensor) -> torch.Tensor:
        """Supervised-contrastive loss over the latent manifold.

        The second dimension of arXiv:2606.20075's decomposition. The trace loss
        pins what each latent *decodes to*; nothing pins how the latents are
        *arranged*, so they are free to drift and collapse in every direction the
        readout ignores. This term says: two latent positions standing for the same
        intermediate value belong together, and positions standing for different
        values belong apart.

        It is relational, not absolute -- deliberately. Pinning each latent to a
        fixed embedding of its value would be exactly the rigid geometric
        constraint that analysis finds collapses the reasoning space; a
        supervised-contrastive objective (arXiv:2004.11362) constrains only the
        relations, leaving the model free to choose coordinates.

        A position's identity is the tuple of trace tokens it owns, so this works
        unchanged under ``trace_compress > 1``. Positions with no positive pair in
        the batch contribute nothing.
        """
        head = self.reasoner.space_head
        if head is None:
            return torch.zeros((), device=latent_h.device)
        c = self.reasoner.trace_compress
        z = torch.nn.functional.normalize(head(latent_h), dim=-1)   # (B, L, k)
        b, L, _ = z.shape
        tau = self.cfg.space_tau
        total = torch.zeros((), device=latent_h.device)
        counted = 0
        eye = torch.eye(b, dtype=torch.bool, device=latent_h.device)
        for i in range(L):
            sl = slice(i * c, (i + 1) * c)
            valid = tr_mask[:, sl].min(dim=1).values > 0.5      # all c slots supervised
            if int(valid.sum()) < 2:
                continue
            keys = tr_ids[:, sl]
            same = (keys.unsqueeze(1) == keys.unsqueeze(0)).all(dim=-1)   # (B, B)
            pos = same & valid.unsqueeze(0) & valid.unsqueeze(1) & ~eye
            if not bool(pos.any()):
                continue
            sim = (z[:, i] @ z[:, i].t()) / tau
            # Contrast only against positions that actually carry a supervised
            # value. Rows whose trace ran out hold no intermediate, so using them
            # as negatives would push apart latents for no stated reason.
            sim = sim.masked_fill(eye | ~valid.unsqueeze(0), float("-inf"))
            logp = torch.log_softmax(sim, dim=-1)
            npos = pos.sum(dim=1)
            rows = (npos > 0) & valid
            if not bool(rows.any()):
                continue
            # select, do not multiply: the self-mask leaves -inf on the diagonal
            # and -inf * 0 is NaN.
            picked = torch.where(pos, logp, torch.zeros_like(logp))
            per_row = picked.sum(dim=1)[rows] / npos[rows].clamp_min(1)
            total = total - per_row.mean()
            counted += 1
        return total / max(1, counted)

    @torch.no_grad()
    def collapse_metric(self, n_tasks: int = 128) -> float:
        """Mean pairwise cosine between latent positions -- the collapse detector.

        1.0 means every latent position holds the same vector (the degenerate
        solution); low values mean the positions specialise. This is what the space
        term is supposed to move, so it is measured rather than assumed.
        """
        self.reasoner.eval()
        tasks = self._eval_set(n_tasks)
        prompt, aids, aab, apad, *_ = self._collate(tasks)
        _, _, _, aux = self.reasoner(prompt, aids, aab, apad)
        h = torch.nn.functional.normalize(aux["latent_h"], dim=-1)   # (B, L, d)
        g = h @ h.transpose(1, 2)                                    # (B, L, L)
        L = g.size(1)
        if L < 2:
            return float("nan")
        off = ~torch.eye(L, dtype=torch.bool, device=g.device)
        return float(g[:, off].mean())

    # -- training ---------------------------------------------------------
    def _lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup:
            return c.lr * (step + 1) / max(1, c.warmup)
        prog = (step - c.warmup) / max(1, c.steps - c.warmup)
        return 0.5 * c.lr * (1.0 + torch.cos(torch.tensor(prog * 3.141592653589793)).item())

    def _train_step(self, step: int) -> Dict[str, float]:
        c = self.cfg
        self.reasoner.train()
        tasks = self._sample_batch(c.batch_size)
        prompt, aids, aab, apad, targets, tmask, tr_ids, tr_mask = self._collate(tasks)
        for g in self.opt.param_groups:
            g["lr"] = self._lr_at(step)
        with self.amp.autocast():
            ans_logits, latent_logits, switch_logits, aux = self.reasoner(prompt, aids, aab, apad)
            ans_loss = _masked_ce(ans_logits, targets, tmask)
            # Per-position supervision on the latent block. With trace_coef=0 this
            # reduces to the answer-only ablation (parallel latents, no trace).
            tr_loss = (_masked_ce(latent_logits, tr_ids, tr_mask)
                       if c.trace_coef > 0 and float(tr_mask.sum()) > 0
                       else torch.zeros((), device=ans_logits.device))
            # Entry boundary: make "start reasoning latently" a predicted token, so
            # the latent segment has a probability an RL objective can act on.
            if switch_logits is not None and c.switch_coef > 0:
                bot = torch.full((switch_logits.size(0),), self.tok.BOT,
                                 dtype=torch.long, device=switch_logits.device)
                sw_loss = torch.nn.functional.cross_entropy(switch_logits, bot)
            else:
                sw_loss = torch.zeros((), device=ans_logits.device)
            # Space supervision: arrange the latent manifold, do not just decode it.
            sp_loss = (self._space_loss(aux["latent_h"], tr_ids, tr_mask)
                       if c.space_coef > 0 else torch.zeros((), device=ans_logits.device))
            # The latent ALU: one value per slot, coded in residues. The answer is
            # composed from the leaves by exact arithmetic instead of predicted, so
            # nothing about composition has to generalise -- it is not learned.
            if c.alu_coef > 0:
                tgt, amask, trees, _, raw = self._alu_batch(tasks)
                if isinstance(self.reasoner.alu, ScalarALU):
                    vals = self.reasoner.alu.values(aux["latent_h"])
                    alu_loss = self.reasoner.alu.value_loss(vals, raw, amask)
                    codes = None
                else:
                    codes = self.reasoner.alu.codes(aux["latent_h"])
                    alu_loss = self.reasoner.alu.code_loss(codes, tgt, amask)
                # The label-free agreement term is available but OFF by default.
                # Both the stated answer and the composed one are functions of the
                # same latent block, so training on their agreement invites the
                # model to satisfy it by routing rather than by being right. It is
                # worth more as a *measurement* than as a loss until measured.
                if c.alu_consistency_coef > 0 and codes is not None:
                    alu_loss = alu_loss + c.alu_consistency_coef * self.reasoner.alu.consistency(
                        codes, trees, self.root_slot).mean()
            else:
                alu_loss = torch.zeros((), device=ans_logits.device)
            loss = (ans_loss + c.trace_coef * tr_loss + c.switch_coef * sw_loss
                    + c.space_coef * sp_loss + c.alu_coef * alu_loss)
        self.amp.backward_step(loss, self.opt, self.reasoner.parameters(), c.grad_clip)
        return {"loss": float(loss.detach()), "ans": float(ans_loss.detach()),
                "trace": float(tr_loss.detach()), "switch": float(sw_loss.detach()),
                "space": float(sp_loss.detach()), "alu": float(alu_loss.detach())}

    # -- evaluation -------------------------------------------------------
    @torch.no_grad()
    def accuracy(self, n_tasks: Optional[int] = None,
                 digits: Optional[int] = None) -> float:
        """Exact-match accuracy, decoding only the answer (latents stay latent)."""
        n = n_tasks or self.cfg.eval_tasks
        tasks = self._eval_set(n, digits)
        ans = self.reasoner.solve([e for e, _, _ in tasks], self.tok, self.max_ans,
                                  device=self.device)
        return sum(self.verifier.check(e, a) for (e, _, _), a in zip(tasks, ans)) / n

    @torch.no_grad()
    def trace_probe(self, n_tasks: int = 128) -> float:
        """Diagnostic only: how well the latent positions predict the gold trace.

        This is never used at inference -- it just reports whether the latent block
        actually carries the intermediate results it was supervised on.
        """
        self.reasoner.eval()
        tasks = self._eval_set(n_tasks)
        prompt, aids, aab, apad, _, _, tr_ids, tr_mask = self._collate(tasks)
        _, latent_logits, _, _ = self.reasoner(prompt, aids, aab, apad)
        correct = (latent_logits.argmax(dim=-1) == tr_ids).float() * tr_mask
        return float(correct.sum() / tr_mask.sum().clamp_min(1.0))

    @torch.no_grad()
    def _boundary_data(self, tasks: List[Task]):
        """Entry-boundary states plus whether the model actually answers correctly."""
        prompt, *_ = self._collate(tasks)
        states = self.reasoner.boundary_states(prompt)
        ans = self.reasoner.solve([e for e, _, _ in tasks], self.tok, self.max_ans,
                                  device=self.device)
        y = torch.tensor([1.0 if self.verifier.check(e, a) else 0.0
                          for (e, _, _), a in zip(tasks, ans)], device=self.device)
        return states, y

    def boundary_probe(self, n_tasks: int = 256, steps: int = 300) -> Tuple[float, float]:
        """Monitorability check: is the blackbox readable at the boundary?

        Fits a linear probe on the *entry-boundary* hidden state to predict whether
        the model will answer correctly, and reports ``(held-out accuracy,
        majority-class baseline)``. Nothing is decoded and no reasoning is
        verbalised -- but if the probe beats the baseline, an opaque latent model is
        still auditable at a single, fixed position. This is the practical answer to
        the chain-of-thought-monitorability objection (arXiv:2507.11473), using the
        attachment point the boundary tokens create.
        """
        tasks = self._eval_set(n_tasks)
        states, y = self._boundary_data(tasks)
        if states is None:
            return float("nan"), float("nan")
        # Balance the classes, so the baseline is 0.5 by construction and the number
        # means something. Without this the metric is degenerate once the model is
        # accurate: at 93% correct, "always say correct" scores 0.94 and there are
        # too few errors left to fit a probe against.
        pos = (y > 0.5).nonzero(as_tuple=True)[0]
        neg = (y <= 0.5).nonzero(as_tuple=True)[0]
        k = min(pos.numel(), neg.numel())
        if k < 16:
            return float("nan"), 0.5      # too few of one class to say anything
        sel = torch.cat([pos[:k], neg[:k]])
        sel = sel[torch.randperm(sel.numel(), device=sel.device)]
        states, y = states[sel], y[sel]
        n = states.size(0)
        cut = n // 2
        xtr, ytr, xte, yte = states[:cut], y[:cut], states[cut:], y[cut:]
        mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        probe = torch.nn.Linear(states.size(1), 1).to(self.device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    probe(xtr).squeeze(-1), ytr)
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = (probe(xte).squeeze(-1) > 0).float()
            acc = float((pred == yte).float().mean())
        return acc, 0.5   # balanced by construction

    def train(self) -> Dict[str, float]:
        c = self.cfg
        print(f"[lotus] {device_report(self.device, self.amp)} backend={self.verifier.backend}")
        print(f"[lotus] task=depth{c.depth}/digits{c.digits}/ops{c.ops_key} "
              f"latents={c.n_latent} loops={c.loops} trace_coef={c.trace_coef} "
              f"boundaries={c.use_boundaries} params={self.reasoner.model.num_params()}")
        run = {"loss": 0.0, "ans": 0.0, "trace": 0.0, "switch": 0.0}
        final: Dict[str, float] = {}
        for step in range(c.steps):
            m = self._train_step(step)
            for k in run:
                run[k] += m[k]
            if (step + 1) % c.log_every == 0:
                print(f"  step {step+1:5d}/{c.steps}  loss {run['loss']/c.log_every:.4f} "
                      f"(ans {run['ans']/c.log_every:.4f} trace {run['trace']/c.log_every:.4f} "
                      f"switch {run['switch']/c.log_every:.4f})  lr {self._lr_at(step):.2e}")
                run = {k: 0.0 for k in run}
            if (step + 1) % c.eval_every == 0 or step == c.steps - 1:
                acc, probe = self.accuracy(), self.trace_probe()
                final = {"acc": acc, "trace_probe": probe}
                print(f"  [eval @ {step+1}] answer acc {acc:.3f}   "
                      f"latent trace-probe {probe:.3f} (diagnostic; never decoded)")
        if c.use_boundaries:
            p_acc, p_base = self.boundary_probe()
            final.update({"boundary_probe": p_acc, "boundary_base": p_base})
            print(f"  [monitorability] boundary probe predicts correctness "
                  f"{p_acc:.3f} vs balanced baseline {p_base:.3f} "
                  f"({'inconclusive: too few errors to fit' if p_acc != p_acc else 'class-balanced'})")
        return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LAMb Stage A (restructured): LOTUS-style parallel supervised latent reasoning")
    p.add_argument("--steps", type=int, default=LotusConfig.steps)
    p.add_argument("--batch-size", type=int, default=LotusConfig.batch_size)
    p.add_argument("--depth", type=int, default=LotusConfig.depth)
    p.add_argument("--digits", type=int, default=LotusConfig.digits)
    p.add_argument("--ops-key", type=int, default=LotusConfig.ops_key)
    p.add_argument("--n-latent", type=int, default=LotusConfig.n_latent)
    p.add_argument("--loops", type=int, default=LotusConfig.loops)
    p.add_argument("--trace-compress", type=int, default=LotusConfig.trace_compress,
                   help="trace tokens supervised per latent position (multi-token "
                        "prediction on the latent block). 1 = one latent per trace "
                        "token; c > 1 gives capacity n_latent*c without more latents.")
    p.add_argument("--trace-coef", type=float, default=LotusConfig.trace_coef,
                   help="weight on per-latent-position supervision (0 = answer-only ablation)")
    p.add_argument("--boundaries", dest="use_boundaries", action="store_true",
                   default=LotusConfig.use_boundaries,
                   help="opt in to the SWITCH entry boundary before the latent segment. "
                        "Off by default: on a clean eval split it showed no measurable "
                        "effect (0.945 without vs 0.934 with) and its RL rationale was "
                        "falsified. Kept as the probe/policy attachment point.")
    p.add_argument("--exit-boundary", action="store_true",
                   help="also emit an exit marker (measured harmful: it blocks the "
                        "answer readout from the latent states)")
    p.add_argument("--switch-coef", type=float, default=LotusConfig.switch_coef,
                   help="weight on predicting the entry boundary (the RL hook)")
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--recurrent-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=LotusConfig.seed)
    add_hardware_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device, amp = resolve_hardware(args)
    cfg = LotusConfig(steps=args.steps, batch_size=args.batch_size, depth=args.depth,
                      digits=args.digits, ops_key=args.ops_key, n_latent=args.n_latent,
                      loops=args.loops, trace_coef=args.trace_coef, seed=args.seed,
                      trace_compress=args.trace_compress,
                      use_boundaries=args.use_boundaries, switch_coef=args.switch_coef,
                      use_exit_boundary=args.exit_boundary, device=device, amp=amp)
    tok = ArithmeticTokenizer()
    mcfg = ModelConfig(d_model=args.d_model, n_heads=4, d_ff=2 * args.d_model,
                       n_prelude=1, n_recurrent=1, n_coda=1, recurrent_steps=args.recurrent_steps)
    LotusTrainer(cfg, tok, mcfg).train()


if __name__ == "__main__":
    main()
