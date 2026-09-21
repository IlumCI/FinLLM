"""The LAMb model: number-native embedding + latent core + tied LM head.

Provides the training loss (answer-only cross-entropy over next tokens) and a
batched greedy ``solve`` that decodes answers for a list of problem strings --
the interface the self-play loop and the evaluator use.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..tokenizer import ArithmeticTokenizer
from .embeddings import NumberAwareEmbedding
from .latent_core import RecurrentDepthCore
from .transformer import RMSNorm


class LAMb(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = NumberAwareEmbedding(cfg)
        self.core = RecurrentDepthCore(cfg)
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.token_emb.weight

        # Coconut continuous-thought interface (Stage A). A "thought" is a latent
        # scratchpad position inserted between the prompt and the answer: its
        # input embedding is the core's own last hidden state, fed straight back
        # in continuous space -- never decoded to a token. ``thought_norm`` maps a
        # residual-stream hidden state into input-embedding statistics (the core
        # expects a normalised input, and this keeps the feedback loop stable);
        # ``thought_marker`` is a learned additive tag so the model can tell it is
        # consuming its own thought rather than a real token (a latent analogue of
        # Coconut's <bot>/<eot> boundaries). Zero-init marker => at start a thought
        # is exactly its normed hidden state, so the interface is inert until the
        # model learns to use it.
        self.thought_norm = RMSNorm(cfg.d_model)
        self.thought_marker = nn.Parameter(torch.zeros(cfg.d_model))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:  # tied weights counted once
                seen.add(id(p))
                total += p.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        abacus_ids: torch.Tensor,
        value: torch.Tensor,
        value_mask: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        hidden_adapter=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.embed(input_ids, abacus_ids, value, value_mask)
        h_core, aux = self.core(x, pad_mask, n_steps)
        logits = self._readout(h_core, hidden_adapter)
        return logits, aux

    def _readout(self, h_core: torch.Tensor, hidden_adapter=None) -> torch.Tensor:
        """Map a residual-stream core output to vocabulary logits.

        ``norm_f`` + ``lm_head`` is the *verbalisation* head. The Coconut latent
        thought deliberately bypasses it -- a thought is the pre-readout residual
        stream, never a decoded token -- so this stays a thin, shared readout.
        """
        h = self.norm_f(h_core)
        # Optional per-environment adapter (shared-backbone POET): a tiny module
        # that specialises the shared backbone's hidden state before the LM head.
        if hidden_adapter is not None:
            h = hidden_adapter(h)
        return self.lm_head(h)

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], n_steps: Optional[int] = None, hidden_adapter=None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits, aux = self.forward(
            batch["input_ids"],
            batch["abacus_ids"],
            batch["value"],
            batch["value_mask"],
            batch.get("pad_mask"),
            n_steps,
            hidden_adapter=hidden_adapter,
        )
        b, t, vsz = logits.shape
        ce = F.cross_entropy(
            logits.reshape(b * t, vsz), batch["target_ids"].reshape(b * t), reduction="none"
        ).view(b, t)
        mask = batch["loss_mask"]
        denom = mask.sum().clamp_min(1.0)
        loss = (ce * mask).sum() / denom

        metrics: Dict[str, float] = {}
        if "ponder" in aux:
            loss = loss + self.cfg.ponder_cost * aux["ponder"]
            metrics["ponder"] = float(aux["ponder"].detach())
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            metrics["token_acc"] = float(((pred == batch["target_ids"]).float() * mask).sum() / denom)
        return loss, metrics

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def _generate_ids(
        self,
        problems: List[str],
        tokenizer: ArithmeticTokenizer,
        max_answer_len: int = 20,
        n_steps: Optional[int] = None,
        device: str = "cpu",
        greedy: bool = True,
        temperature: float = 1.0,
        hidden_adapter=None,
    ) -> List[List[int]]:
        """Autoregressively decode answer token ids for each problem (batched).

        Greedy when ``greedy`` else multinomial at ``temperature``. Returns, per
        problem, the generated answer token ids up to and including EOS (trailing
        padding stripped) -- the raw material for both ``solve`` and GRPO.
        """
        self.eval()
        prompts = [tokenizer.encode_prompt(p) for p in problems]
        width = max(len(p) for p in prompts)
        b = len(prompts)

        ids = torch.full((b, width), tokenizer.PAD, dtype=torch.long, device=device)
        abacus = torch.zeros((b, width), dtype=torch.long, device=device)
        value = torch.zeros((b, width), dtype=torch.float32, device=device)
        vmask = torch.zeros((b, width), dtype=torch.float32, device=device)
        pad = torch.ones((b, width), dtype=torch.bool, device=device)  # True where padding
        for i, e in enumerate(prompts):
            off = width - len(e)  # left pad
            ids[i, off:] = torch.tensor(e.ids, device=device)
            abacus[i, off:] = torch.tensor(e.abacus, device=device)
            value[i, off:] = torch.tensor(e.value, device=device)
            vmask[i, off:] = torch.tensor(e.value_mask, device=device)
            pad[i, off:] = False

        answer_start = width
        done = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_answer_len):
            logits, _ = self.forward(ids, abacus, value, vmask, pad, n_steps, hidden_adapter=hidden_adapter)
            step_logits = logits[:, -1, :]
            if greedy:
                nxt = step_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(step_logits / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, tokenizer.PAD), nxt)

            prev_id = ids[:, -1]
            prev_ab = abacus[:, -1]
            new_ab = torch.zeros_like(nxt)
            for i in range(b):
                new_ab[i] = tokenizer.abacus_after(int(prev_id[i]), int(prev_ab[i]), int(nxt[i]))

            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            abacus = torch.cat([abacus, new_ab.unsqueeze(1)], dim=1)
            value = torch.cat([value, torch.zeros((b, 1), dtype=torch.float32, device=device)], dim=1)
            vmask = torch.cat([vmask, torch.zeros((b, 1), dtype=torch.float32, device=device)], dim=1)
            pad = torch.cat([pad, done.unsqueeze(1)], dim=1)

            done = done | (nxt == tokenizer.EOS)
            if bool(done.all()):
                break

        out: List[List[int]] = []
        for row in ids[:, answer_start:].tolist():
            answer: List[int] = []
            for tid in row:
                if tid == tokenizer.PAD:
                    break
                answer.append(tid)
                if tid == tokenizer.EOS:
                    break
            out.append(answer)
        return out

    @torch.no_grad()
    def solve(
        self,
        problems: List[str],
        tokenizer: ArithmeticTokenizer,
        max_answer_len: int = 20,
        n_steps: Optional[int] = None,
        device: str = "cpu",
        hidden_adapter=None,
    ) -> List[Optional[str]]:
        """Greedy-decode and return the answer string for each problem."""
        gen = self._generate_ids(
            problems, tokenizer, max_answer_len, n_steps, device, greedy=True,
            hidden_adapter=hidden_adapter,
        )
        return [tokenizer.decode_answer(ids) for ids in gen]

    @torch.no_grad()
    def sample(
        self,
        problems: List[str],
        tokenizer: ArithmeticTokenizer,
        max_answer_len: int = 20,
        n_steps: Optional[int] = None,
        device: str = "cpu",
        temperature: float = 1.0,
    ) -> List[List[int]]:
        """Sample answer token ids (for GRPO rollouts)."""
        return self._generate_ids(
            problems, tokenizer, max_answer_len, n_steps, device, greedy=False,
            temperature=temperature,
        )

    # -- Coconut continuous-thought path (Stage A) ------------------------
    def _roll_thoughts(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        n_thoughts: int,
        n_steps: Optional[int] = None,
        thought_dropout: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Append ``n_thoughts`` latent scratchpad positions to input embeddings.

        Each thought's input embedding is the core's own last hidden state, mapped
        back into embedding space by ``thought_norm`` (bounded magnitude -- the
        stability trick the Coconut paper relies on) plus the learned latent
        marker; it is never decoded to a token. ``thought_dropout`` > 0 perturbs
        the fed-back state so independently rolled trajectories differ (the
        diversity source for verifier-selected best-of-N; deterministic when 0).
        Costs one core forward per thought (the paper's ``n+1`` sequential passes);
        gradients flow back through the whole chain when called under grad.
        """
        for _ in range(max(0, int(n_thoughts))):
            h_core, _ = self.core(x, pad_mask, n_steps)
            thought = self.thought_norm(h_core[:, -1]) + self.thought_marker  # (B, d)
            if thought_dropout > 0.0:
                thought = F.dropout(thought, p=thought_dropout, training=True)
            x = torch.cat([x, thought.unsqueeze(1)], dim=1)
            if pad_mask is not None:
                keep = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
                pad_mask = torch.cat([pad_mask, keep], dim=1)
        return x, pad_mask

    def coconut_logits(
        self,
        prompt: Dict[str, torch.Tensor],
        answer_ids: torch.Tensor,
        answer_abacus: torch.Tensor,
        answer_pad: torch.Tensor,
        n_thoughts: int,
        n_steps: Optional[int] = None,
        thought_dropout: float = 0.0,
        hidden_adapter=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Teacher-forced answer logits with ``n_thoughts`` latent steps inserted.

        ``prompt`` holds left-padded ``[BOS] expr =`` tensors (keys ``input_ids``,
        ``abacus_ids``, ``value``, ``value_mask``, ``pad_mask``); ``answer_ids`` are
        the right-padded answer *content* tokens (no EOS). Returns logits over the
        ``A + 1`` positions that predict ``[a_0 .. a_{A-1}, EOS]`` -- i.e. the last
        thought position followed by each answer-input position -- so the caller
        scores them against ``[answer_ids, EOS]``. Differentiable end to end
        (BPTT through the thought chain).
        """
        x = self.embed(prompt["input_ids"], prompt["abacus_ids"], prompt["value"], prompt["value_mask"])
        pad = prompt.get("pad_mask")
        x, pad = self._roll_thoughts(x, pad, n_thoughts, n_steps, thought_dropout)

        zeros = torch.zeros_like(answer_abacus, dtype=torch.float32)
        a = self.embed(answer_ids, answer_abacus, zeros, zeros)
        x = torch.cat([x, a], dim=1)
        if pad is not None:
            pad = torch.cat([pad, answer_pad], dim=1)

        h_core, aux = self.core(x, pad, n_steps)
        logits = self._readout(h_core, hidden_adapter)
        a_len = answer_ids.size(1)
        return logits[:, -(a_len + 1):, :], aux

    @torch.no_grad()
    def coconut_solve(
        self,
        problems: List[str],
        tokenizer: ArithmeticTokenizer,
        n_thoughts: int,
        max_answer_len: int = 20,
        n_steps: Optional[int] = None,
        device: str = "cpu",
        thought_dropout: float = 0.0,
    ) -> List[Optional[str]]:
        """Greedy-decode answers after ``n_thoughts`` latent scratchpad steps.

        ``thought_dropout`` > 0 makes the latent phase stochastic (call repeatedly
        for diverse trajectories); the *answer* is always decoded greedily with
        dropout off, matching the 2510.12167 recipe (perturb the thoughts, not the
        emitted tokens).
        """
        self.eval()
        prompts = [tokenizer.encode_prompt(p) for p in problems]
        width = max(len(p) for p in prompts)
        b = len(prompts)

        ids = torch.full((b, width), tokenizer.PAD, dtype=torch.long, device=device)
        abacus = torch.zeros((b, width), dtype=torch.long, device=device)
        value = torch.zeros((b, width), dtype=torch.float32, device=device)
        vmask = torch.zeros((b, width), dtype=torch.float32, device=device)
        pad = torch.ones((b, width), dtype=torch.bool, device=device)
        last_id = torch.full((b,), tokenizer.EQ, dtype=torch.long, device=device)
        last_ab = torch.zeros((b,), dtype=torch.long, device=device)
        for i, e in enumerate(prompts):
            off = width - len(e)  # left pad
            ids[i, off:] = torch.tensor(e.ids, device=device)
            abacus[i, off:] = torch.tensor(e.abacus, device=device)
            value[i, off:] = torch.tensor(e.value, device=device)
            vmask[i, off:] = torch.tensor(e.value_mask, device=device)
            pad[i, off:] = False

        x = self.embed(ids, abacus, value, vmask)
        x, pad = self._roll_thoughts(x, pad, n_thoughts, n_steps, thought_dropout)

        done = torch.zeros(b, dtype=torch.bool, device=device)
        cols: List[torch.Tensor] = []
        for _ in range(max_answer_len):
            h_core, _ = self.core(x, pad, n_steps)
            nxt = self._readout(h_core)[:, -1, :].argmax(dim=-1)
            nxt = torch.where(done, torch.full_like(nxt, tokenizer.PAD), nxt)

            new_ab = torch.zeros_like(nxt)
            for i in range(b):
                new_ab[i] = tokenizer.abacus_after(int(last_id[i]), int(last_ab[i]), int(nxt[i]))
            z = torch.zeros((b, 1), dtype=torch.float32, device=device)
            tok_emb = self.embed(nxt[:, None], new_ab[:, None], z, z)
            x = torch.cat([x, tok_emb], dim=1)
            pad = torch.cat([pad, done.unsqueeze(1)], dim=1)

            cols.append(nxt)
            last_id, last_ab = nxt, new_ab
            done = done | (nxt == tokenizer.EOS)
            if bool(done.all()):
                break

        gen = torch.stack(cols, dim=1) if cols else torch.zeros((b, 0), dtype=torch.long)
        out: List[Optional[str]] = []
        for row in gen.tolist():
            answer: List[int] = []
            for tid in row:
                if tid == tokenizer.PAD:
                    break
                answer.append(tid)
                if tid == tokenizer.EOS:
                    break
            out.append(tokenizer.decode_answer(answer))
        return out


def build_model(cfg: ModelConfig, tokenizer: ArithmeticTokenizer) -> LAMb:
    """Construct a LAMb with vocab size taken from the tokenizer."""
    cfg.vocab_size = tokenizer.vocab_size
    return LAMb(cfg)
