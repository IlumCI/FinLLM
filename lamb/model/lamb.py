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
        h, aux = self.core(x, pad_mask, n_steps)
        h = self.norm_f(h)
        # Optional per-environment adapter (shared-backbone POET): a tiny module
        # that specialises the shared backbone's hidden state before the LM head.
        if hidden_adapter is not None:
            h = hidden_adapter(h)
        logits = self.lm_head(h)
        return logits, aux

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


def build_model(cfg: ModelConfig, tokenizer: ArithmeticTokenizer) -> LAMb:
    """Construct a LAMb with vocab size taken from the tokenizer."""
    cfg.vocab_size = tokenizer.vocab_size
    return LAMb(cfg)
