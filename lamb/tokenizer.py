"""Number-native tokenizer for LAMb.

LAMb does not use BPE. Its vocabulary is digits, arithmetic operators, and a few
structural tokens, so numbers are represented *compositionally* by their digits.
Two ideas from the arithmetic-reasoning literature are baked in:

* **Reversed (LSB-first) digits** -- writing numbers least-significant-digit
  first aligns carry propagation with the autoregressive scan and markedly
  improves length generalisation (Abacus / "reverse the digits" results).
* **Abacus positions** -- each digit additionally carries its position *within
  its own number* (reset at every number boundary), enabling the positional
  scheme that generalises addition to unseen lengths.

A companion *value channel* (computed in :mod:`lamb.model.embeddings`) gives each
operand digit access to its number's magnitude. Crucially, value features are
attached **only to the problem operands**, never to the answer being predicted,
so magnitude information can never leak the label.

This module is deliberately torch-free; batching lives in :mod:`lamb.data`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Encoded:
    """One tokenised ``problem = answer`` example (all lists share length)."""

    ids: List[int]
    abacus: List[int]          # intra-number position (0 for non-digits), 1-indexed
    value: List[float]         # operand magnitude on problem digits, else 0.0
    value_mask: List[float]    # 1.0 where `value` is meaningful, else 0.0
    ans_start: int             # index of the first answer token in `ids`

    def __len__(self) -> int:
        return len(self.ids)


class ArithmeticTokenizer:
    """Maps arithmetic strings to/from digit-level token sequences."""

    def __init__(self, base: int = 10, reverse_digits: bool = True, max_number_len: int = 24):
        if not 2 <= base <= 10:
            raise ValueError("base must be in [2, 10] for single-char digits")
        self.base = base
        self.reverse_digits = reverse_digits
        self.max_number_len = max_number_len

        # Fixed special/operator ids, then digits.
        self.PAD, self.BOS, self.EOS, self.EQ = 0, 1, 2, 3
        self._op_ids: Dict[str, int] = {"+": 4, "-": 5, "*": 6, "(": 7, ")": 8}
        self._digit0 = 9  # digit d -> _digit0 + d
        self.vocab_size = self._digit0 + base

        self._id_to_char: Dict[int, str] = {
            self.EQ: "=",
            **{v: k for k, v in self._op_ids.items()},
            **{self._digit0 + d: str(d) for d in range(base)},
        }

    # -- id helpers -------------------------------------------------------
    def is_digit_id(self, tid: int) -> bool:
        return self._digit0 <= tid < self._digit0 + self.base

    def digit_value(self, tid: int) -> int:
        return tid - self._digit0

    def _digit_id(self, ch: str) -> int:
        return self._digit0 + int(ch)

    # -- encoding ---------------------------------------------------------
    def _emit_number(
        self, enc: Encoded, num_str: str, on_problem_side: bool
    ) -> None:
        digits = list(num_str)
        if self.reverse_digits:
            digits = digits[::-1]  # least-significant digit first
        magnitude = float(int(num_str))
        for pos, ch in enumerate(digits, start=1):
            enc.ids.append(self._digit_id(ch))
            enc.abacus.append(min(pos, self.max_number_len))
            if on_problem_side:
                enc.value.append(magnitude)
                enc.value_mask.append(1.0)
            else:
                enc.value.append(0.0)
                enc.value_mask.append(0.0)

    def _emit_symbol(self, enc: Encoded, tid: int) -> None:
        enc.ids.append(tid)
        enc.abacus.append(0)
        enc.value.append(0.0)
        enc.value_mask.append(0.0)

    def _scan_expression(self, enc: Encoded, text: str, on_problem_side: bool) -> None:
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch.isdigit():
                j = i
                while j < n and text[j].isdigit():
                    j += 1
                self._emit_number(enc, text[i:j], on_problem_side)
                i = j
            elif ch in self._op_ids:
                self._emit_symbol(enc, self._op_ids[ch])
                i += 1
            elif ch.isspace():
                i += 1
            else:
                raise ValueError(f"unexpected character {ch!r} in {text!r}")

    def encode(self, problem: str, answer: str) -> Encoded:
        """Tokenise ``[BOS] problem = answer [EOS]`` with masks and Abacus ids."""
        enc = Encoded(ids=[], abacus=[], value=[], value_mask=[], ans_start=0)
        self._emit_symbol(enc, self.BOS)
        self._scan_expression(enc, problem, on_problem_side=True)
        self._emit_symbol(enc, self.EQ)
        enc.ans_start = len(enc.ids)
        self._scan_expression(enc, answer, on_problem_side=False)
        self._emit_symbol(enc, self.EOS)
        return enc

    def encode_prompt(self, problem: str) -> Encoded:
        """Tokenise just ``[BOS] problem =`` -- the context for generation."""
        enc = Encoded(ids=[], abacus=[], value=[], value_mask=[], ans_start=0)
        self._emit_symbol(enc, self.BOS)
        self._scan_expression(enc, problem, on_problem_side=True)
        self._emit_symbol(enc, self.EQ)
        enc.ans_start = len(enc.ids)
        return enc

    def build_from_answer_ids(self, problem: str, answer_ids: List[int]) -> Encoded:
        """Build a full teacher-forcing example from *sampled* answer token ids.

        Used by GRPO: the answer comes from the policy's own rollout (a token id
        list, possibly ending in EOS) rather than a ground-truth string, so we
        splice those exact tokens onto the problem prompt and compute their
        Abacus positions incrementally. An EOS is appended if the rollout did not
        already end with one, so the example always has a supervised final token.
        """
        enc = self.encode_prompt(problem)
        prev_id = enc.ids[-1] if enc.ids else None
        prev_ab = enc.abacus[-1] if enc.abacus else 0
        ended = False
        for tid in answer_ids:
            if tid in (self.PAD,):
                continue
            ab = self.abacus_after(prev_id, prev_ab, tid)
            enc.ids.append(tid)
            enc.abacus.append(ab)
            enc.value.append(0.0)
            enc.value_mask.append(0.0)
            prev_id, prev_ab = tid, ab
            if tid == self.EOS:
                ended = True
                break
        if not ended:
            self._emit_symbol(enc, self.EOS)
        return enc

    # -- decoding ---------------------------------------------------------
    def decode_answer(self, ids: List[int]) -> Optional[str]:
        """Turn a generated answer token span into a canonical integer string.

        Accepts the tokens strictly between ``=`` and ``EOS`` (a leading ``-`` is
        honoured as a sign). Returns ``None`` if no digits are present.
        """
        sign = 1
        digits: List[str] = []
        for tid in ids:
            if tid in (self.EOS, self.PAD):
                break
            if tid == self._op_ids["-"] and not digits:
                sign = -1
            elif self.is_digit_id(tid):
                digits.append(str(self.digit_value(tid)))
        if not digits:
            return None
        if self.reverse_digits:
            digits = digits[::-1]
        return str(sign * int("".join(digits)))

    def abacus_after(self, prev_id: Optional[int], prev_abacus: int, new_id: int) -> int:
        """Incremental Abacus index for a token emitted during generation."""
        if not self.is_digit_id(new_id):
            return 0
        if prev_id is not None and self.is_digit_id(prev_id):
            return min(prev_abacus + 1, self.max_number_len)
        return 1
