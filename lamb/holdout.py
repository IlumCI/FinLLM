"""A train/eval partition of the *problem space*, not of the random seeds.

Disjoint seeds are not disjoint problems. The depth-2/1-digit grammar has only
~80k distinct expressions, so a 1000-step run at batch 64 draws 64k of them and
over half of a seed-separated "held-out" set had in fact been trained on -- and the
contamination grows with training length, which biases exactly the longer-vs-shorter
comparisons one wants to make.

So membership is decided by a hash of the problem string itself: stable across
runs, processes and seeds, and independent of how long training goes. A problem is
either always eval or always train, forever.
"""

from __future__ import annotations

import hashlib

BUCKETS = 20
EVAL_BUCKETS = 3          # 15% of the problem space is reserved for evaluation


def _bucket(problem: str) -> int:
    h = hashlib.blake2b(problem.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % BUCKETS


def is_heldout(problem: str) -> bool:
    """True iff this problem belongs to the evaluation partition."""
    return _bucket(problem) < EVAL_BUCKETS


def is_trainable(problem: str) -> bool:
    return not is_heldout(problem)
