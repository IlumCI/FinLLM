"""Self-play self-improvement: proposer, verifier, and the training loop."""

from __future__ import annotations

from .loop import SelfPlayTrainer, StepStats
from .proposer import Proposer
from .verifier import Verifier

__all__ = ["SelfPlayTrainer", "StepStats", "Proposer", "Verifier"]
