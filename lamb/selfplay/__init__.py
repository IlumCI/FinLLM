"""Self-play self-improvement: proposers, verifier, GRPO, and the training loop."""

from __future__ import annotations

from .hyperproposer import GRPOHyperProposer
from .league import League
from .loop import SelfPlayTrainer, StepStats
from .proposer import BanditProposer, BaseProposer, Proposer, learnability
from .verifier import Verifier

__all__ = [
    "SelfPlayTrainer",
    "StepStats",
    "BaseProposer",
    "BanditProposer",
    "Proposer",
    "GRPOHyperProposer",
    "League",
    "learnability",
    "Verifier",
]
