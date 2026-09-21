"""Self-play self-improvement: proposers, verifier, GRPO, and the training loop."""

from __future__ import annotations

from .factoredhyper import FactoredHyperProposer
from .grammar import Descriptor, TaskGrammar
from .hyperproposer import GRPOHyperProposer
from .league import League
from .loop import SelfPlayTrainer, StepStats
from .openended import FixedGridCurriculum, OpenEndedCurriculum, build_curriculum
from .proposer import BanditProposer, BaseProposer, Proposer, learnability
from .verifier import Verifier

__all__ = [
    "SelfPlayTrainer",
    "StepStats",
    "BaseProposer",
    "BanditProposer",
    "Proposer",
    "GRPOHyperProposer",
    "FactoredHyperProposer",
    "League",
    "TaskGrammar",
    "Descriptor",
    "OpenEndedCurriculum",
    "FixedGridCurriculum",
    "build_curriculum",
    "learnability",
    "Verifier",
]
