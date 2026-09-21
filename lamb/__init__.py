"""LAMb -- Latent Arithmetic Machine.

A number-native, depth-recurrent latent-reasoning autoregressive model with
test-time neural memory, trained by verifiable arithmetic self-play.

Light imports only at package load (no torch): the tokenizer, native-kernel
dispatch, and configs. Import :mod:`lamb.model` / :mod:`lamb.selfplay` for the
torch-backed pieces.
"""

from __future__ import annotations

from ._native import USING_RUST, backend
from .config import ModelConfig, POETConfig, TrainConfig
from .tokenizer import ArithmeticTokenizer, Encoded

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "USING_RUST",
    "backend",
    "ModelConfig",
    "TrainConfig",
    "POETConfig",
    "ArithmeticTokenizer",
    "Encoded",
]
