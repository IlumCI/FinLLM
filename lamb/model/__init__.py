"""LAMb model components."""

from __future__ import annotations

from .embeddings import NumberAwareEmbedding
from .lamb import LAMb, build_model
from .latent_core import RecurrentDepthCore
from .memory import NeuralMemory
from .transformer import Block, RMSNorm, SelfAttention, SwiGLU

__all__ = [
    "NumberAwareEmbedding",
    "RecurrentDepthCore",
    "NeuralMemory",
    "Block",
    "RMSNorm",
    "SelfAttention",
    "SwiGLU",
    "LAMb",
    "build_model",
]
