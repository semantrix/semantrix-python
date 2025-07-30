"""
Semantrix Embedding Module

This module provides an interface for text embedding models with async support.
"""

from .base import BaseEmbedder
from .embedders import (
    SentenceTransformerEmbedder,
    OnnxEmbedder,
)

__all__ = [
    'BaseEmbedder',
    'SentenceTransformerEmbedder',
    'OnnxEmbedder',
]
