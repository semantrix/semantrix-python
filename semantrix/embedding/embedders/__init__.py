"""
Embedding implementations for Semantrix.

This module contains concrete implementations of the BaseEmbedder class.
"""

from .sentence_transformer_embedder import SentenceTransformerEmbedder
from .onnx_embedder import OnnxEmbedder

__all__ = [
    'SentenceTransformerEmbedder',
    'OnnxEmbedder',
]
