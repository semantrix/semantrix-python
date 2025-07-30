"""
Embedding implementations for Semantrix.

This module contains concrete implementations of the BaseEmbedder class.
"""

from semantrix.embedding.implementations.sentence_transformer_embedder import SentenceTransformerEmbedder
from semantrix.embedding.implementations.onnx_embedder import OnnxEmbedder

__all__ = [
    'SentenceTransformerEmbedder',
    'OnnxEmbedder',
]
