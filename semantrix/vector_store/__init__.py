"""
Semantrix Vector Store Module

This module provides vector storage and similarity search functionality.
It includes a base interface and various implementations for different
vector databases.
"""

from .base import BaseVectorStore
from .stores.faiss import FAISSVectorStore

__all__ = [
    'BaseVectorStore',
    'FAISSVectorStore',
]
