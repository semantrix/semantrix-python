"""
Embedding implementations for Semantrix.

This module contains concrete implementations of the BaseEmbedder class.
"""

from .sentence_transformer_embedder import SentenceTransformerEmbedder
from .onnx_embedder import OnnxEmbedder
from .openai_embedder import OpenAIEmbedder
from .langchain_embedder import LangChainEmbedder
from .ollama_embedder import OllamaEmbedder
from .mistral_embedder import MistralEmbedder
from .cohere_embedder import CohereEmbedder

__all__ = [
    'SentenceTransformerEmbedder',
    'OnnxEmbedder',
    'OpenAIEmbedder',
    'LangChainEmbedder',
    'OllamaEmbedder',
    'MistralEmbedder',
    'CohereEmbedder',
]
