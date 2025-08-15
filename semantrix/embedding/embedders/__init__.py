"""
Embedding implementations for Semantrix.

This module contains concrete implementations of the BaseEmbedder class.
"""

from .sentence_transformer_embedder import SentenceTransformerEmbedder

# Make other embedders optional
try:
    from .onnx_embedder import OnnxEmbedder
    ONNX_AVAILABLE = True
except ImportError:
    OnnxEmbedder = None
    ONNX_AVAILABLE = False

try:
    from .openai_embedder import OpenAIEmbedder
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAIEmbedder = None
    OPENAI_AVAILABLE = False

try:
    from .langchain_embedder import LangChainEmbedder
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LangChainEmbedder = None
    LANGCHAIN_AVAILABLE = False

try:
    from .ollama_embedder import OllamaEmbedder
    OLLAMA_AVAILABLE = True
except ImportError:
    OllamaEmbedder = None
    OLLAMA_AVAILABLE = False

try:
    from .mistral_embedder import MistralEmbedder
    MISTRAL_AVAILABLE = True
except ImportError:
    MistralEmbedder = None
    MISTRAL_AVAILABLE = False

try:
    from .cohere_embedder import CohereEmbedder
    COHERE_AVAILABLE = True
except ImportError:
    CohereEmbedder = None
    COHERE_AVAILABLE = False

__all__ = [
    'SentenceTransformerEmbedder',
    'OnnxEmbedder',
    'OpenAIEmbedder',
    'LangChainEmbedder',
    'OllamaEmbedder',
    'MistralEmbedder',
    'CohereEmbedder',
    'ONNX_AVAILABLE',
    'OPENAI_AVAILABLE',
    'LANGCHAIN_AVAILABLE',
    'OLLAMA_AVAILABLE',
    'MISTRAL_AVAILABLE',
    'COHERE_AVAILABLE',
]
