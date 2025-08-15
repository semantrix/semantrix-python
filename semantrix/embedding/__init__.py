"""
Semantrix Embedding Module

This module provides an interface for text embedding models with async support.
"""

from typing import Optional, Type, Dict, Any, TypeVar

from .base import BaseEmbedder
from .embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

# Make other embedders optional
try:
    from .embedders.onnx_embedder import OnnxEmbedder
    ONNX_AVAILABLE = True
except ImportError:
    OnnxEmbedder = None
    ONNX_AVAILABLE = False

try:
    from .embedders.openai_embedder import OpenAIEmbedder
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAIEmbedder = None
    OPENAI_AVAILABLE = False

# Other optional imports
try:
    from .embedders.langchain_embedder import LangChainEmbedder
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LangChainEmbedder = None
    LANGCHAIN_AVAILABLE = False

try:
    from .embedders.ollama_embedder import OllamaEmbedder
    OLLAMA_AVAILABLE = True
except ImportError:
    OllamaEmbedder = None
    OLLAMA_AVAILABLE = False

try:
    from .embedders.mistral_embedder import MistralEmbedder
    MISTRAL_AVAILABLE = True
except ImportError:
    MistralEmbedder = None
    MISTRAL_AVAILABLE = False

try:
    from .embedders.cohere_embedder import CohereEmbedder
    COHERE_AVAILABLE = True
except ImportError:
    CohereEmbedder = None
    COHERE_AVAILABLE = False

# Type variable for embedder classes
EmbedderT = TypeVar('EmbedderT', bound=BaseEmbedder)

# Default embedder configuration
DEFAULT_EMBEDDERS: Dict[str, Type[BaseEmbedder]] = {
    'sentence-transformers': SentenceTransformerEmbedder,
    'onnx': OnnxEmbedder,
    'openai': OpenAIEmbedder,
    'langchain': LangChainEmbedder,
    'ollama': OllamaEmbedder,
    'mistral': MistralEmbedder,
    'cohere': CohereEmbedder,
}

def get_embedder(embedder_type: str, **kwargs: Any) -> BaseEmbedder:
    """
    Get an embedder instance by type.
    
    Args:
        embedder_type: Type of embedder to create (e.g., 'sentence-transformers', 'onnx')
        **kwargs: Additional arguments to pass to the embedder constructor
        
    Returns:
        An instance of the requested embedder
        
    Raises:
        ValueError: If the requested embedder type is not available or not supported
    """
    # Normalize the embedder type
    embedder_type = embedder_type.lower().replace('_', '-').strip()
    
    # Check if the requested embedder is available
    if embedder_type not in DEFAULT_EMBEDDERS or DEFAULT_EMBEDDERS[embedder_type] is None:
        available = [k for k, v in DEFAULT_EMBEDDERS.items() if v is not None]
        raise ValueError(
            f"Embedder type '{embedder_type}' is not available. "
            f"Available embedders: {', '.join(available)}"
        )
    
    # Create and return the embedder instance
    embedder_class = DEFAULT_EMBEDDERS[embedder_type]
    return embedder_class(**kwargs)

__all__ = [
    'BaseEmbedder',
    'SentenceTransformerEmbedder',
    'OnnxEmbedder',
    'OpenAIEmbedder',
    'LangChainEmbedder',
    'OllamaEmbedder',
    'MistralEmbedder',
    'CohereEmbedder',
    'get_embedder',
    'ONNX_AVAILABLE',
    'OPENAI_AVAILABLE',
    'LANGCHAIN_AVAILABLE',
    'OLLAMA_AVAILABLE',
    'MISTRAL_AVAILABLE',
    'COHERE_AVAILABLE',
]
