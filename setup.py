"""
Semantrix - Semantic Caching for AI Applications

A high-performance semantic cache for AI applications, supporting multiple backends
and advanced features like eviction policies, TTL, and distributed caching.
"""

import os
import re
from setuptools import setup, find_packages

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get package version
with open(os.path.join("semantrix", "__init__.py"), "r", encoding="utf-8") as f:
    version_match = re.search(r'^__version__ = ["\']([^\"\']+)[\"\']', f.read(), re.MULTILINE)
    if version_match:
        VERSION = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string in semantrix/__init__.py")

# Core dependencies
install_requires = [
    "numpy>=1.19.0",
    "pydantic>=1.8.0,<3.0.0",
    "typing-extensions>=4.0.0",
    "tqdm>=4.0.0",
    "requests>=2.25.0",
    "aiohttp>=3.8.0",
    "orjson>=3.6.0",
]

# Optional dependencies
extras_require = {
    # Cache store backends
    "dynamodb": ["boto3>=1.26.0"],
    "elasticache": ["redis>=4.3.0"],
    "google-memorystore": ["google-cloud-redis>=2.0.0"],
    "sqlite": ["aiosqlite>=0.17.0"],
    "mongodb": ["motor>=3.0.0"],
    "postgresql": ["asyncpg>=0.25.0"],
    "documentdb": ["pymongo>=4.0.0"],
    
    # Embedding models
    "sentence-transformers": ["sentence-transformers>=2.2.0"],
    "onnx": ["onnxruntime>=1.12.0"],
    
    # Vector stores
    "faiss": ["faiss-cpu>=1.7.0"],
    "chroma": ["chromadb>=0.4.0"],
    "pinecone": ["pinecone-client>=2.2.0"],
    "qdrant": ["qdrant-client>=1.1.0"],
    "milvus": ["pymilvus>=2.2.0"],
    
    # Development and testing
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.20.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "isort>=5.0.0",
        "mypy>=0.990",
        "types-requests>=2.28.0",
        "types-python-dateutil>=2.8.0",
    ],
    
    # Documentation
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "sphinx-autodoc-typehints>=1.15.0",
        "myst-parser>=0.18.0",
    ],
}

# Create 'all' extra that includes all optional dependencies
extras_require["all"] = list(
    {dep for deps in extras_require.values() for dep in deps if not dep.startswith("dev") and not dep.startswith("docs")}
)

setup(
    name="semantrix",
    version=VERSION,
    author="Semantrix Team",
    author_email="info@semantrix.ai",
    description="A high-performance semantic cache for AI applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/semantrix/semantrix-python",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        "semantrix": ["py.typed"],
    },
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "semantic-cache",
        "ai",
        "machine-learning",
        "nlp",
        "embeddings",
        "vector-search",
        "cache",
        "llm",
        "langchain",
    ],
    project_urls={
        "Bug Reports": "https://github.com/semantrix/semantrix-python/issues",
        "Source": "https://github.com/semantrix/semantrix-python",
        "Documentation": "https://semantrix.readthedocs.io/",
        "License": "https://www.mongodb.com/licensing/server-side-public-license"
    },
    license="SSPL-1.0",
    license_files=("LICENSE",),
    zip_safe=False,
)

# Create a py.typed file for type checking
if not os.path.exists(os.path.join("semantrix", "py.typed")):
    with open(os.path.join("semantrix", "py.typed"), "w") as f:
        f.write("")
        f.flush()
