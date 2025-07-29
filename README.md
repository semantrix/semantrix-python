# Semantrix · [![SSPL License](https://img.shields.io/badge/License-SSPL--1.0-ff69b4)](LICENSE)  
**Blazing-fast semantic caching for LLMs in Python, Java, and Go.**  

## Features  
- 🚀 Multi-language support (Python/Java/Go)  
- 🔍 Semantic matching (not just exact matches)  
- ⚡ Async-first API for high performance
- 🛡️ Resource management with async context managers
- 📦 Lightweight and efficient

## Quick Start  

### Installation
```bash
# Python
pip install semantrix-python

# Go
go get github.com/semantrix/go
```

### Basic Usage (Synchronous)
```python
from semantrix import Semantrix

# Create a cache instance
cache = Semantrix()

# Add to cache
cache.set("Explain quantum physics", "It's about...")

# Get from cache (with semantic matching)
response = cache.get("What is quantum theory?")
print(response)  # Returns cached response
```

### Async Usage (Recommended)

```python
import asyncio
from semantrix import Semantrix

async def main():
    # Create a cache instance (no cleanup needed)
    cache = Semantrix()
    
    # Add to cache
    await cache.set("Explain quantum physics", "It's about...")
    
    # Get from cache (with semantic matching)
    response = await cache.get("What is quantum theory?")
    print(response)  # Returns cached response
    
    # The cache will automatically clean up when it's garbage collected

# Run the async function
asyncio.run(main())
```

### Simplified Usage

You don't need to worry about context managers - the cache manages its resources automatically:

```python
import asyncio
from semantrix import Semantrix

async def main():
    # Just create and use - cleanup is automatic
    cache = Semantrix()
    await cache.set("key1", "value1")
    value = await cache.get("key1")
    print(f"Got value: {value}")
    # No need to close anything - it's handled automatically

asyncio.run(main())
```

### Advanced: Custom Configuration

```python
import asyncio
from semantrix import Semantrix
from semantrix.cache_store.stores import InMemoryStore
from semantrix.cache_store.strategies import LRUEvictionStrategy, TTLStrategy
from semantrix.cache_store.eviction_policies import StrategyBasedEvictionPolicy

async def main():
    # Create a custom cache store with LRU + TTL eviction
    policy = StrategyBasedEvictionPolicy(TTLStrategy(ttl_seconds=3600))
    
    # Configure with memory pressure detection and custom settings
    cache_store = InMemoryStore(
        max_size=1000,                # Max items in cache
        eviction_policy=policy,       # Our custom policy
        enable_ttl=True,              # Enable TTL eviction
        ttl_seconds=3600,             # 1 hour TTL
        eviction_interval=60.0,       # Check every minute
        enable_memory_pressure=True,  # Auto-evict under memory pressure
        memory_pressure_threshold=0.8 # 80% memory usage threshold
    )
    
    # Initialize with custom configuration
    cache = Semantrix(
        similarity_threshold=0.9,  # How similar queries should be to match
        cache_store=cache_store    # Our custom store
    )
    
    # Use the cache normally - no cleanup needed
    await cache.set("key", "value")
    value = await cache.get("key")
    print(f"Got value: {value}")

asyncio.run(main())
```
```

## License  
Semantrix uses the [SSPL-1.0 license](LICENSE). **Commercial hosting requires**:  
- Open-sourcing your service stack, OR  
- A [commercial license](mailto:licensing@semantrix.org)  

---
[📚 Documentation](--) | [💬 Discuss](--)  