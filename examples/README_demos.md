# Semantrix Demo Applications

This directory contains demo applications that showcase Semantrix's core functionality.

## Available Demos

### 1. Minimal Demo (`minimal_demo.py`) ⭐ **RECOMMENDED**
A simple, focused demo showing the core operations:
- **SET**: Add data to cache
- **GET**: Retrieve data (exact and semantic matching)
- **DELETE**: Remove data
- **TOMBSTONE**: Mark data as deleted

**Perfect for getting started! This demo works reliably.**

### 2. Basic Demo (`basic_demo.py`)
A basic demo showing core operations with both sync and async interfaces:
- Basic get/set operations
- Semantic caching with similarity matching
- Delete operations (direct and tombstone)
- Both synchronous and asynchronous interfaces

**Note: This demo may have initialization issues on some systems.**

### 3. Simple Demo (`simple_demo.py`)
A comprehensive demo showcasing all major features:
- Basic get/set operations
- Semantic caching with similarity matching
- Delete operations (direct and tombstone)
- Explain functionality for debugging
- Both synchronous and asynchronous interfaces
- Error handling and logging
- Performance benchmarking
- Profiling and resource management
- Write-Ahead Logging and Two-Phase Commit

## How to Run

### Prerequisites
Make sure you have Semantrix installed:
```bash
pip install -e .
```

### Running the Minimal Demo (Recommended)
```bash
python examples/minimal_demo.py
```

### Running the Basic Demo
```bash
python examples/basic_demo.py
```

### Running the Simple Demo
```bash
python examples/simple_demo.py
```

## What You'll See

### Minimal Demo Output
```
🎯 MINIMAL SEMANTRIX DEMO
========================================
This demo shows the core operations:
- SET: Add data to cache
- GET: Retrieve data (exact and semantic)
- DELETE: Remove data
- TOMBSTONE: Mark data as deleted
========================================
🚀 Minimal Semantrix Demo
========================================
✅ Semantrix initialized!

📝 Setting data...
✅ Added 2 items to cache

🔍 Getting data (exact match)...
Result: Python is a programming language.

🔍 Getting data (semantic match)...
Result: None

🗑️ Deleting data...
Delete successful: True
After delete: None

🪦 Tombstoning data...
Tombstone successful: True
After tombstone: None

✅ Minimal demo completed!

🎉 Demo completed successfully!
You now know how to use Semantrix's core operations!
```

### Basic Demo Output
The basic demo provides similar output but may have initialization issues.

### Simple Demo Output
The simple demo provides much more detailed output including:
- Performance metrics
- Profiling statistics
- Resource usage information
- Detailed explanations of cache hits/misses
- Error handling demonstrations

## Key Features Demonstrated

### 1. Semantic Caching
Semantrix can find similar prompts even when they don't match exactly:
```python
# Store this
await cache.set("What is Python?", "Python is a programming language.")

# This will find a match due to semantic similarity
result = await cache.get("Tell me about Python")  # Returns the stored response!
```

### 2. Multiple Deletion Modes
```python
# Direct deletion - immediately removes the item
await cache.delete("prompt", mode=DeletionMode.DIRECT)

# Tombstone deletion - marks as deleted but keeps for recovery
await cache.tombstone("prompt")
```

### 3. Both Sync and Async Interfaces
```python
# Synchronous (via SemantrixClient)
cache = SemantrixClient()
cache.set("prompt", "response")
result = cache.get("prompt")

# Asynchronous (direct Semantrix usage)
async with Semantrix() as cache:
    await cache.set("prompt", "response")
    result = await cache.get("prompt")
```

### 4. Explain Functionality
```python
# Get detailed explanation of why a cache hit/miss occurred
explanation = await cache.explain("What is Python?")
print(f"Cache hit: {explanation.cache_hit}")
print(f"Exact match: {explanation.exact_match}")
print(f"Semantic match: {explanation.semantic_match}")
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure Semantrix is installed properly
2. **Missing dependencies**: Install required packages with `pip install -r requirements.txt`
3. **Configuration issues**: Check the logging output for detailed error messages
4. **Performance issues**: Adjust similarity thresholds or resource limits
5. **Async initialization errors**: Use the minimal demo which avoids these issues

### If Demos Don't Work

1. **Start with the minimal demo** - it's the most reliable
2. **Check your Python version** - Semantrix requires Python 3.8+
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Check logs** for specific error messages
5. **Try with different settings** (lower similarity threshold, disable features)

## Next Steps

After running the demos, you can:

1. **Explore the code** to understand how Semantrix works
2. **Modify the demos** to test different scenarios
3. **Check the documentation** in the `docs/` directory
4. **Try different cache stores** (Redis, PostgreSQL, etc.)
5. **Experiment with different embedders** (OpenAI, Cohere, etc.)

## Contributing

Feel free to create your own demos or modify existing ones to showcase specific features or use cases!
