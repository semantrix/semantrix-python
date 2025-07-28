from collections import OrderedDict

class LRUEviction:
    def __init__(self, max_size: int = 10_000):
        self.cache = OrderedDict()
        self.max_size = max_size
        
    def get_exact(self, prompt: str) -> str | None:
        if prompt in self.cache:
            self.cache.move_to_end(prompt)
            return self.cache[prompt]
        return None
        
    def add(self, prompt: str, response: str):
        self.cache[prompt] = response
        self.cache.move_to_end(prompt)
        
    def enforce_limits(self, limits):
        """Called after each set() operation"""
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest 