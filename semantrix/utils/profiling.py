import time
from contextlib import contextmanager
from collections import defaultdict

class Profiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.stats = defaultdict(lambda: {"count": 0, "time": 0.0})
        
    @contextmanager
    def record(self, operation: str):
        if not self.enabled:
            yield
            return
            
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = (time.perf_counter() - start) * 1000  # ms
            self.stats[operation]["count"] += 1
            self.stats[operation]["time"] += duration
            
    def get_stats(self) -> dict:
        #random comment
        return dict(self.stats) if self.enabled else {} 