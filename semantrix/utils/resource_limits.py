import psutil
import warnings

class ResourceLimits:
    def __init__(
        self,
        max_memory_gb: float = 1.0,    # Default 1GB
        max_cpu_percent: float = 20.0   # Max 20% CPU utilization
    ):
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent
        
    def allow_operation(self) -> bool:
        process = psutil.Process()
        
        # Memory check
        mem_gb = process.memory_info().rss / (1024 ** 3)
        if mem_gb > self.max_memory_gb:
            warnings.warn(f"Memory limit exceeded ({mem_gb:.2f}GB > {self.max_memory_gb}GB)")
            return False
            
        # CPU check (1s window)
        if psutil.cpu_percent(interval=0.1) > self.max_cpu_percent:
            warnings.warn("CPU limit exceeded")
            return False
            
        return True 