import psutil
import warnings
from typing import Optional

# Default resource limits
DEFAULT_MAX_MEMORY_GB = 0.5      # 512MB
DEFAULT_MAX_CPU_PERCENT = 60.0   # 60% CPU
DEFAULT_MAX_MEMORY_PERCENT = 50.0  # 50% memory

class ResourceLimits:
    def __init__(
        self,
        max_memory_gb: Optional[float] = None,    # Use absolute memory limit (GB)
        max_cpu_percent: float = DEFAULT_MAX_CPU_PERCENT,
        max_memory_percent: Optional[float] = None  # Use percentage memory limit
    ):
        # Validate that only one memory limit is specified
        if max_memory_gb is not None and max_memory_percent is not None:
            raise ValueError("Specify either max_memory_gb OR max_memory_percent, not both")
        if max_memory_gb is None and max_memory_percent is None:
            # Default to absolute memory limit
            max_memory_gb = DEFAULT_MAX_MEMORY_GB
            
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        
    def allow_operation(self) -> bool:
        process = psutil.Process()
        
        # Memory check (either absolute or percentage, not both)
        if self.max_memory_gb is not None:
            mem_gb = process.memory_info().rss / (1024 ** 3)
            if mem_gb > self.max_memory_gb:
                warnings.warn(f"Memory limit exceeded ({mem_gb:.2f}GB > {self.max_memory_gb}GB)")
                return False
        elif self.max_memory_percent is not None:
            mem_percent = psutil.virtual_memory().percent
            if mem_percent > self.max_memory_percent:
                warnings.warn(f"Memory percentage limit exceeded ({mem_percent:.1f}% > {self.max_memory_percent}%)")
                return False
            
        # CPU check (1s window)
        if psutil.cpu_percent(interval=0.1) > self.max_cpu_percent:
            warnings.warn("CPU limit exceeded")
            return False
            
        return True
    
    def is_memory_high(self) -> bool:
        """Check if memory usage is high."""
        process = psutil.Process()
        
        if self.max_memory_gb is not None:
            mem_gb = process.memory_info().rss / (1024 ** 3)
            return mem_gb > (self.max_memory_gb * 0.8)  # 80% of limit
        elif self.max_memory_percent is not None:
            mem_percent = psutil.virtual_memory().percent
            return mem_percent > (self.max_memory_percent * 0.8)  # 80% of limit
        
        return False 