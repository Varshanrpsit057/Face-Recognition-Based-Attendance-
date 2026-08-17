import time
import psutil
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class LatencyStats:
    count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    p50: float
    p95: float
    p99: float

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Profiler(metaclass=Singleton):
    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, operation: str) -> None:
        self._start_times[operation] = time.perf_counter()

    def stop(self, operation: str) -> float:
        end_time = time.perf_counter()
        if operation not in self._start_times:
            logger.warning(f"Operation {operation} stopped without starting.")
            return 0.0
        duration = end_time - self._start_times.pop(operation)
        self.record(operation, duration)
        return duration

    def record(self, operation: str, duration: float) -> None:
        if operation not in self._timings:
            self._timings[operation] = []
        self._timings[operation].append(duration)

    def get_stats(self, operation: str) -> LatencyStats:
        if operation not in self._timings or not self._timings[operation]:
            return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        data = np.array(self._timings[operation])
        return LatencyStats(
            count=int(len(data)),
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            min_val=float(np.min(data)),
            max_val=float(np.max(data)),
            p50=float(np.percentile(data, 50)),
            p95=float(np.percentile(data, 95)),
            p99=float(np.percentile(data, 99))
        )

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        return {op: self.get_stats(op) for op in self._timings}

    def get_fps(self, operation: str = 'total_pipeline') -> float:
        if operation not in self._timings or not self._timings[operation]:
            return 0.0
        stats = self.get_stats(operation)
        if stats.mean > 0:
            return 1.0 / stats.mean
        return 0.0

    def get_avg_fps(self) -> float:
        return self.get_fps('total_pipeline')

    def get_min_fps(self) -> float:
        if 'total_pipeline' not in self._timings or not self._timings['total_pipeline']:
            return 0.0
        data = np.array(self._timings['total_pipeline'])
        max_duration = float(np.max(data))
        return 1.0 / max_duration if max_duration > 0 else 0.0

    def get_max_fps(self) -> float:
        if 'total_pipeline' not in self._timings or not self._timings['total_pipeline']:
            return 0.0
        data = np.array(self._timings['total_pipeline'])
        min_duration = float(np.min(data))
        return 1.0 / min_duration if min_duration > 0 else 0.0

    def get_memory_usage(self) -> Dict[str, float]:
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'RSS': mem_info.rss / (1024 * 1024),
            'VMS': mem_info.vms / (1024 * 1024)
        }

    def get_cpu_usage(self) -> float:
        return psutil.cpu_percent(interval=None)

    def get_gpu_usage(self) -> Optional[float]:
        return None

    def get_system_metrics(self) -> Dict[str, Any]:
        return {
            'memory_usage_mb': self.get_memory_usage(),
            'cpu_usage_percent': self.get_cpu_usage(),
            'gpu_usage_percent': self.get_gpu_usage()
        }

    def reset(self) -> None:
        self._timings.clear()
        self._start_times.clear()

    def summary(self) -> str:
        stats_dict = self.get_all_stats()
        lines = ["Profiler Summary:"]
        for op, stats in stats_dict.items():
            lines.append(f"  {op}: count={stats.count}, mean={stats.mean:.4f}s, p95={stats.p95:.4f}s")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stats': {op: stats.__dict__ for op, stats in self.get_all_stats().items()},
            'system_metrics': self.get_system_metrics()
        }

def get_profiler() -> Profiler:
    return Profiler()
