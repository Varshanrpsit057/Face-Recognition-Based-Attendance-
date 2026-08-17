"""
Device Manager — hardware detection, ONNX Runtime provider selection,
and system resource monitoring. Singleton pattern.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil
import onnxruntime as ort

from config import EXECUTION_PROVIDER_PRIORITY


@dataclass
class DeviceInfo:
    """System hardware and capabilities snapshot."""
    os_name: str
    os_version: str
    python_version: str
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: str
    gpu_memory_gb: float
    cuda_available: bool
    cuda_version: str
    cudnn_available: bool
    cudnn_version: str
    tensorrt_available: bool
    tensorrt_version: str
    onnx_providers: List[str]
    best_provider: str


class DeviceManager:
    """Singleton that detects hardware and selects optimal ONNX execution providers."""

    _instance: Optional["DeviceManager"] = None

    def __new__(cls) -> "DeviceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self) -> None:
        self.device_info: Optional[DeviceInfo] = None
        self.refresh()

    # ── Detection ──────────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-detect all hardware and update DeviceInfo."""
        os_name = platform.system()
        os_version = platform.release()
        python_version = platform.python_version()
        cpu_name = platform.processor() or "Unknown CPU"
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_threads = psutil.cpu_count(logical=True) or 1

        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024 ** 3)
        available_ram_gb = vm.available / (1024 ** 3)

        # GPU detection via nvidia-smi
        gpu_available = False
        gpu_name = ""
        gpu_memory_gb = 0.0
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.STDOUT,
                text=True,
            ).strip()
            if output:
                gpu_available = True
                parts = output.split(", ")
                gpu_name = parts[0]
                if len(parts) > 1:
                    gpu_memory_gb = float(parts[1]) / 1024.0
        except Exception:
            pass

        # ONNX providers
        onnx_providers = ort.get_available_providers()
        cuda_available = "CUDAExecutionProvider" in onnx_providers
        tensorrt_available = "TensorrtExecutionProvider" in onnx_providers

        # Best provider from priority list
        # EXECUTION_PROVIDER_PRIORITY is List[ExecutionProvider] enum
        best_provider = "CPUExecutionProvider"
        for ep in EXECUTION_PROVIDER_PRIORITY:
            ep_str = ep.value if hasattr(ep, "value") else str(ep)
            if ep_str in onnx_providers:
                best_provider = ep_str
                break

        # CUDA version
        cuda_version = ""
        if cuda_available:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True, stderr=subprocess.STDOUT,
                ).strip()
                cuda_version = out
            except Exception:
                pass

        self.device_info = DeviceInfo(
            os_name=os_name,
            os_version=os_version,
            python_version=python_version,
            cpu_name=cpu_name,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            cudnn_available=False,
            cudnn_version="",
            tensorrt_available=tensorrt_available,
            tensorrt_version="",
            onnx_providers=onnx_providers,
            best_provider=best_provider,
        )

    # ── Public API ─────────────────────────────────────────────────

    def get_device_info(self) -> DeviceInfo:
        """Return the current DeviceInfo snapshot."""
        if self.device_info is None:
            self.refresh()
        return self.device_info

    def get_best_providers(self) -> List[str]:
        """Return available ONNX providers in priority order, always ending with CPU."""
        if self.device_info is None:
            self.refresh()
        providers: List[str] = []
        for ep in EXECUTION_PROVIDER_PRIORITY:
            ep_str = ep.value if hasattr(ep, "value") else str(ep)
            if ep_str in self.device_info.onnx_providers and ep_str not in providers:
                providers.append(ep_str)
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers

    def get_best_device(self) -> str:
        """Return 'cuda' or 'cpu'."""
        if self.device_info is None:
            self.refresh()
        if self.device_info.cuda_available or self.device_info.gpu_available:
            return "cuda"
        return "cpu"

    def get_onnx_session_options(self) -> ort.SessionOptions:
        """Return optimised ONNX SessionOptions for the current hardware."""
        if self.device_info is None:
            self.refresh()
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(self.device_info.cpu_cores, 4)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options

    def is_gpu_available(self) -> bool:
        """Check if any GPU is available."""
        if self.device_info is None:
            self.refresh()
        return self.device_info.gpu_available or self.device_info.cuda_available

    def get_memory_usage(self) -> Dict[str, float]:
        """Current process memory usage in MB."""
        process = psutil.Process()
        mem = process.memory_info()
        return {
            "rss_mb": mem.rss / (1024 * 1024),
            "vms_mb": mem.vms / (1024 * 1024),
        }

    def get_cpu_usage(self) -> float:
        """Current CPU utilisation percentage."""
        return psutil.cpu_percent(interval=None)

    def get_gpu_usage(self) -> Optional[float]:
        """GPU utilisation if available, else None."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.STDOUT,
            ).strip()
            return float(out)
        except Exception:
            return None

    def summary(self) -> str:
        """Human-readable summary of the system."""
        d = self.get_device_info()
        gpu_part = f"GPU: {d.gpu_name} ({d.gpu_memory_gb:.1f} GB)" if d.gpu_available else "GPU: None"
        return (
            f"OS: {d.os_name} {d.os_version} | CPU: {d.cpu_name} "
            f"({d.cpu_cores}c/{d.cpu_threads}t) | RAM: {d.total_ram_gb:.1f} GB | "
            f"{gpu_part} | Provider: {d.best_provider}"
        )
