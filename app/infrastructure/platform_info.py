"""Hardware and platform detection utility."""

import os
import platform
import sys
from typing import Any, Dict


def get_platform_info() -> Dict[str, Any]:
    """Inspects host operating system, architecture, Python runtime, and resources."""
    os_name = platform.system()
    machine = platform.machine().lower()
    is_rpi = False
    model_name = "Unknown"

    # Check Raspberry Pi device tree model on Linux
    if os_name == "Linux" and os.path.exists("/proc/device-tree/model"):
        try:
            with open("/proc/device-tree/model", "r") as f:
                model_name = f.read().strip("\x00\n ")
                if "Raspberry Pi" in model_name:
                    is_rpi = True
        except Exception:
            pass
    elif os_name == "Windows":
        model_name = f"Windows PC ({platform.version()})"

    # Check GPU / CUDA availability if torch is installed
    cuda_available = False
    gpu_name = None
    try:
        import torch  # type: ignore
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "os": os_name,
        "platform_release": platform.release(),
        "architecture": machine,
        "is_raspberry_pi": is_rpi,
        "device_model": model_name,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count() or 1,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }
