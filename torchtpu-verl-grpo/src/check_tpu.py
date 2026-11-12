#!/usr/bin/env python3
"""
Hardware & Environment Health Check for TorchTPU
Verifies Native PyTorch on Google Cloud TPU (v5e / v6e Trillium)
"""

import sys
import time

def check_environment():
    print("=" * 60)
    print("🔍 TorchTPU Hardware & Environment Diagnostic")
    print("=" * 60)

    # 1. Check Python version
    py_ver = sys.version.split()[0]
    print(f"🐍 Python Version: {py_ver}")
    if sys.version_info < (3, 11):
        print("⚠️ Warning: Python 3.11+ is strongly recommended for TorchTPU.")

    # 2. Check PyTorch
    try:
        import torch
        print(f"🔥 PyTorch Version: {torch.__version__}")
    except ImportError:
        print("❌ Error: PyTorch is not installed.")
        return False

    # 3. Check TorchTPU plugin
    try:
        import torch_tpu
        print(f"💎 TorchTPU Plugin: Detected ({getattr(torch_tpu, '__version__', 'latest')})")
    except ImportError:
        print("ℹ️ Note: torch_tpu plugin module not directly imported, checking device availability...")

    # 4. Check device availability
    is_tpu_available = False
    try:
        if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "is_available"):
            acc_name = torch.accelerator.current_accelerator().type
            print(f"🚀 Current Accelerator: {acc_name}")
            is_tpu_available = (acc_name == "tpu")
        elif hasattr(torch, "tpu") and torch.tpu.is_available():
            is_tpu_available = True
    except Exception as e:
        print(f"⚠️ Accelerator check note: {e}")

    try:
        # Direct device test
        tpu_device = torch.device("tpu")
        x = torch.ones((1024, 1024), device=tpu_device, dtype=torch.bfloat16)
        y = torch.matmul(x, x)
        if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
            torch.accelerator.synchronize()
        print(f"✅ Hardware Test Passed: 1024x1024 BFloat16 Matmul executed on {tpu_device}!")
        is_tpu_available = True
    except Exception as e:
        print(f"❌ Direct TPU test result: {e}")

    # 5. Check Ecosystem Packages
    packages = ["torchtitan", "verl", "vllm", "ray"]
    print("\n📦 Ecosystem Package Status:")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"  - {pkg:<12}: ✅ Available ({ver})")
        except ImportError:
            print(f"  - {pkg:<12}: ⏳ Not installed yet")

    print("=" * 60)
    return is_tpu_available

if __name__ == "__main__":
    check_environment()
