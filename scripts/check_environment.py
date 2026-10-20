from __future__ import annotations
import importlib.metadata
import platform
import sys


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"OS: {platform.platform()}")
    for package in ("ultralytics", "torch", "torchvision", "opencv-python", "PyYAML", "Pillow"):
        print(f"{package}: {version(package)}")
    try:
        import torch
        print(f"PyTorch CUDA build: {torch.version.cuda}")
        available = torch.cuda.is_available()
        print(f"CUDA available: {available}")
        if available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU count: {torch.cuda.device_count()}")
        else:
            print("GPU not available to PyTorch. Check NVIDIA driver, CUDA wheel compatibility, and VM GPU access.")
    except Exception as exc:
        print(f"PyTorch diagnostic failed: {type(exc).__name__}: {exc}")
        return 1
    try:
        import ultralytics
        print(f"Ultralytics import OK: {ultralytics.__version__}")
    except Exception as exc:
        print(f"Ultralytics import failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
