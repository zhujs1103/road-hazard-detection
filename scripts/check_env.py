from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys


def check_import(name: str) -> None:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"[FAIL] import {name}: {exc}")
        return
    version = getattr(module, "__version__", "unknown")
    print(f"[ OK ] import {name}: {version}")


def main() -> int:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Executable: {sys.executable}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        print(f"nvidia-smi: {nvidia_smi}")
        try:
            result = subprocess.run(
                [nvidia_smi],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(result.stdout.splitlines()[0])
        except Exception as exc:  # pragma: no cover - diagnostic script
            print(f"[WARN] could not run nvidia-smi: {exc}")
    else:
        print("[WARN] nvidia-smi not found on PATH")

    for name in [
        "torch",
        "torchvision",
        "ultralytics",
        "cv2",
        "pandas",
        "yaml",
        "gdown",
        "ensemble_boxes",
    ]:
        check_import(name)

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"[FAIL] torch CUDA check: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
