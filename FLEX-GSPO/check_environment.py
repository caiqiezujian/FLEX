#!/usr/bin/env python3
"""Read-only server inventory. Does not install, download or load model weights."""
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys


def main():
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "ffmpeg": shutil.which("ffmpeg"),
              "ffprobe": shutil.which("ffprobe"), "packages": {}}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy",
                 "sentencepiece", "safetensors", "requests", "vllm"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    if shutil.which("nvidia-smi"):
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False)
        result["gpu_inventory"] = completed.stdout.strip()
        result["nvidia_smi_error"] = completed.stderr.strip()
    try:
        import torch
        result["torch_cuda_build"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        result["cuda_available"] = False
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            result["cpu_memory"] = [line.strip() for line in handle
                                    if line.startswith(("MemTotal:", "MemAvailable:"))]
    except OSError:
        result["cpu_memory"] = "not available on this host"
    result["note"] = "Environment inventory only; no GPU training or reward-quality validation performed."
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
