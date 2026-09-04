"""Pin CUDA to this machine's RTX 5050 / 5090.

CUDA defaults to FASTEST_FIRST, so device 0 is the 5090. Always set
CUDA_DEVICE_ORDER=PCI_BUS_ID and bind by UUID.
"""
from __future__ import annotations

import os
from pathlib import Path

UUID_5050 = "GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29"
UUID_5090 = "GPU-58a7c28b-e698-307f-c149-24d4ecd88bf4"

ROOT = Path(__file__).resolve().parents[1]


def apply_gpu(env: dict, which: str = "small") -> dict:
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if which in ("small", "5050", "lt27b"):
        env["CUDA_VISIBLE_DEVICES"] = UUID_5050
        env["KVMEM_GPU_NAME"] = "RTX 5050"
    elif which in ("27b", "5090"):
        env["CUDA_VISIBLE_DEVICES"] = UUID_5090
        env["KVMEM_GPU_NAME"] = "RTX 5090"
    else:
        raise ValueError(f"unknown gpu selector {which}")
    libdir = str(ROOT / "build/bin")
    cu13 = "/home/leye/kvmem_qw3/.cu13-env/lib"
    extra = libdir + ":" + cu13
    env["LD_LIBRARY_PATH"] = extra + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def require_device(stderr: str, expect: str = "RTX 5050") -> None:
    if expect not in stderr:
        raise SystemExit(
            f"refusing to continue: expected CUDA device {expect} in llama logs, got:\n"
            + "\n".join(
                ln for ln in stderr.splitlines()
                if "CUDA" in ln or "Device" in ln or "offload" in ln
            )[:2000]
        )
    if "offloaded" not in stderr or "to GPU" not in stderr:
        raise SystemExit("refusing to continue: layers were not offloaded to GPU")
