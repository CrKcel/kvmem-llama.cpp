#!/usr/bin/env python3
"""P3-2 NVMe spill canary on RTX 5050.

Tiny CPU arena so prefill pressure overflows to NVMe. Asserts stage_out_nvme
trace on a short prompt, then 512K does not OOM and KV cells stay bounded.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"


def find_cli() -> Path:
    for p in (
        ROOT / "build/bin/llama-kvmem-cli",
        ROOT / "build/llama-kvmem-cli",
    ):
        if p.is_file():
            return p
    raise SystemExit("llama-kvmem-cli not found")


def run(cli: Path, model: Path, ntok: int, ctx: int, nvme_dir: str) -> dict:
    env = gpu_env.apply_gpu(os.environ.copy(), "small")
    env.setdefault("KVMEM_TRACE", "1")
    prompt = " ping" * ntok
    with tempfile.NamedTemporaryFile("w", prefix="kvmem_nvme_", suffix=".txt",
                                     delete=False) as fh:
        fh.write(prompt)
        path = fh.name
    try:
        cmd = [
            str(cli), "-m", str(model), "-n", "1", "-ngl", "99", "--no-prompt",
            "--kvmem", "--kvmem-method", "recency",
            "--kvmem-budget", "256", "--kvmem-gen-reserve", "128",
            "--kvmem-block-tokens", "32",
            "--kvmem-cpu-gb", "0.008",
            "--kvmem-nvme-gb", "0.25",
            "--kvmem-nvme-dir", nvme_dir,
            "-c", str(ctx), "-b", "128",
            "-f", path,
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-8000:])
        raise SystemExit(f"decode failed rc={proc.returncode}")
    gpu_env.require_device(proc.stderr, "RTX 5050")
    info = {
        "n_prompt": None,
        "cells": None,
        "bytes": None,
        "nvme_out": 0,
        "nvme_in": 0,
        "cpu_out": 0,
        "tiers": None,
    }
    for ln in proc.stderr.splitlines():
        if m := re.search(r"n_prompt=(\d+)", ln):
            info["n_prompt"] = int(m.group(1))
        if m := re.search(r"KVMEM_KV_BYTES bytes=(\d+) cells=(\d+)", ln):
            info["bytes"] = int(m.group(1))
            info["cells"] = int(m.group(2))
        if "stage_out_nvme" in ln:
            info["nvme_out"] += 1
        if "stage_in_nvme" in ln:
            info["nvme_in"] += 1
        if "stage_out_cpu" in ln:
            info["cpu_out"] += 1
        if ln.startswith("KVMEM_TIERS"):
            info["tiers"] = ln.strip()
            print("  ", ln, flush=True)
    return info


def main() -> int:
    model = DEFAULT_MODEL
    cli = find_cli()
    nvme_dir = tempfile.mkdtemp(prefix="kvmem_nvme_")
    print(f"nvme_dir={nvme_dir}", flush=True)

    print("--- 8K spill ---", flush=True)
    short = run(cli, model, 8192, 16384, nvme_dir)
    print(
        f"  n_prompt={short['n_prompt']} cells={short['cells']} "
        f"cpu_out={short['cpu_out']} nvme_out={short['nvme_out']}",
        flush=True,
    )
    if not short["nvme_out"] and not short["cpu_out"]:
        print("FAIL: no CPU/NVMe stage_out on 8K pressure", file=sys.stderr)
        return 1
    if short["nvme_out"] == 0:
        print("FAIL: CPU did not overflow to NVMe", file=sys.stderr)
        return 1

    print("--- 512K no-OOM ---", flush=True)
    long = run(cli, model, 524288, 524288, nvme_dir)
    print(
        f"  n_prompt={long['n_prompt']} cells={long['cells']} "
        f"kv_bytes={long['bytes']} nvme_out={long['nvme_out']}",
        flush=True,
    )
    if long["n_prompt"] is None or long["n_prompt"] < 400000:
        print(f"FAIL: n_prompt={long['n_prompt']}", file=sys.stderr)
        return 1
    if long["cells"] != short["cells"] or long["bytes"] != short["bytes"]:
        print("FAIL: KV pool grew at 512K", file=sys.stderr)
        return 1
    if long["nvme_out"] == 0:
        print("FAIL: 512K had no NVMe spills", file=sys.stderr)
        return 1
    print("PASS: CPU-full spills to NVMe; 512K did not OOM; KV bytes constant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
