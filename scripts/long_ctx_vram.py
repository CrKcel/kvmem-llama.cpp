#!/usr/bin/env python3
"""P3-1 VRAM canary: GPU attention KV bytes stay constant as prompt grows.

Runs the same budget/slot-pool on RTX 5050 with prompts of 8K, 32K, 64K, and
(optionally) 128K tokens. Asserts KVMEM_KV_BYTES / slot-pool cells do not grow
with T. Prefill pressure must fire once T exceeds the budget.
"""
from __future__ import annotations

import argparse
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
    raise SystemExit("llama-kvmem-cli not found; run scripts/build-cuda.sh")


def parse_stats(stderr: str) -> dict:
    info = {
        "bytes": None,
        "cells": None,
        "budget": None,
        "n_prompt": None,
        "pressure": 0,
        "device_ok": "RTX 5050" in stderr,
    }
    for ln in stderr.splitlines():
        if m := re.search(r"KVMEM_KV_BYTES bytes=(\d+) cells=(\d+).*budget=(\d+)", ln):
            info["bytes"] = int(m.group(1))
            info["cells"] = int(m.group(2))
            info["budget"] = int(m.group(3))
        if m := re.search(r"n_prompt=(\d+)", ln):
            info["n_prompt"] = int(m.group(1))
        if "prefill_pressure" in ln:
            info["pressure"] += 1
    return info


def run(cli: Path, model: Path, prompt: str, n_ctx: int, budget: int,
        gen_reserve: int, batch: int, block_tokens: int) -> dict:
    env = gpu_env.apply_gpu(os.environ.copy(), "small")
    env.setdefault("KVMEM_TRACE", "1")
    with tempfile.NamedTemporaryFile("w", prefix="kvmem_long_", suffix=".txt",
                                     delete=False) as fh:
        fh.write(prompt)
        path = fh.name
    try:
        cmd = [
            str(cli), "-m", str(model), "-n", "1", "-ngl", "99", "--no-prompt",
            "--kvmem",
            "--kvmem-method", "recency",
            "--kvmem-budget", str(budget),
            "--kvmem-gen-reserve", str(gen_reserve),
            "--kvmem-block-tokens", str(block_tokens),
            "-c", str(n_ctx),
            "-b", str(batch),
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
    return parse_stats(proc.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--gen-reserve", type=int, default=128)
    ap.add_argument("--block-tokens", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--skip-128k", action="store_true")
    args = ap.parse_args()

    lengths = [8192, 32768, 65536]
    if not args.skip_128k:
        lengths.append(131072)

    cli = find_cli()
    rows = []
    for ntok in lengths:
        prompt = " ping" * ntok
        print(f"--- target~{ntok} ---", flush=True)
        st = run(cli, args.model, prompt, args.ctx, args.budget,
                 args.gen_reserve, args.batch, args.block_tokens)
        print(
            f"  n_prompt={st['n_prompt']} cells={st['cells']} "
            f"kv_bytes={st['bytes']} pressure={st['pressure']}",
            flush=True,
        )
        if st["bytes"] is None or st["cells"] is None:
            print("FAIL: missing KVMEM_KV_BYTES log", file=sys.stderr)
            return 1
        if st["n_prompt"] is None or st["n_prompt"] < ntok // 2:
            print(f"FAIL: n_prompt={st['n_prompt']} too small for target {ntok}",
                  file=sys.stderr)
            return 1
        rows.append((ntok, st))

    base_bytes = rows[0][1]["bytes"]
    base_cells = rows[0][1]["cells"]
    ok = True
    print("\n========== P3-1 VRAM ==========")
    for ntok, st in rows:
        same = st["bytes"] == base_bytes and st["cells"] == base_cells
        over = st["n_prompt"] > args.budget
        pressure_ok = (st["pressure"] > 0) if over else True
        mark = "PASS" if same and pressure_ok else "FAIL"
        if mark == "FAIL":
            ok = False
        print(
            f"  {mark} T~{ntok:6d} n_prompt={st['n_prompt']:6d} "
            f"cells={st['cells']} bytes={st['bytes']} pressure={st['pressure']}"
        )
        if not same:
            print(f"       expected cells={base_cells} bytes={base_bytes}",
                  file=sys.stderr)
        if over and not pressure_ok:
            print("       expected prefill_pressure once T > budget",
                  file=sys.stderr)

    if ok:
        print("PASS: GPU attention KV bytes/cells constant across prompt lengths")
        return 0
    print("FAIL: KV working set grew with T or pressure never fired", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
