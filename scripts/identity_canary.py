#!/usr/bin/env python3
"""P1 identity canary: greedy tokens with KVMem-off vs KVMem-on (budget >= prompt).

Uses GPU 0 (RTX 5050) via CUDA_VISIBLE_DEVICES. Never touches GPU 1.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
DEFAULT_PROMPT = "Hello my name is"


def find_cli() -> Path:
    for p in (
        ROOT / "build/bin/llama-kvmem-cli",
        ROOT / "build/llama-kvmem-cli",
    ):
        if p.is_file():
            return p
    raise SystemExit("llama-kvmem-cli not found; run scripts/build-cuda.sh")


def run(cli: Path, model: Path, extra: list[str], prompt: str) -> list[int]:
    env = gpu_env.apply_gpu(os.environ.copy(), "small")
    cmd = [
        str(cli),
        "-m",
        str(model),
        "-n",
        "32",
        "-ngl",
        "99",
        "--tokens-only",
        *extra,
        prompt,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed rc={proc.returncode}: {' '.join(cmd)}")
    gpu_env.require_device(proc.stderr, "RTX 5050")
    ids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            ids.append(int(line))
    if not ids:
        raise SystemExit(f"no tokens produced: {' '.join(cmd)}\n{proc.stderr}")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("-p", "--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"missing model {args.model}")

    cli = find_cli()
    off = run(cli, args.model, [], args.prompt)
    on = run(cli, args.model, ["--kvmem", "--kvmem-method", "recency",
                               "--kvmem-block-tokens", "32"], args.prompt)
    print("off", off)
    print("on ", on)
    if off != on:
        print("FAIL: KVMem-on identity mismatch vs KVMem-off", file=sys.stderr)
        return 1
    print("PASS: identity (KVMem-on greedy tokens == KVMem-off)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
