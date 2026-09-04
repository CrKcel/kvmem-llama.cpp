#!/usr/bin/env python3
"""Equal-budget needle matrix + identity at block=32, on RTX 5050.

Runs:
  1) identity canary (CLI default block=32)
  2) recency  budget=256  — expect miss
  3) retrieval budget=256 — true revive? (same window size as recency)
  4) retrieval budget=256 --force-substr BLUEBIRD-42 — writeback oracle
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDLE = ROOT / "scripts" / "needle_recall.py"
IDENTITY = ROOT / "scripts" / "identity_canary.py"

MODELS = [
    ROOT / "models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
    ROOT / "models/unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf",
    ROOT / "models/unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
]


def run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    return p.returncode


def main() -> int:
    rc = 0
    print("========== identity block=32 (0.6B) ==========")
    r = run([sys.executable, str(IDENTITY), "--model", str(MODELS[0])])
    if r != 0:
        rc = r

    for model in MODELS:
        if not model.is_file():
            print(f"skip missing {model}")
            continue
        name = model.name
        print(f"\n################ {name} ################")
        for args in (
            ["--method", "recency", "--budget", "256", "--block-tokens", "32"],
            ["--method", "retrieval", "--budget", "256", "--block-tokens", "32"],
            ["--method", "retrieval", "--budget", "256", "--block-tokens", "32",
             "--force-substr", "BLUEBIRD-42"],
        ):
            r = run([sys.executable, str(NEEDLE), "-m", str(model), *args])
            if r != 0:
                rc = r
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
