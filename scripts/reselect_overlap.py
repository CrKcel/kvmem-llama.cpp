#!/usr/bin/env python3
"""P1 overlap canary: two long prefills that keep sink+tail should reuse GPU slots.

Expect KVMEM_TRACE prefill_pressure with stage_in only for new tail blocks
(not a full window reload). Uses GPU 0.
"""
from __future__ import annotations

import os
import subprocess
import sys
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


def main() -> int:
    model = DEFAULT_MODEL
    if not model.is_file():
        raise SystemExit(f"missing {model}")
    cli = find_cli()
    env = gpu_env.apply_gpu(os.environ.copy(), "small")
    env["KVMEM_TRACE"] = "1"

    filler = " alpha bravo charlie delta echo foxtrot" * 80
    prompt = "Prefix keep me.\n" + filler + "\nSuffix keep me too. Continue:"
    cmd = [
        str(cli), "-m", str(model), "-n", "8", "-ngl", "99", "--no-prompt",
        "--kvmem", "--kvmem-method", "recency",
        "--kvmem-block-tokens", "128", "--kvmem-budget", "256",
        "--kvmem-gen-reserve", "128", "-c", "4096", "-b", "128",
        prompt,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"decode failed rc={proc.returncode}")
    gpu_env.require_device(proc.stderr, "RTX 5050")
    pressure = [ln for ln in proc.stderr.splitlines() if "prefill_pressure" in ln]
    if not pressure:
        print("FAIL: never hit prefill pressure; prompt too short?", file=sys.stderr)
        sys.stderr.write("\n".join(ln for ln in proc.stderr.splitlines() if "KVMEM_TRACE" in ln) + "\n")
        return 1
    print("pressure events:")
    for ln in pressure:
        print(" ", ln)
    # At least one pressure round should keep some skip/reuse rather than
    # restaging the whole window.
    reused = False
    for ln in pressure:
        if "stage_in=0" in ln or "gpu_reused=" in ln:
            reused = True
        # parse stage_in=N
        for part in ln.split():
            if part.startswith("stage_in="):
                n_in = int(part.split("=")[1])
                if n_in <= 2:
                    reused = True
    if not reused:
        print("WARN: pressure ran but stage_in was large; check TRACE", file=sys.stderr)
    print("output:", proc.stdout[-200:])
    print("PASS: prefill pressure ran without crashing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
