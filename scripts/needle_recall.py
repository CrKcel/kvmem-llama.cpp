#!/usr/bin/env python3
"""Needle canary: recency should miss a mid-prompt fact; retrieval may revive it.

Default GPU is the 5050 (`--gpu small`). 27B must use `--gpu 27b` (5090).
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
NEEDLE = "The secret code is BLUEBIRD-42."


def find_cli() -> Path:
    for p in (
        ROOT / "build/bin/llama-kvmem-cli",
        ROOT / "build/llama-kvmem-cli",
    ):
        if p.is_file():
            return p
    raise SystemExit("llama-kvmem-cli not found; run scripts/build-cuda.sh")


def make_prompt(no_think: bool = False) -> str:
    filler = " lorem ipsum dolor sit amet" * 80
    body = (
        "Read the following notes and then answer: what is the secret code?\n"
        + filler
        + "\n"
        + NEEDLE
        + "\n"
        + filler
        + "\nQuestion: what is the secret code?"
    )
    if not no_think:
        return body + " Answer:"
    # Qwen3 / 3.5 / 3.8 enable_thinking=false: empty think block, then answer.
    return (
        "<|im_start|>user\n"
        + body
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + "<think>\n\n</think>\n\n"
    )


def run_text(cli: Path, model: Path, extra: list[str], prompt: str, n: int,
             gpu: str = "small") -> tuple[str, str]:
    env = gpu_env.apply_gpu(os.environ.copy(), gpu)
    env.setdefault("KVMEM_TRACE", "1")
    cmd = [
        str(cli), "-m", str(model), "-n", str(n), "-ngl", "99", "--no-prompt",
        *extra, prompt,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed rc={proc.returncode}")
    gpu_env.require_device(proc.stderr, env["KVMEM_GPU_NAME"])
    keys = (
        "KVMEM_TRACE retrieval",
        "stage_in_raw",
        "stage_in_packed",
        "query_replay",
        "force_pos",
        "selected",
        "n_prompt=",
        "Device 0:",
        "needle_block",
        "KVMEM_TRACE occupy",
        "KVMEM_TRACE cells",
        "KVMEM_TRACE after_",
        "KVMEM_TRACE before_",
        "KVMEM_KV",
        "layout_orig_pos",
        "writeback",
        "KVMEM_TRACE mask last_pos",
        "gdn_ckpt",
        "gdn_restore",
        "recr_seq_pos",
        "recr_ckpt",
    )
    for ln in proc.stderr.splitlines():
        if any(k in ln for k in keys):
            print("  ", ln, file=sys.stderr)
    return proc.stdout, proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--method", choices=("recency", "retrieval"), default="recency")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--block-tokens", type=int, default=32)
    ap.add_argument("--gen-reserve", type=int, default=128)
    ap.add_argument("--query-last", type=int, default=64)
    ap.add_argument("--force-substr", default="")
    ap.add_argument("-n", "--n-predict", type=int, default=48)
    ap.add_argument("--gpu", choices=("small", "27b", "5050", "5090"), default="small",
                    help="small/5050 = RTX 5050; 27b/5090 = RTX 5090 (27B only)")
    ap.add_argument("--no-think", action="store_true",
                    help="Qwen chat wrap with empty <think></think> (disable reasoning)")
    args = ap.parse_args()

    extra = [
        "--kvmem",
        "--kvmem-block-tokens", str(args.block_tokens),
        "--kvmem-budget", str(args.budget),
        "--kvmem-gen-reserve", str(args.gen_reserve),
        "-c", "2048",
        "-b", "128",
    ]
    if args.method == "retrieval":
        extra += ["--kvmem-method", "retrieval", "--kvmem-query-last", str(args.query_last)]
        if args.force_substr:
            extra += ["--kvmem-force-substr", args.force_substr]
    else:
        extra += ["--kvmem-method", "recency"]

    out, _ = run_text(find_cli(), args.model, extra, make_prompt(no_think=args.no_think),
                      args.n_predict, gpu=args.gpu)
    print("output:", out[-500:])
    hit = "BLUEBIRD-42" in out
    tag = f"{args.method} budget={args.budget} bt={args.block_tokens}"
    if args.force_substr:
        tag += f" force={args.force_substr}"
    if args.method == "retrieval":
        if hit:
            print(f"PASS: {tag} revived the middle needle")
        else:
            print(f"NOTE: {tag} did not print BLUEBIRD-42")
            print("GO/NO-GO: retrieval needle miss at this budget — not an automatic stop")
        return 0
    if hit:
        print(f"FAIL: {tag} recalled the needle (window too large?)", file=sys.stderr)
        return 1
    if not out.strip():
        print("FAIL: empty output", file=sys.stderr)
        return 1
    print(f"PASS: {tag} did not recall the middle needle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
