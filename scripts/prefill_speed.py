#!/usr/bin/env python3
"""Prefill tok/s vs prompt length: off / recency / capture-only / retrieval.

RTX 5050 only. Prints a table and a short factor breakdown.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
UNIT = "The quick brown fox jumps over the lazy dog. "
PERF_RE = re.compile(
    r"KVMEM_PERF load_ms=(?P<load_ms>[\d.]+) prompt_n=(?P<prompt_n>\d+) "
    r"prompt_ms=(?P<prompt_ms>[\d.]+) prompt_toks=(?P<prompt_toks>[\d.]+)"
    r"(?: gen_n=(?P<gen_n>\d+) gen_ms=(?P<gen_ms>[\d.]+) gen_toks=(?P<gen_toks>[\d.]+))?"
)
STAGE_RE = re.compile(
    r"KVMEM_STAGE prefill_n=(?P<prefill_n>\d+) prefill_ms=(?P<prefill_ms>[\d.]+) "
    r"prefill_toks=(?P<prefill_toks>[\d.]+) retrieval_ms=(?P<retrieval_ms>[\d.]+) "
    r"replay_n=(?P<replay_n>\d+) replay_eval_ms=(?P<replay_eval_ms>[\d.]+) "
    r"replay_wall_ms=(?P<replay_wall_ms>[\d.]+)"
)


def find_cli() -> Path:
    for p in (
        ROOT / "build/bin/llama-kvmem-cli",
        ROOT / "build/llama-kvmem-cli",
    ):
        if p.is_file():
            return p
    raise SystemExit("llama-kvmem-cli not found; run scripts/build-cuda.sh")


def run_once(cli: Path, model: Path, extra: list[str], prompt: str, n_ctx: int,
             n_predict: int) -> dict:
    env = gpu_env.apply_gpu(os.environ.copy(), "small")
    env.pop("KVMEM_TRACE", None)
    with tempfile.NamedTemporaryFile("w", prefix="kvmem_prefill_", suffix=".txt",
                                     delete=False) as fh:
        fh.write(prompt)
        path = fh.name
    try:
        cmd = [
            str(cli), "-m", str(model), "-n", str(n_predict), "-c", str(n_ctx),
            "-b", "256", "-ngl", "99", "--no-prompt", "-f", path, *extra,
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(f"command failed rc={proc.returncode}")
    gpu_env.require_device(proc.stderr, "RTX 5050")
    pm = PERF_RE.search(proc.stderr)
    sm = STAGE_RE.search(proc.stderr)
    if not pm or not sm:
        raise RuntimeError("missing KVMEM_PERF/KVMEM_STAGE in stderr")
    return {
        "prompt_n": int(pm.group("prompt_n")),
        "prompt_ms": float(pm.group("prompt_ms")),
        "prompt_toks": float(pm.group("prompt_toks")),
        "gen_n": int(pm.group("gen_n") or 0),
        "gen_ms": float(pm.group("gen_ms") or 0.0),
        "gen_toks": float(pm.group("gen_toks") or 0.0),
        "prefill_n": int(sm.group("prefill_n")),
        "prefill_ms": float(sm.group("prefill_ms")),
        "prefill_toks": float(sm.group("prefill_toks")),
        "retrieval_ms": float(sm.group("retrieval_ms")),
        "replay_n": int(sm.group("replay_n")),
        "replay_eval_ms": float(sm.group("replay_eval_ms")),
        "replay_wall_ms": float(sm.group("replay_wall_ms")),
    }


def mean_rows(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        out[k] = statistics.fmean(r[k] for r in rows)
    return out


def make_prompt(target: int) -> str:
    # ~10 tokens / UNIT (speed canary: 485 tok / 48 repeats). Stay under ctx.
    n = max(1, target // 10)
    return UNIT * n + "Write a short continuation:"


def extra_for(mode: str) -> list[str]:
    if mode == "off":
        return []
    if mode == "recency":
        return ["--kvmem", "--kvmem-method", "recency"]
    if mode == "cap":
        # retrieval capture during prefill, no query replay / score / writeback
        return ["--kvmem", "--kvmem-query-last", "0"]
    if mode == "retr":
        return ["--kvmem"]  # default retrieval + query-last 64
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--lengths", default="256,512,1024,2048,4096")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("-n", "--n-predict", type=int, default=1,
                    help="generated tokens (1 = prefill-only; 64 for decode sweep)")
    ap.add_argument("--modes", default="off,recency,cap,retr",
                    help="comma list: off,recency,cap,retr")
    args = ap.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"missing model {args.model}")
    cli = find_cli()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    modes = tuple(x.strip() for x in args.modes.split(",") if x.strip())
    for mode in modes:
        extra_for(mode)

    print(f"model={args.model.name} warmup={args.warmup} runs={args.runs} "
          f"n_predict={args.n_predict} device=RTX 5050")
    print("modes: off | recency (no capture) | cap (retrieval capture, no replay) | "
          "retr (capture + score/writeback + query-last 64)")
    print()
    header = (
        f"{'len':>5} {'mode':>8} {'prefill_n':>9} {'prefill':>10} "
        f"{'ms':>8} {'px':>6} {'retr_ms':>8} {'rep_n':>5} {'rep_ms':>8} "
        f"{'tot_n':>6} {'tot_tok/s':>10}"
    )
    if args.n_predict > 1:
        header += f" {'gen_n':>5} {'decode':>8} {'dx':>6}"
    print(header)
    print("-" * len(header))

    # results[length][mode] = mean dict
    results: dict[int, dict[str, dict]] = {}
    for target in lengths:
        prompt = make_prompt(target)
        n_ctx = max(target + max(args.n_predict, 64) + 256, 1024)
        results[target] = {}
        for mode in modes:
            extra = extra_for(mode)
            try:
                for _ in range(args.warmup):
                    run_once(cli, args.model, extra, prompt, n_ctx, args.n_predict)
                rows = [
                    run_once(cli, args.model, extra, prompt, n_ctx, args.n_predict)
                    for _ in range(args.runs)
                ]
            except RuntimeError as exc:
                print(f"{target:5d} {mode:>8}  FAIL {exc}", flush=True)
                continue
            avg = mean_rows(rows)
            results[target][mode] = avg
            off = results[target].get("off")
            off_tps = off["prefill_toks"] if off else avg["prefill_toks"]
            px = (off_tps / avg["prefill_toks"]) if avg["prefill_toks"] > 0 else 0.0
            line = (
                f"{target:5d} {mode:>8} {avg['prefill_n']:9.0f} {avg['prefill_toks']:10.1f} "
                f"{avg['prefill_ms']:8.1f} {px:6.2f} {avg['retrieval_ms']:8.1f} "
                f"{avg['replay_n']:5.0f} {avg['replay_eval_ms']:8.1f} "
                f"{avg['prompt_n']:6.0f} {avg['prompt_toks']:10.1f}"
            )
            if args.n_predict > 1:
                off_dx = off["gen_toks"] if off else avg["gen_toks"]
                dx = (off_dx / avg["gen_toks"]) if avg["gen_toks"] > 0 else 0.0
                line += f" {avg['gen_n']:5.0f} {avg['gen_toks']:8.1f} {dx:6.2f}"
            print(line, flush=True)
        print(flush=True)

    print("px = off_prefill_toks / mode_prefill_toks  (1.0 = same as stock)")
    print("dx = off_decode_toks / mode_decode_toks    (1.0 = same as stock)")
    print("retr tot_tok/s includes query-replay tokens in llama prompt eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
