#!/usr/bin/env python3
"""Prefill tok/s vs prompt length: off / recency / capture-only / retrieval.

Default GPU is the 5050 (`--gpu small`). 27B must use `--gpu 27b` (5090).
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
# MTP verify is n>1 so llama_perf gen_toks land in prompt-eval. Wall clock
# of emitted tokens is the comparable decode number.
WALL_RE = re.compile(
    r"KVMEM_GEN_WALL n=(?P<n>\d+) ms=(?P<ms>[\d.]+) toks=(?P<toks>[\d.]+)"
)
SPEC_RE = re.compile(
    r"KVMEM_TRACE spec_stats n_gen=(?P<n_gen>\d+) n_drafted=(?P<n_drafted>\d+) "
    r"n_accept=(?P<n_accept>\d+) n_restore=(?P<n_restore>\d+) "
    r"accept_pct=(?P<accept_pct>[\d.]+)"
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
             n_predict: int, gpu: str = "small",
             n_batch: int = 256, n_ubatch: int = 0) -> dict:
    env = gpu_env.apply_gpu(os.environ.copy(), gpu)
    env["KVMEM_TRACE"] = "1"  # Preserve the parsed MTP statistics; includes tracing overhead.
    with tempfile.NamedTemporaryFile("w", prefix="kvmem_prefill_", suffix=".txt",
                                     delete=False) as fh:
        fh.write(prompt)
        path = fh.name
    try:
        ub = n_ubatch if n_ubatch > 0 else n_batch
        cmd = [
            str(cli), "-m", str(model), "-n", str(n_predict), "-c", str(n_ctx),
            "-b", str(n_batch), "-ub", str(ub), "-ngl", "99", "--temp", "0",
            "--no-prompt", "-f", path, *extra,
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
    gpu_env.require_device(proc.stderr, env["KVMEM_GPU_NAME"])
    pm = PERF_RE.search(proc.stderr)
    sm = STAGE_RE.search(proc.stderr)
    wm = WALL_RE.search(proc.stderr)
    sp = SPEC_RE.search(proc.stderr)
    if not pm or not sm:
        raise RuntimeError("missing KVMEM_PERF/KVMEM_STAGE in stderr")
    wall_n = int(wm.group("n")) if wm else int(pm.group("gen_n") or 0)
    wall_ms = float(wm.group("ms")) if wm else float(pm.group("gen_ms") or 0.0)
    wall_toks = float(wm.group("toks")) if wm else float(pm.group("gen_toks") or 0.0)
    return {
        "prompt_n": int(pm.group("prompt_n")),
        "prompt_ms": float(pm.group("prompt_ms")),
        "prompt_toks": float(pm.group("prompt_toks")),
        "gen_n": int(pm.group("gen_n") or 0),
        "gen_ms": float(pm.group("gen_ms") or 0.0),
        "gen_toks": float(pm.group("gen_toks") or 0.0),
        "wall_n": wall_n,
        "wall_ms": wall_ms,
        "wall_toks": wall_toks,
        "prefill_n": int(sm.group("prefill_n")),
        "prefill_ms": float(sm.group("prefill_ms")),
        "prefill_toks": float(sm.group("prefill_toks")),
        "retrieval_ms": float(sm.group("retrieval_ms")),
        "replay_n": int(sm.group("replay_n")),
        "replay_eval_ms": float(sm.group("replay_eval_ms")),
        "replay_wall_ms": float(sm.group("replay_wall_ms")),
        "spec_n_gen": float(sp.group("n_gen")) if sp else 0.0,
        "spec_n_drafted": float(sp.group("n_drafted")) if sp else 0.0,
        "spec_n_accept": float(sp.group("n_accept")) if sp else 0.0,
        "spec_accept_pct": float(sp.group("accept_pct")) if sp else 0.0,
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


def extra_for(mode: str, spec_n_max: int = 2, budget: int = 0,
              gen_reserve: int = 0) -> list[str]:
    mtp = mode.endswith("+mtp")
    base = mode[:-4] if mtp else mode
    extra: list[str] = []
    if base == "off":
        pass
    elif base in ("recency", "rec"):
        extra += ["--kvmem", "--kvmem-method", "recency"]
    elif base == "cap":
        extra += ["--kvmem", "--kvmem-query-last", "0"]
    elif base in ("retr", "retrieval"):
        extra += ["--kvmem"]  # default retrieval + query-last 64
    else:
        raise ValueError(mode)
    if extra and budget:
        extra += ["--kvmem-budget", str(budget)]
    if extra and gen_reserve:
        extra += ["--kvmem-gen-reserve", str(gen_reserve)]
    if mtp:
        extra += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(spec_n_max)]
    return extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--lengths", default="256,512,1024,2048,4096")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("-n", "--n-predict", type=int, default=1,
                    help="generated tokens (1 = prefill-only; 64 for decode sweep)")
    ap.add_argument("--modes", default="off,recency,cap,retr",
                    help="comma list: off,recency,cap,retr and +mtp variants "
                         "(off+mtp, rec+mtp, retr+mtp)")
    ap.add_argument("--spec-n-max", type=int, default=2,
                    help="--spec-draft-n-max for +mtp modes")
    ap.add_argument("--budget", type=int, default=0,
                    help="--kvmem-budget for kvmem modes; 0 = CLI default (n_ctx)")
    ap.add_argument("--gen-reserve", type=int, default=0,
                    help="--kvmem-gen-reserve for kvmem modes; 0 = CLI default")
    ap.add_argument("--gpu", choices=("small", "27b", "5050", "5090"), default="small",
                    help="small/5050 = RTX 5050; 27b/5090 = RTX 5090 (27B only)")
    ap.add_argument("-b", "--batch", type=int, default=256,
                    help="llama.cpp -b (n_batch). 27B comparisons should use 512.")
    ap.add_argument("-ub", "--ubatch", type=int, default=0,
                    help="llama.cpp -ub (n_ubatch). 0 = same as --batch.")
    args = ap.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"missing model {args.model}")
    cli = find_cli()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    modes = tuple(x.strip() for x in args.modes.split(",") if x.strip())
    for mode in modes:
        extra_for(mode, args.spec_n_max, args.budget, args.gen_reserve)

    gpu_name = "RTX 5090" if args.gpu in ("27b", "5090") else "RTX 5050"
    print(f"model={args.model.name} warmup={args.warmup} runs={args.runs} "
          f"n_predict={args.n_predict} device={gpu_name} spec_n_max={args.spec_n_max} "
          f"batch={args.batch} ubatch={args.ubatch or args.batch}")
    print("modes: off | recency | cap | retr | off+mtp | rec+mtp | retr+mtp")
    print("decode = KVMEM_GEN_WALL (emitted tokens / wall); MTP verify is n>1 so "
          "llama_perf gen_toks is not comparable.")
    print()
    header = (
        f"{'len':>5} {'mode':>8} {'prefill_n':>9} {'prefill':>10} "
        f"{'ms':>8} {'px':>6} {'retr_ms':>8} {'rep_n':>5} {'rep_ms':>8} "
        f"{'tot_n':>6} {'tot_tok/s':>10}"
    )
    if args.n_predict > 1:
        header += f" {'wall_n':>6} {'decode':>8} {'dx':>6} {'acc%':>6}"
    print(header)
    print("-" * len(header))

    # results[length][mode] = mean dict
    results: dict[int, dict[str, dict]] = {}
    for target in lengths:
        prompt = make_prompt(target)
        n_ctx = max(target + max(args.n_predict, 64) + 256, 1024)
        results[target] = {}
        for mode in modes:
            extra = extra_for(mode, args.spec_n_max, args.budget, args.gen_reserve)
            try:
                for _ in range(args.warmup):
                    run_once(cli, args.model, extra, prompt, n_ctx, args.n_predict,
                             gpu=args.gpu, n_batch=args.batch, n_ubatch=args.ubatch)
                rows = [
                    run_once(cli, args.model, extra, prompt, n_ctx, args.n_predict,
                             gpu=args.gpu, n_batch=args.batch, n_ubatch=args.ubatch)
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
                off_dx = off["wall_toks"] if off else avg["wall_toks"]
                dx = (off_dx / avg["wall_toks"]) if avg["wall_toks"] > 0 else 0.0
                line += (
                    f" {avg['wall_n']:6.0f} {avg['wall_toks']:8.1f} {dx:6.2f} "
                    f"{avg['spec_accept_pct']:6.1f}"
                )
            print(line, flush=True)
        print(flush=True)

    print("px = off_prefill_toks / mode_prefill_toks  (1.0 = same as stock; <1 = faster)")
    print("dx = off_wall_toks / mode_wall_toks        (1.0 = same as stock; <1 = faster)")
    print("decode uses KVMEM_GEN_WALL. retr tot_tok/s includes query-replay in llama prompt eval.")
    print("Speed is not a go/no-go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
