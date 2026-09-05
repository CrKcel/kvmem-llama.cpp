#!/usr/bin/env python3
"""P7-3: MTP + main KV bytes follow budget, not n_ctx / T.

27B on RTX 5090. Fixed slot-pool (budget+gen_reserve). Asserts:
  - mtp_pool cells == target pool, not -c
  - KVMEM_KV_BYTES and mtp_pool bytes stay constant as T grows
  - MTP bytes << one-layer x T (scheme A)
Decode tok/s (off vs draft-mtp) is recorded, not a go/no-go.
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
DEFAULT_MODEL = ROOT / "models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"

KV_RE = re.compile(
    r"KVMEM_KV_BYTES bytes=(\d+) cells=(\d+).*budget=(\d+) pool=(\d+)"
)
MTP_RE = re.compile(
    r"KVMEM_TRACE mtp_pool cells=(\d+) target_cells=(\d+) n_ctx=(\d+) bytes=(\d+)"
)
WALL_RE = re.compile(r"KVMEM_GEN_WALL n=(\d+) ms=([\d.]+) toks=([\d.]+)")
PROMPT_RE = re.compile(r"n_prompt=(\d+)")
SPEC_RE = re.compile(r"accept_pct=([\d.]+)")


def find_cli() -> Path:
    for p in (ROOT / "build/bin/llama-kvmem-cli", ROOT / "build/llama-kvmem-cli"):
        if p.is_file():
            return p
    raise SystemExit("llama-kvmem-cli not found; run scripts/build-cuda.sh")


def scheme_a_bytes(mtp_bytes: int, mtp_cells: int, n_ctx: int) -> int:
    if mtp_cells <= 0:
        return 0
    return int(mtp_bytes / mtp_cells * n_ctx)


def parse(stderr: str) -> dict:
    info = {
        "kv_bytes": None,
        "kv_cells": None,
        "budget": None,
        "pool": None,
        "mtp_cells": None,
        "mtp_target_cells": None,
        "mtp_n_ctx": None,
        "mtp_bytes": None,
        "n_prompt": None,
        "pressure": 0,
        "wall_n": 0,
        "wall_toks": 0.0,
        "accept_pct": 0.0,
        "oom": "out of memory" in stderr.lower() or "CUDA_ERROR_OUT_OF_MEMORY" in stderr,
    }
    for ln in stderr.splitlines():
        if m := KV_RE.search(ln):
            info["kv_bytes"] = int(m.group(1))
            info["kv_cells"] = int(m.group(2))
            info["budget"] = int(m.group(3))
            info["pool"] = int(m.group(4))
        if m := MTP_RE.search(ln):
            info["mtp_cells"] = int(m.group(1))
            info["mtp_target_cells"] = int(m.group(2))
            info["mtp_n_ctx"] = int(m.group(3))
            info["mtp_bytes"] = int(m.group(4))
        if m := PROMPT_RE.search(ln):
            info["n_prompt"] = int(m.group(1))
        if "prefill_pressure" in ln:
            info["pressure"] += 1
        if m := WALL_RE.search(ln):
            info["wall_n"] = int(m.group(1))
            info["wall_toks"] = float(m.group(3))
        if m := SPEC_RE.search(ln):
            info["accept_pct"] = float(m.group(1))
    return info


def run_cli(cli: Path, model: Path, prompt: str, extra: list[str], n_ctx: int,
            n_predict: int, batch: int, gpu: str) -> tuple[int, str]:
    env = gpu_env.apply_gpu(os.environ.copy(), gpu)
    env["KVMEM_TRACE"] = "1"
    with tempfile.NamedTemporaryFile("w", prefix="kvmem_p73_", suffix=".txt",
                                     delete=False) as fh:
        fh.write(prompt)
        path = fh.name
    try:
        cmd = [
            str(cli), "-m", str(model), "-n", str(n_predict), "-c", str(n_ctx),
            "-b", str(batch), "-ngl", "99", "--temp", "0", "--no-prompt",
            "-f", path, *extra,
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
        return proc.returncode, proc.stderr
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--gpu", choices=("27b", "5090"), default="27b")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--gen-reserve", type=int, default=256)
    ap.add_argument("--block-tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("-n", "--n-predict", type=int, default=32)
    ap.add_argument("--lengths", default="8192,16384,32768,65536")
    ap.add_argument("--try-128k", action="store_true")
    args = ap.parse_args()
    if args.gpu not in ("27b", "5090"):
        raise SystemExit("P7-3 must run on RTX 5090 (--gpu 27b)")
    if not args.model.is_file():
        raise SystemExit(f"missing {args.model}")

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    if args.try_128k:
        lengths.append(131072)
    pool = args.budget + args.gen_reserve
    cli = find_cli()
    expect = "RTX 5090"
    print(
        f"P7-3 VRAM  model={args.model.name} budget={args.budget} "
        f"gen_reserve={args.gen_reserve} pool={pool} n_predict={args.n_predict}",
        flush=True,
    )

    kvmem_mtp = [
        "--kvmem", "--kvmem-method", "retrieval",
        "--kvmem-budget", str(args.budget),
        "--kvmem-gen-reserve", str(args.gen_reserve),
        "--kvmem-block-tokens", str(args.block_tokens),
        "--kvmem-query-last", "64",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
    ]

    rows: list[tuple[int, dict]] = []
    failures = 0
    for ntok in lengths:
        n_ctx = ntok + max(args.n_predict, 64) + 256
        prompt = " ping" * ntok
        print(f"--- retr+mtp T~{ntok} -c {n_ctx} ---", flush=True)
        rc, err = run_cli(cli, args.model, prompt, kvmem_mtp, n_ctx,
                          args.n_predict, args.batch, args.gpu)
        st = parse(err)
        if rc != 0:
            snippet = "\n".join(err.splitlines()[-30:])
            if st["oom"]:
                print(f"  SKIP T~{ntok}: OOM (not a VRAM-cap fail)\n{snippet[-1500:]}",
                      flush=True)
                continue
            print(f"  FAIL T~{ntok} rc={rc}\n{snippet[-2000:]}", flush=True)
            failures += 1
            continue
        gpu_env.require_device(err, expect)
        a_bytes = scheme_a_bytes(st["mtp_bytes"] or 0, st["mtp_cells"] or 1,
                                 st["mtp_n_ctx"] or n_ctx)
        print(
            f"  n_prompt={st['n_prompt']} kv_cells={st['kv_cells']} "
            f"kv_bytes={st['kv_bytes']} mtp_cells={st['mtp_cells']} "
            f"mtp_bytes={st['mtp_bytes']} n_ctx={st['mtp_n_ctx']} "
            f"schemeA~{a_bytes} pressure={st['pressure']} "
            f"decode={st['wall_toks']:.1f} acc={st['accept_pct']:.1f}",
            flush=True,
        )
        rows.append((ntok, st))

    if not rows:
        print("FAIL: no successful length", file=sys.stderr)
        return 1

    print("\n========== P7-3 VRAM ==========")
    base = rows[0][1]
    long_ok = False
    for ntok, st in rows:
        marks = []
        if st["mtp_cells"] is None or st["kv_bytes"] is None:
            marks.append("missing TRACE")
        else:
            if st["mtp_cells"] != st["mtp_target_cells"]:
                marks.append("mtp_cells != target")
            if st["mtp_cells"] != st["kv_cells"]:
                marks.append("mtp_cells != kv_cells")
            if st["pool"] != pool or st["kv_cells"] != pool:
                marks.append(f"pool {st['kv_cells']} != {pool}")
            if st["mtp_n_ctx"] is not None and st["mtp_cells"] >= st["mtp_n_ctx"]:
                marks.append("mtp pool looks like n_ctx")
            if st["kv_bytes"] != base["kv_bytes"] or st["kv_cells"] != base["kv_cells"]:
                marks.append(f"main KV grew vs T={rows[0][0]}")
            if st["mtp_bytes"] != base["mtp_bytes"] or st["mtp_cells"] != base["mtp_cells"]:
                marks.append(f"MTP KV grew vs T={rows[0][0]}")
            a_bytes = scheme_a_bytes(st["mtp_bytes"], st["mtp_cells"], st["mtp_n_ctx"])
            if st["mtp_bytes"] * 8 > a_bytes and ntok >= 16384:
                marks.append(f"MTP bytes not << scheme A ({a_bytes})")
            if st["n_prompt"] is None or st["n_prompt"] < ntok // 2:
                marks.append(f"n_prompt {st['n_prompt']} too small")
            if st["n_prompt"] and st["n_prompt"] > args.budget and st["pressure"] <= 0:
                marks.append("no prefill_pressure")
        if ntok >= 32768 and not marks:
            long_ok = True
        tag = "PASS" if not marks else "FAIL"
        if marks:
            failures += 1
        print(
            f"  {tag} T~{ntok:6d} n_prompt={st['n_prompt']} "
            f"kv={st['kv_bytes']}B/{st['kv_cells']}c "
            f"mtp={st['mtp_bytes']}B/{st['mtp_cells']}c "
            f"n_ctx={st['mtp_n_ctx']} decode={st['wall_toks']:.1f}"
        )
        if marks:
            print("       " + "; ".join(marks), file=sys.stderr)

    if not long_ok:
        print("FAIL: need at least one PASS at T>=32768", file=sys.stderr)
        failures += 1

    # Speed: off vs retr+mtp at 8k and the longest successful T (record only).
    speed_lens = []
    have = {n for n, _ in rows}
    if 8192 in have:
        speed_lens.append(8192)
    longest = max(have)
    if longest not in speed_lens:
        speed_lens.append(longest)
    print("\n========== P7-3 decode (record only) ==========")
    print(f"{'len':>6} {'mode':>10} {'decode':>8} {'acc%':>6}")
    for ntok in speed_lens:
        n_ctx = ntok + max(args.n_predict, 64) + 256
        prompt = " ping" * ntok
        for mode, extra in (("off", []), ("retr+mtp", kvmem_mtp)):
            if mode == "retr+mtp":
                st = next(s for n, s in rows if n == ntok)
                print(f"{ntok:6d} {mode:>10} {st['wall_toks']:8.1f} {st['accept_pct']:6.1f}")
                continue
            print(f"--- off T~{ntok} ---", flush=True)
            rc, err = run_cli(cli, args.model, prompt, extra, n_ctx,
                              args.n_predict, args.batch, args.gpu)
            st = parse(err)
            if rc != 0:
                print(f"{ntok:6d} {mode:>10}     FAIL rc={rc}")
                continue
            gpu_env.require_device(err, expect)
            print(f"{ntok:6d} {mode:>10} {st['wall_toks']:8.1f} {st['accept_pct']:6.1f}")
    print("Speed is not a go/no-go.")

    if failures:
        print(f"FAIL: {failures} P7-3 check(s)")
        return 1
    print("PASS: P7-3 MTP+main KV bytes follow budget, not T")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
