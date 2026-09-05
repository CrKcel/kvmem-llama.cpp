#!/usr/bin/env python3
"""P7-0..P7-2: draft-mtp + lockstep pool + retrieval writeback on 0.8B / RTX 5050.

Exit:
  1. no --kvmem + --spec-type draft-mtp generates
  2. --kvmem recency (identity budget) + n_max=2 greedy; spec_verify TRACE
  3. identity without MTP still matches
  4. reject path (p_min=0) GDN restore does not crash
  5. MTP pool cells == budget+reserve, not -c
  6. recency 256 + MTP misses the middle needle
  7. retrieval 256 + MTP: ranker + mtp_follow writeback; query replay prefix
     non-empty. Generation BLUEBIRD-42 is GO/NO-GO, not an automatic fail.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_env  # noqa: E402
import needle_recall  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAIN_08B = ROOT / "models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
MTP_08B_CANDIDATES = [
    ROOT / "models/unsloth/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-Q8_0.gguf",
    ROOT / "models/unsloth/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf",
    ROOT / "models/unsloth/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
]
PROMPT = "Hello my name is"


def find_cli() -> Path:
    for p in (ROOT / "build/bin/llama-kvmem-cli", ROOT / "build/llama-kvmem-cli"):
        if p.is_file():
            return p
    raise SystemExit("llama-kvmem-cli not found; run scripts/build-cuda.sh")


def find_mtp_model() -> Path:
    for p in MTP_08B_CANDIDATES:
        if p.is_file():
            return p
    raise SystemExit(
        "no 0.8B MTP GGUF. The daily Qwen3.5-0.8B-Q8_0 has no nextn tensors.\n"
        "Download Unsloth MTP from ModelScope, e.g.\n"
        "  .venv/bin/python -c \"from modelscope import snapshot_download; "
        "snapshot_download('unsloth/Qwen3.5-0.8B-MTP-GGUF', "
        "allow_patterns=['*Q8_0.gguf','*Q4_K_M.gguf'], "
        "local_dir='models/unsloth/Qwen3.5-0.8B-MTP-GGUF')\""
    )


def run(cli: Path, model: Path, extra: list[str], gpu: str, prompt: str = PROMPT) -> tuple[int, str, str]:
    env = gpu_env.apply_gpu(os.environ.copy(), gpu)
    cmd = [
        str(cli), "-m", str(model), "-n", "32", "-ngl", "99",
        "--temp", "0", "--no-prompt", *extra, prompt,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=None, help="MTP-capable 0.8B GGUF")
    ap.add_argument("--gpu", choices=("small", "27b", "5050", "5090"), default="small")
    args = ap.parse_args()
    if args.gpu not in ("small", "5050"):
        raise SystemExit("P7-0 must run on RTX 5050 (--gpu small)")

    cli = find_cli()
    mtp_model = args.model if args.model else find_mtp_model()
    if not mtp_model.is_file():
        raise SystemExit(f"missing {mtp_model}")

    failures = 0

    if PLAIN_08B.is_file():
        print("== identity (no MTP) ==")
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/identity_canary.py"),
             "-m", str(PLAIN_08B), "--gpu", args.gpu],
            check=False,
        ).returncode
        if rc != 0:
            print("FAIL: identity without MTP")
            failures += 1
        else:
            print("PASS: identity without MTP")
    else:
        print("SKIP identity: missing", PLAIN_08B)

    print("== no-kvmem + draft-mtp ==")
    rc, out, err = run(cli, mtp_model, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], args.gpu)
    gpu_env.require_device(err, "RTX 5050")
    if rc != 0:
        sys.stderr.write(err)
        print("FAIL: no-kvmem MTP rc", rc)
        failures += 1
    elif "spec_start type=draft-mtp" not in err:
        sys.stderr.write(err)
        print("FAIL: no spec_start TRACE")
        failures += 1
    elif not out.strip():
        sys.stderr.write(err)
        print("FAIL: no-kvmem MTP produced no text")
        failures += 1
    else:
        print("PASS: no-kvmem MTP generated", repr(out.strip()[:80]))
        print(" ", [ln for ln in err.splitlines() if "spec_stats" in ln][-1:] or ["(no spec_stats)"])

    print("== kvmem recency + draft-mtp n_max=2 ==")
    extra = [
        "--kvmem", "--kvmem-method", "recency", "--kvmem-block-tokens", "32",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0",
    ]
    rc, out, err = run(cli, mtp_model, extra, args.gpu)
    gpu_env.require_device(err, "RTX 5050")
    if rc != 0:
        sys.stderr.write(err)
        print("FAIL: kvmem MTP rc", rc)
        failures += 1
    elif "spec_verify" not in err:
        sys.stderr.write(err)
        print("FAIL: missing spec_verify TRACE")
        failures += 1
    elif not out.strip():
        sys.stderr.write(err)
        print("FAIL: kvmem MTP produced no text")
        failures += 1
    else:
        print("PASS: kvmem MTP generated", repr(out.strip()[:80]))
        for ln in err.splitlines():
            if "spec_ckpt" in ln or "spec_stats" in ln:
                print(" ", ln)

    print("== P7-1 pool: MTP cells == target pool, not n_ctx ==")
    extra_pool = [
        "--kvmem", "--kvmem-method", "recency", "--kvmem-block-tokens", "32",
        "--kvmem-budget", "256", "--kvmem-gen-reserve", "256",
        "-c", "8192",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
    ]
    env = gpu_env.apply_gpu(os.environ.copy(), args.gpu)
    env["KVMEM_TRACE"] = "1"
    cmd = [
        str(cli), "-m", str(mtp_model), "-n", "16", "-ngl", "99",
        "--temp", "0", "--no-prompt", *extra_pool, PROMPT,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    gpu_env.require_device(proc.stderr, "RTX 5050")
    pool_lines = [ln for ln in proc.stderr.splitlines() if "mtp_pool" in ln]
    occupy_lines = [ln for ln in proc.stderr.splitlines() if "mtp_occupy" in ln]
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        print("FAIL: P7-1 pool run rc", proc.returncode)
        failures += 1
    elif not pool_lines:
        sys.stderr.write(proc.stderr[-2000:])
        print("FAIL: missing mtp_pool TRACE")
        failures += 1
    else:
        print(" ", pool_lines[0])
        if occupy_lines:
            print(" ", occupy_lines[0])
        # cells=512 (256+256), n_ctx=8192
        ok_cells = "cells=512" in pool_lines[0] and "target_cells=512" in pool_lines[0]
        ok_ctx = "n_ctx=8192" in pool_lines[0]
        if not ok_cells or not ok_ctx:
            print("FAIL: expected cells=512 target_cells=512 n_ctx=8192")
            failures += 1
        elif "n_ctx=512" in pool_lines[0].split("cells")[0]:
            print("FAIL: MTP pool looks like n_ctx")
            failures += 1
        else:
            print("PASS: MTP pool follows budget+reserve, not -c")

    print("== P7-2 recency 256 + MTP: middle needle should miss ==")
    needle = needle_recall.make_prompt(no_think=True)
    extra_rec = [
        "--kvmem", "--kvmem-method", "recency", "--kvmem-block-tokens", "32",
        "--kvmem-budget", "256", "--kvmem-gen-reserve", "256",
        "-c", "2048", "-b", "128",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
    ]
    env = gpu_env.apply_gpu(os.environ.copy(), args.gpu)
    env["KVMEM_TRACE"] = "1"
    cmd = [str(cli), "-m", str(mtp_model), "-n", "48", "-ngl", "99",
           "--temp", "0", "--no-prompt", *extra_rec, needle]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    gpu_env.require_device(proc.stderr, "RTX 5050")
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        print("FAIL: recency+MTP needle rc", proc.returncode)
        failures += 1
    elif "BLUEBIRD-42" in proc.stdout:
        print("FAIL: recency+MTP recalled BLUEBIRD-42")
        failures += 1
    else:
        print("PASS: recency+MTP did not recall the middle needle")

    print("== P7-2 retrieval 256 + MTP: ranker + optional generation ==")
    extra_ret = [
        "--kvmem", "--kvmem-method", "retrieval", "--kvmem-block-tokens", "32",
        "--kvmem-budget", "256", "--kvmem-gen-reserve", "256",
        "--kvmem-query-last", "64", "--kvmem-force-substr", "BLUEBIRD-42",
        "-c", "2048", "-b", "128",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
    ]
    cmd = [str(cli), "-m", str(mtp_model), "-n", "48", "-ngl", "99",
           "--temp", "0", "--no-prompt", *extra_ret, needle]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    gpu_env.require_device(proc.stderr, "RTX 5050")
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        print("FAIL: retrieval+MTP rc", proc.returncode)
        failures += 1
    else:
        sel = [ln for ln in proc.stderr.splitlines() if "KVMEM_TRACE selected" in ln]
        follow = [ln for ln in proc.stderr.splitlines() if "mtp_follow" in ln or "mtp_selected" in ln]
        replay = [ln for ln in proc.stderr.splitlines() if "mtp_after_query_replay" in ln]
        needle_blk = [ln for ln in proc.stderr.splitlines() if "needle_block" in ln]
        for ln in needle_blk[:1] + sel[:1] + follow[:2] + replay[:1]:
            print(" ", ln)
        if not sel:
            print("FAIL: missing retrieval selected TRACE")
            failures += 1
        elif not follow:
            print("FAIL: missing mtp_follow/mtp_selected TRACE")
            failures += 1
        else:
            follow_line = next((ln for ln in follow if "mtp_follow" in ln), follow[0])
            n_wb = n_miss = n_gpu = -1
            if "n_writeback=" in follow_line:
                try:
                    parts = {p.split("=")[0]: p.split("=")[1] for p in follow_line.split() if "=" in p}
                    n_wb = int(parts.get("n_writeback", "-1"))
                    n_miss = int(parts.get("n_no_raw", "-1"))
                    n_gpu = int(parts.get("n_gpu", "-1"))
                except ValueError:
                    pass
            if n_wb == 0:
                print("FAIL: mtp_follow wrote 0 blocks")
                failures += 1
            elif n_miss > 0:
                print("FAIL: mtp_follow n_no_raw=%d n_gpu=%d (MTP raw miss)" % (n_miss, n_gpu))
                failures += 1
            else:
                print("PASS: retrieval+MTP ranker/follow TRACE")
            replay_ok = False
            if replay:
                # seq_pos=[-1,-1] means the follower was emptied and not refilled.
                replay_ok = "seq_pos=[-1,-1]" not in replay[0] and "seq_pos=[" in replay[0]
            if not replay:
                print("FAIL: missing mtp_after_query_replay TRACE")
                failures += 1
            elif not replay_ok:
                print("FAIL: MTP prefix empty after query replay:", replay[0])
                failures += 1
            else:
                print("PASS: MTP prefix after query replay")
        if "BLUEBIRD-42" in proc.stdout:
            print("PASS: retrieval+MTP generated BLUEBIRD-42")
        else:
            print("GO/NO-GO: retrieval+MTP generation miss (ranker TRACE above)")
            print("output:", proc.stdout[-300:])

    print("== reject / GDN restore (p_min=0, n_max=2) ==")
    restores = [ln for ln in err.splitlines() if "spec_verify" in ln and "restore=1" in ln]
    rs = [ln for ln in err.splitlines() if "spec_ckpt tgt=RS" in ln]
    if rs:
        print("PASS: GDN n_rs_seq GPU planes", rs[0])
        if restores:
            print(" ", "host ckpt still used", len(restores), "times (draft longer than n_rs?)")
    elif "hybrid GDN" in err or "spec_ckpt tgt=PARTIAL_ONLY" in err:
        if restores:
            print("PASS: GDN restore hit", len(restores), "times")
        else:
            print("GO: hybrid GDN ckpt armed; this run had no partial reject (not a fail)")
    else:
        print("GO: not hybrid ckpt (dense or vanilla); restore count", len(restores))

    if failures:
        print(f"FAIL: {failures} P7-2 check(s)")
        return 1
    print("PASS: P7-2 mtp canary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
