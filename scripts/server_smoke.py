#!/usr/bin/env python3
"""P5 smoke: greedy + streaming chat completions, retrieval TRACE on last-user query."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
NEEDLE = "The secret code is BLUEBIRD-42."


def find_server() -> Path:
    for p in (ROOT / "build/bin/llama-kvmem-server", ROOT / "build/llama-kvmem-server"):
        if p.is_file():
            return p
    raise SystemExit("llama-kvmem-server not found; run scripts/build-cuda.sh")


def post_json(url: str, body: dict, timeout: int = 180) -> tuple[int, str]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def wait_health(base: str, timeout: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.3)
    raise SystemExit("server did not become healthy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=18181)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"missing model {args.model}")

    env = gpu_env.apply_gpu(os.environ.copy(), "small")
    env["KVMEM_TRACE"] = "1"
    log_path = ROOT / "logs" / "server_smoke.stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    cmd = [
        str(find_server()), "-m", str(args.model),
        "--host", args.host, "--port", str(args.port),
        "-c", "2048", "-b", "128", "-ngl", "99",
        "--kvmem", "--kvmem-budget", "256", "--kvmem-block-tokens", "32",
        "--kvmem-method", "retrieval",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=logf)
    base = f"http://{args.host}:{args.port}"
    try:
        t0 = time.time()
        while time.time() - t0 < 90:
            logf.flush()
            txt = log_path.read_text()
            if "llama-kvmem-server listening" in txt:
                break
            if proc.poll() is not None:
                raise SystemExit(f"server exited rc={proc.returncode}\n{txt[-4000:]}")
            time.sleep(0.3)
        else:
            raise SystemExit("server did not print listening line:\n" + log_path.read_text()[-4000:])
        wait_health(base)
        gpu_env.require_device(log_path.read_text(), "RTX 5050")

        st, body = post_json(base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hi in one word."}],
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
        })
        if st != 200:
            raise SystemExit(f"greedy chat failed {st}: {body}")
        greedy = json.loads(body)
        text = greedy["choices"][0]["message"]["content"]
        if not text.strip():
            raise SystemExit("greedy chat returned empty content")
        print("PASS: greedy /v1/chat/completions ->", text[:80].replace("\n", " "))

        st, body = post_json(base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Count to three."}],
            "max_tokens": 32,
            "temperature": 0,
            "stream": True,
        })
        if st != 200 or "data:" not in body:
            raise SystemExit(f"stream chat failed {st}: {body[:400]}")
        if "data: [DONE]" not in body:
            raise SystemExit("stream missing [DONE]")
        print("PASS: streaming /v1/chat/completions")

        filler = " lorem ipsum dolor sit amet" * 80
        st, body = post_json(base + "/v1/chat/completions", {
            "messages": [
                {"role": "system", "content": "Read the notes and answer the user."},
                {"role": "user", "content": filler + "\n" + NEEDLE + "\n" + filler},
                {"role": "user", "content": "What is the secret code?"},
            ],
            "max_tokens": 48,
            "temperature": 0,
            "stream": False,
        }, timeout=180)
        if st != 200:
            raise SystemExit(f"retrieval chat failed {st}: {body}")
        logf.flush()
        err = log_path.read_text()
        if "KVMEM_TRACE query_replay" not in err:
            raise SystemExit("missing KVMEM_TRACE query_replay in server log")
        if "KVMEM_TRACE retrieval" not in err and "KVMEM_TRACE selected" not in err:
            raise SystemExit("missing retrieval TRACE in server log")
        print("PASS: query-conditioned request printed retrieval TRACE")
        retr_text = json.loads(body)["choices"][0]["message"]["content"]
        print("retrieval output:", retr_text[:200].replace("\n", " "))
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()


if __name__ == "__main__":
    raise SystemExit(main())
