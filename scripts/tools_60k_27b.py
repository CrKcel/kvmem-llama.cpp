#!/usr/bin/env python3
"""27B / 5090: ~60k notes + multi-turn tool calls. Record speed, VRAM, RSS.

Recipe matches the 60k programming canary:
  -c 65536 -b 512 --kv-dtype q8_0
  --kvmem-budget 20000 --kvmem-gen-reserve 10000 --kvmem-block-tokens 128
  --spec-type none
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"
PROG_SRC = Path("/tmp/kvmem_prog_60k.txt")
NEEDLE = "The secret code is BLUEBIRD-42."
GPU_UUID = gpu_env.UUID_5090

TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup_code",
        "description": "Look up a secret code from the project registry",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
}]


def find_server() -> Path:
    for p in (ROOT / "build/bin/llama-kvmem-server", ROOT / "build/llama-kvmem-server"):
        if p.is_file():
            return p
    raise SystemExit("llama-kvmem-server not found; run scripts/build-cuda.sh")


def load_notes() -> str:
    if not PROG_SRC.is_file():
        raise SystemExit(f"missing {PROG_SRC}")
    body = PROG_SRC.read_text()
    if "<|im_start|>user" in body:
        body = body.split("<|im_start|>user\n", 1)[1]
        body = body.split("<|im_end|>", 1)[0]
    if "\nRequirements:" in body:
        body = body.split("\nRequirements:", 1)[0]
    body = body.strip()
    mid = len(body) // 2
    nl = body.find("\n", mid)
    if nl < 0:
        nl = mid
    return body[:nl] + "\n\n" + NEEDLE + "\n\n" + body[nl:]


def post_json(url: str, body: dict, timeout: int) -> tuple[int, str]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def ival(s: str, key: str) -> int:
    m = re.search(rf"{re.escape(key)}=(-?\d+)", s)
    return int(m.group(1)) if m else -1


def fval(s: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)}=(-?[\d.]+)", s)
    return float(m.group(1)) if m else float("nan")


def qspan(s: str) -> tuple[int, int]:
    m = re.search(r"query=\[(-?\d+),(-?\d+)\)", s)
    if not m:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def snapshot_mem(pid: int) -> dict:
    gpu_mib = -1
    gpu_util = -1
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for ln in out.splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 3 and GPU_UUID in parts[0]:
                gpu_mib = int(float(parts[1]))
                gpu_util = int(float(parts[2]))
    except Exception:
        pass
    rss_kib = hwm_kib = -1
    try:
        st = Path(f"/proc/{pid}/status").read_text()
        for ln in st.splitlines():
            if ln.startswith("VmRSS:"):
                rss_kib = int(ln.split()[1])
            elif ln.startswith("VmHWM:"):
                hwm_kib = int(ln.split()[1])
    except Exception:
        pass
    return {
        "ts": time.time(),
        "gpu_mib": gpu_mib,
        "gpu_util": gpu_util,
        "rss_kib": rss_kib,
        "hwm_kib": hwm_kib,
    }


class MemSampler(threading.Thread):
    def __init__(self, pid: int, path: Path, interval: float = 0.5):
        super().__init__(daemon=True)
        self.pid = pid
        self.path = path
        self.interval = interval
        self.stop_ev = threading.Event()
        self.rows: list[dict] = []
        self.peak = {"gpu_mib": 0, "rss_kib": 0, "hwm_kib": 0}

    def run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as fh:
            fh.write("ts_epoch,gpu_mib,gpu_util,rss_kib,hwm_kib\n")
            while not self.stop_ev.is_set():
                row = snapshot_mem(self.pid)
                self.rows.append(row)
                for k in ("gpu_mib", "rss_kib", "hwm_kib"):
                    if row[k] > self.peak[k]:
                        self.peak[k] = row[k]
                fh.write(
                    f"{row['ts']:.6f},{row['gpu_mib']},{row['gpu_util']},"
                    f"{row['rss_kib']},{row['hwm_kib']}\n"
                )
                fh.flush()
                self.stop_ev.wait(self.interval)

    def stop(self) -> None:
        self.stop_ev.set()
        self.join(timeout=3)


def last_line(chunk: str, needle: str) -> str:
    hits = [ln for ln in chunk.splitlines() if needle in ln]
    return hits[-1] if hits else ""


def parse_turn_log(chunk: str) -> dict:
    reuse = last_line(chunk, "prefix_reuse")
    wall = last_line(chunk, "KVMEM_GEN_WALL")
    turn = last_line(chunk, "KVMEM_CHAT_TURN")
    pf = last_line(chunk, "KVMEM_CHAT_PREFILL")
    retr = last_line(chunk, "KVMEM_RETR_SUM")
    npr = last_line(chunk, "KVMEM_TRACE n_prompt=")
    qcap = last_line(chunk, "query_q_capture")
    qrep = last_line(chunk, "query_replay")
    prot = last_line(chunk, "retrieval_protect")
    spec = last_line(chunk, "spec_stats")
    fallback = last_line(chunk, "greedy fallback")
    q0, q1 = qspan(reuse or npr)
    used_mtp = bool(spec) and not fallback
    return {
        "reuse_line": reuse,
        "reused": ival(reuse, "reused") if reuse else 0,
        "n_past": ival(reuse, "n_past") if reuse else -1,
        "n_prompt": ival(reuse or npr or pf, "n_prompt"),
        "n_new": ival(reuse, "n_new") if reuse else -1,
        "query": [q0, q1],
        "prefill_ms": fval(pf or turn, "ms" if pf else "prefill_ms"),
        "gen_n": ival(wall or turn, "n" if wall else "n_gen"),
        "gen_ms": fval(wall, "ms") if wall else fval(turn, "gen_ms"),
        "gen_toks": fval(wall, "toks") if wall else fval(turn, "gen_toks"),
        "turn_wall_ms": fval(turn, "wall_ms"),
        "retrieval_ms": fval(retr, "total_ms"),
        "query_q_capture": qcap,
        "query_replay": qrep,
        "retrieval_protect": prot,
        "spec_line": spec,
        "spec_fallback": fallback,
        "used_mtp": used_mtp,
        "n_drafted": ival(spec, "n_drafted") if spec else 0,
        "n_accept": ival(spec, "n_accept") if spec else 0,
        "accept_pct": fval(spec, "accept_pct") if spec else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=18191)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--spec-type", default="none", choices=("none", "draft-mtp"))
    ap.add_argument("--spec-draft-n-max", type=int, default=2)
    ap.add_argument("--log-tag", default="", help="suffix for log files, e.g. mtp")
    args = ap.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"missing model {args.model}")

    notes = load_notes()
    (ROOT / "logs" / "kvmem_prog_60k_notes.txt").write_text(notes)

    env = gpu_env.apply_gpu(os.environ.copy(), "27b")
    env["KVMEM_TRACE"] = "1"
    log_tag = args.log_tag or ("mtp" if args.spec_type == "draft-mtp" else "")
    stem = "tools_60k_27b" + (f"_{log_tag}" if log_tag else "")
    log_path = ROOT / "logs" / f"{stem}.stderr.log"
    mem_path = ROOT / "logs" / f"{stem}_mem.csv"
    sum_path = ROOT / "logs" / f"{stem}.summary"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    cmd = [
        str(find_server()), "--verbosity", "4", "-m", str(args.model),
        "--host", args.host, "--port", str(args.port),
        "-c", "65536", "-n", "256", "-b", "512", "-ngl", "99",
        "--kvmem", "--kvmem-method", "retrieval",
        "--kvmem-budget", "20000", "--kvmem-gen-reserve", "10000",
        "--kvmem-block-tokens", "128", "--kv-dtype", "q8_0",
        "--spec-type", args.spec_type,
    ]
    if args.spec_type == "draft-mtp":
        cmd += ["--spec-draft-n-max", str(args.spec_draft_n_max)]
    print("cmd:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=logf)
    sampler = MemSampler(proc.pid, mem_path)
    sampler.start()
    base = f"http://{args.host}:{args.port}"
    turns: list[dict] = []
    try:
        t0 = time.time()
        while time.time() - t0 < 180:
            logf.flush()
            txt = log_path.read_text()
            if "llama-kvmem-server listening" in txt:
                break
            if proc.poll() is not None:
                raise SystemExit(f"server exited rc={proc.returncode}\n{txt[-4000:]}")
            time.sleep(0.5)
        else:
            raise SystemExit("server did not print listening line:\n" + log_path.read_text()[-4000:])
        gpu_env.require_device(log_path.read_text(), "RTX 5090")
        print(f"server up in {time.time() - t0:.1f}s  idle_mem={snapshot_mem(proc.pid)}", flush=True)

        t1_user = (
            "Look up the secret code using the lookup_code tool. "
            "After the tool returns, reply with only the code."
        )
        messages: list[dict] = [
            {
                "role": "system",
                "content": "Read the project notes. Use tools when asked to look up a code. "
                           "Do not invent codes.",
            },
            {"role": "user", "content": notes},
            {"role": "user", "content": t1_user},
        ]

        def chat(body: dict, tag: str, timeout: int = 1800) -> dict:
            mark = log_path.read_text()
            print(f"--- {tag} POST ---", flush=True)
            tw = time.time()
            st, raw = post_json(base + "/v1/chat/completions", body, timeout=timeout)
            http_s = time.time() - tw
            logf.flush()
            chunk = log_path.read_text()[len(mark):]
            info = parse_turn_log(chunk)
            info["tag"] = tag
            info["http_s"] = http_s
            info["http_status"] = st
            info["mem"] = snapshot_mem(proc.pid)
            if st != 200:
                info["error"] = raw[:800]
                print(f"{tag} HTTP {st} wall={http_s:.1f}s body={raw[:400]}", flush=True)
                return info
            parsed = json.loads(raw)
            ch0 = (parsed.get("choices") or [{}])[0]
            msg = ch0.get("message") or {}
            info["finish_reason"] = ch0.get("finish_reason")
            info["content"] = msg.get("content") or ""
            info["tool_calls"] = msg.get("tool_calls") or []
            info["usage"] = parsed.get("usage") or {}
            info["message"] = msg
            mtp_s = (
                f" mtp={int(info['used_mtp'])} acc={info['accept_pct']:.1f}%"
                if info.get("used_mtp") else
                (" mtp=0 fallback" if info.get("spec_fallback") else " mtp=0")
            )
            print(
                f"{tag}: http={http_s:.1f}s n_prompt={info['n_prompt']} reused={info['reused']} "
                f"n_past={info['n_past']} n_new={info['n_new']} query={info['query']} "
                f"prefill_ms={info['prefill_ms']:.0f} gen_n={info['gen_n']} "
                f"gen_toks={info['gen_toks']:.2f} retr_ms={info['retrieval_ms']:.1f} "
                f"gpu={info['mem']['gpu_mib']}MiB rss={info['mem']['rss_kib']/1024:.0f}MiB "
                f"finish={info['finish_reason']}{mtp_s}",
                flush=True,
            )
            content_one = (info["content"] or "").replace("\n", " ")[:180]
            print(f"  content: {content_one}", flush=True)
            if info["tool_calls"]:
                print(f"  tool_calls: {json.dumps(info['tool_calls'])[:400]}", flush=True)
            return info

        t1 = chat({
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "required",
            "max_tokens": 96,
            "temperature": 0,
            "stream": False,
            "kvmem": {"enable_thinking": False},
        }, "turn1-tool")
        turns.append(t1)
        if t1.get("http_status") != 200:
            raise SystemExit("turn 1 failed")
        if t1.get("finish_reason") != "tool_calls" or not t1.get("tool_calls"):
            raise SystemExit(f"turn 1 expected tool_calls, got {t1.get('finish_reason')}")
        call = t1["tool_calls"][0]
        call_id = call.get("id") or "call_1"

        messages.append(t1["message"])
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": "BLUEBIRD-42",
        })
        t2 = chat({
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "none",
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
            "kvmem": {"enable_thinking": False},
        }, "turn2-tool-result")
        turns.append(t2)
        if t2.get("http_status") != 200:
            raise SystemExit("turn 2 failed")

        messages.append({
            "role": "assistant",
            "content": t2.get("content") or "",
        })
        messages.append({
            "role": "user",
            "content": "What was the secret code written in the project notes? Reply with the code only.",
        })
        t3 = chat({
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "none",
            "max_tokens": 48,
            "temperature": 0,
            "stream": False,
            "kvmem": {"enable_thinking": False},
        }, "turn3-notes-needle")
        turns.append(t3)
        if t3.get("http_status") != 200:
            raise SystemExit("turn 3 failed")

        peak = sampler.peak
        lines = [
            f"model={args.model.name}",
            f"device=RTX 5090  recipe=-c 65536 budget=20000 gen_reserve=10000 bt=128 b=512 q8_0 spec={args.spec_type} n_max={args.spec_draft_n_max}",
            f"PEAK gpu_mib={peak['gpu_mib']} rss_mib={peak['rss_kib']/1024:.1f} hwm_mib={peak['hwm_kib']/1024:.1f}",
            "",
        ]
        for t in turns:
            mtp_s = (
                f" mtp=1 drafted={t.get('n_drafted')} accept={t.get('n_accept')} "
                f"accept_pct={t.get('accept_pct'):.1f}"
                if t.get("used_mtp") else
                (" mtp=0 greedy-fallback" if t.get("spec_fallback") else " mtp=0")
            )
            lines.append(
                f"{t['tag']}: http_s={t['http_s']:.2f} n_prompt={t['n_prompt']} reused={t['reused']} "
                f"n_past={t['n_past']} n_new={t['n_new']} query={t['query']} "
                f"prefill_ms={t['prefill_ms']:.1f} gen_n={t['gen_n']} gen_ms={t['gen_ms']:.1f} "
                f"decode={t['gen_toks']:.2f} tok/s retr_ms={t['retrieval_ms']:.1f} "
                f"finish={t.get('finish_reason')} gpu_mib={t['mem']['gpu_mib']} "
                f"rss_mib={t['mem']['rss_kib']/1024:.1f}{mtp_s}"
            )
            if t.get("tool_calls"):
                fn = (t["tool_calls"][0].get("function") or {})
                lines.append(f"  tool={fn.get('name')} args={fn.get('arguments')}")
            content = (t.get("content") or "").replace("\n", " ")[:240]
            lines.append(f"  content={content}")
            if "BLUEBIRD-42" in (t.get("content") or ""):
                lines.append("  needle=HIT")
            elif t.get("finish_reason") == "tool_calls":
                lines.append("  needle=n/a (tool_calls)")
            else:
                lines.append("  needle=MISS")
        text = "\n".join(lines) + "\n"
        sum_path.write_text(text)
        print("===== summary =====", flush=True)
        print(text, flush=True)
        print(f"wrote {sum_path} {mem_path} {log_path}", flush=True)
        return 0
    finally:
        sampler.stop()
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()


if __name__ == "__main__":
    raise SystemExit(main())
