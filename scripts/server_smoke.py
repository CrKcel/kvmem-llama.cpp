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
        usage = None
        for line in body.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
                if chunk.get("choices"):
                    raise SystemExit(f"stream usage chunk must have empty choices: {chunk}")
        if not usage or int(usage.get("prompt_tokens") or 0) < 1:
            raise SystemExit(f"stream missing usage.prompt_tokens: {body[-800:]}")
        if int(usage.get("total_tokens") or 0) != int(usage["prompt_tokens"]) + int(usage.get("completion_tokens") or 0):
            raise SystemExit(f"stream usage arithmetic: {usage}")
        print("PASS: streaming /v1/chat/completions usage", usage)

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
        if "BLUEBIRD-42" not in retr_text:
            print("WARN: turn-1 did not emit BLUEBIRD-42; still testing prefix reuse")

        mark = log_path.read_text()
        st, body = post_json(base + "/v1/chat/completions", {
            "messages": [
                {"role": "system", "content": "Read the notes and answer the user."},
                {"role": "user", "content": filler + "\n" + NEEDLE + "\n" + filler},
                {"role": "user", "content": "What is the secret code?"},
                {"role": "assistant", "content": retr_text},
                {"role": "user", "content": "What was the secret code again? Reply with the code only."},
            ],
            "max_tokens": 48,
            "temperature": 0,
            "stream": False,
        }, timeout=180)
        if st != 200:
            raise SystemExit(f"turn-2 chat failed {st}: {body}")
        logf.flush()
        err2 = log_path.read_text()[len(mark):]
        reuse = [ln for ln in err2.splitlines() if "prefix_reuse" in ln]
        if not reuse:
            raise SystemExit("missing KVMEM_TRACE prefix_reuse on turn 2")
        print("  ", reuse[-1])
        if "reused=1" not in reuse[-1]:
            raise SystemExit("turn 2 did not reuse prefix: " + reuse[-1])
        if "query_replay_skip_same" in err2:
            raise SystemExit("turn 2 new user must re-retrieve, not skip_same")
        if "KVMEM_TRACE query_replay begin=" not in err2:
            raise SystemExit("turn 2 new user missing query replay")
        def _ival(s, key):
            for part in s.split():
                if part.startswith(key + "="):
                    return int(part.split("=", 1)[1])
            return -1
        n_past = _ival(reuse[-1], "n_past")
        n_prompt = _ival(reuse[-1], "n_prompt")
        n_new = _ival(reuse[-1], "n_new")
        if n_past < 64 or n_new < 1 or n_new >= n_past or n_past + n_new > n_prompt + 8:
            raise SystemExit(f"prefix reuse sizes look wrong: {reuse[-1]}")
        turn2 = json.loads(body)["choices"][0]["message"]["content"]
        print("turn-2 output:", turn2[:200].replace("\n", " "))
        if "BLUEBIRD-42" not in turn2:
            raise SystemExit("turn 2 missed the old needle")
        print("PASS: prefix reuse + old needle still recalled")

        mark_t1 = log_path.read_text()
        tools_body = {
            "messages": [
                {"role": "user", "content": "What is the secret code?"},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_code",
                            "arguments": "{\"q\":\"secret\"}",
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "BLUEBIRD-42"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_code",
                    "description": "Look up a secret code",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }],
            "tool_choice": "auto",
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        }
        st, body = post_json(base + "/v1/chat/completions", tools_body, timeout=120)
        if st != 200:
            raise SystemExit(f"T1 tools-history chat failed {st}: {body}")
        logf.flush()
        err_t1 = log_path.read_text()[len(mark_t1):]
        parse_ln = [ln for ln in err_t1.splitlines() if "chat_parse" in ln]
        if not parse_ln:
            raise SystemExit("missing KVMEM_TRACE chat_parse for tools history")
        print("  ", parse_ln[-1])
        if "n_tools=1" not in parse_ln[-1] or "tool_hist=2" not in parse_ln[-1]:
            raise SystemExit("T1 did not parse tools/tool history: " + parse_ln[-1])
        if "prompt_has_tool=0" in parse_ln[-1]:
            raise SystemExit("T1 template prompt missing tool name: " + parse_ln[-1])
        st, body = post_json(base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools_body["tools"],
            "grammar": "root ::= \"a\"",
            "max_tokens": 1,
        })
        if st != 400:
            raise SystemExit(f"T1 tools+grammar should 400, got {st}: {body}")
        print("PASS: T1 tools parse + template sees tool history")

        mark_t2 = log_path.read_text()
        st, body = post_json(base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Look up the secret code."}],
            "tools": tools_body["tools"],
            "tool_choice": "required",
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
        }, timeout=120)
        if st != 200:
            raise SystemExit(f"T2 required-tool chat failed {st}: {body}")
        logf.flush()
        err_t2 = log_path.read_text()[len(mark_t2):]
        sample_ln = [ln for ln in err_t2.splitlines() if "chat_sample" in ln]
        if not sample_ln:
            raise SystemExit("missing KVMEM_TRACE chat_sample")
        print("  ", sample_ln[-1])
        if "grammar_type=tool_calls" not in sample_ln[-1]:
            raise SystemExit("T2 did not enable tool-call grammar: " + sample_ln[-1])
        t2_msg = json.loads(body)["choices"][0]
        print("T3 finish_reason:", t2_msg.get("finish_reason"))
        print("T3 message:", json.dumps(t2_msg.get("message", {}), ensure_ascii=False)[:400])
        if t2_msg.get("finish_reason") != "tool_calls":
            raise SystemExit("T3 expected finish_reason=tool_calls")
        calls = t2_msg.get("message", {}).get("tool_calls") or []
        if not calls:
            raise SystemExit("T3 missing message.tool_calls")
        fn = calls[0].get("function") or {}
        if fn.get("name") != "lookup_code":
            raise SystemExit(f"T3 tool name {fn.get('name')!r} != lookup_code")
        args = fn.get("arguments")
        if not isinstance(args, str) or not args.strip():
            raise SystemExit(f"T3 arguments should be a JSON string, got {args!r}")
        out_ln = [ln for ln in err_t2.splitlines() if "chat_out" in ln]
        if out_ln:
            print("  ", out_ln[-1])
        print("PASS: T3 non-stream OpenAI tool_calls")

        t5_user = "Look up the secret code."
        t5_msg = t2_msg.get("message") or {}
        call_id = (calls[0].get("id") if calls else None) or "call_1"
        mark_t5b = log_path.read_text()
        st, body = post_json(base + "/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": t5_user},
                t5_msg,
                {"role": "tool", "tool_call_id": call_id, "content": "BLUEBIRD-42"},
            ],
            "tools": tools_body["tools"],
            "tool_choice": "none",
            "max_tokens": 48,
            "temperature": 0,
            "stream": False,
        }, timeout=120)
        if st != 200:
            raise SystemExit(f"T5 tool-result chat failed {st}: {body}")
        logf.flush()
        err_t5 = log_path.read_text()[len(mark_t5b):]
        reuse_t5 = [ln for ln in err_t5.splitlines() if "prefix_reuse" in ln]
        if not reuse_t5:
            raise SystemExit("T5 missing KVMEM_TRACE prefix_reuse")
        print("  ", reuse_t5[-1])
        if "reused=1" not in reuse_t5[-1]:
            raise SystemExit("T5 did not reuse prefix: " + reuse_t5[-1])
        skip_t5 = [ln for ln in err_t5.splitlines() if "query_replay_skip_same" in ln]
        if not skip_t5:
            raise SystemExit("T5 same-user tool round should skip query replay: " + reuse_t5[-1])
        print("  ", skip_t5[-1])
        if "warm_skip=1" not in reuse_t5[-1]:
            raise SystemExit("T5 prefix_reuse missing warm_skip=1: " + reuse_t5[-1])
        if "gdn-after-query" in err_t5 or "mtp_resync" in err_t5:
            raise SystemExit("T5 replayed suffix despite same-query skip")
        if "query_replay begin=" in err_t5:
            raise SystemExit("T5 still ran query replay")
        n_prompt_t5 = _ival(reuse_t5[-1], "n_prompt")
        qspan = reuse_t5[-1].split("query=", 1)[-1] if "query=" in reuse_t5[-1] else ""
        q0_t5 = q1_t5 = -1
        if qspan.startswith("["):
            inner = qspan[1:].split(")", 1)[0]
            parts = inner.split(",")
            if len(parts) == 2:
                q0_t5, q1_t5 = int(parts[0]), int(parts[1])
        print("T5 query span:", q0_t5, q1_t5, "n_prompt:", n_prompt_t5)
        if q0_t5 < 0 or q1_t5 <= q0_t5:
            raise SystemExit("T5 query span missing/invalid: " + reuse_t5[-1])
        if n_prompt_t5 > 0 and q1_t5 >= n_prompt_t5:
            raise SystemExit(
                f"T5 query must be last-user only, not the whole prompt: "
                f"query=[{q0_t5},{q1_t5}) n_prompt={n_prompt_t5}")
        t5_out = json.loads(body)["choices"][0]
        t5_text = (t5_out.get("message") or {}).get("content") or ""
        print("T5 finish_reason:", t5_out.get("finish_reason"))
        print("T5 output:", t5_text[:200].replace("\n", " "))
        if t5_out.get("finish_reason") == "tool_calls":
            raise SystemExit("T5 tool_choice=none should not return tool_calls")
        if "BLUEBIRD-42" not in t5_text:
            raise SystemExit("T5 missed the tool result BLUEBIRD-42")
        print("PASS: T5 same-user tool round skip_same + last-user query")

        mark_t4 = log_path.read_text()
        st, body = post_json(base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Look up the secret code."}],
            "tools": tools_body["tools"],
            "tool_choice": "required",
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
        }, timeout=120)
        if st != 200 or "data:" not in body:
            raise SystemExit(f"T4 stream tools failed {st}: {body[:400]}")
        if "data: [DONE]" not in body:
            raise SystemExit("T4 stream missing [DONE]")
        saw_tc = False
        finish = None
        name_acc = ""
        for ln in body.splitlines():
            if not ln.startswith("data:") or ln.strip() == "data: [DONE]":
                continue
            raw = ln[5:].strip()
            if not raw:
                continue
            chunk = json.loads(raw)
            ch0 = (chunk.get("choices") or [{}])[0]
            if ch0.get("finish_reason"):
                finish = ch0["finish_reason"]
            delta = ch0.get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                saw_tc = True
                fn = (tc.get("function") or {})
                if fn.get("name"):
                    name_acc += fn["name"]
        logf.flush()
        err_t4 = log_path.read_text()[len(mark_t4):]
        stream_ln = [ln for ln in err_t4.splitlines() if "chat_stream" in ln]
        if stream_ln:
            print("  ", stream_ln[-1])
        print("T4 finish_reason:", finish, "saw_tool_calls:", saw_tc, "name:", name_acc)
        if not saw_tc:
            raise SystemExit("T4 stream missing delta.tool_calls")
        if "lookup_code" not in name_acc:
            raise SystemExit(f"T4 stream tool name {name_acc!r} != lookup_code")
        if finish != "tool_calls":
            raise SystemExit(f"T4 expected finish_reason=tool_calls, got {finish!r}")
        print("PASS: T4 stream delta.tool_calls")
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
