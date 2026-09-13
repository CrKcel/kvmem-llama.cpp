#!/usr/bin/env bash
# Start IQ3 llama-kvmem-server detached from the caller (Grok session, ssh, etc.).
# setsid -f puts it in a new session so a parent SIGKILL does not take it down.
# Kill only by PID (pidfile or :18200 listener). Never pkill -f.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PIDFILE="$ROOT/logs/iq3_18200.pid"
STDOUT="$ROOT/logs/opencode_kvmem_iq3_256k.stdout.log"
STDERR="$ROOT/logs/opencode_kvmem_iq3_256k.stderr.log"
BIN="$ROOT/build/bin/llama-kvmem-server"
MODEL="$ROOT/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
PORT=18200
RESTART=0
[[ "${1:-}" == "--restart" ]] && RESTART=1

mkdir -p "$ROOT/logs"

listener_pid() {
    ss -ltnp 2>/dev/null | awk -v p=":${PORT} " '
        index($0, p) {
            if (match($0, /pid=[0-9]+/)) {
                s = substr($0, RSTART+4, RLENGTH-4)
                print s
                exit
            }
        }'
}

health_ok() {
    curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

alive_pid() {
    local pid="${1:-}"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_pid() {
    local pid="$1"
    alive_pid "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 40); do
        alive_pid "$pid" || return 0
        sleep 0.25
    done
    kill -9 "$pid" 2>/dev/null || true
}

cur="$(listener_pid)"
if [[ "$RESTART" -eq 0 ]] && alive_pid "$cur" && health_ok; then
    echo "$cur" > "$PIDFILE"
    echo "already up pid=$cur http://127.0.0.1:${PORT}/health"
    exit 0
fi

if [[ "$RESTART" -eq 1 ]]; then
    if alive_pid "$cur"; then
        echo "stopping pid=$cur"
        stop_pid "$cur"
    elif [[ -f "$PIDFILE" ]]; then
        old="$(cat "$PIDFILE" 2>/dev/null || true)"
        if alive_pid "$old"; then
            echo "stopping pidfile pid=$old"
            stop_pid "$old"
        fi
    fi
    cur="$(listener_pid)"
    if alive_pid "$cur"; then
        echo "port ${PORT} still held by pid=$cur" >&2
        exit 1
    fi
elif alive_pid "$cur"; then
    echo "port ${PORT} already in use by pid=$cur (pass --restart)" >&2
    exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
[[ -f "$STDERR" ]] && mv "$STDERR" "${STDERR}.${ts}"
[[ -f "$STDOUT" ]] && mv "$STDOUT" "${STDOUT}.${ts}"

# shellcheck disable=SC1091
source "$ROOT/scripts/gpu.sh" 27b
export LD_LIBRARY_PATH="/home/leye/kvmem_qw3/.cu13-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

setsid -f "$BIN" \
    -m "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    -c 262144 -n 16000 -b 512 -ngl 99 \
    --kvmem --kvmem-method retrieval \
    --kvmem-budget 40000 --kvmem-gen-reserve 16000 \
    --kvmem-block-tokens 128 --kv-dtype q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --enable-thinking --reasoning-budget 4096 \
    > "$STDOUT" 2> "$STDERR" < /dev/null

ok=0
for i in $(seq 1 60); do
    if grep -q 'llama-kvmem-server listening' "$STDERR" 2>/dev/null && health_ok; then
        ok=1
        break
    fi
    sleep 1
done
pid="$(listener_pid)"
if [[ "$ok" -ne 1 ]] || ! alive_pid "$pid"; then
    echo "start failed; tail of $STDERR:" >&2
    tail -20 "$STDERR" >&2 || true
    exit 1
fi
echo "$pid" > "$PIDFILE"
ppid="$(ps -o ppid= -p "$pid" | tr -d ' ')"
echo "started pid=$pid ppid=$ppid http://127.0.0.1:${PORT}/health"
