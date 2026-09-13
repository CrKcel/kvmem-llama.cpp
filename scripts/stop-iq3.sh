#!/usr/bin/env bash
# Stop the detached IQ3 server by PID only (pidfile or :18200 listener).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$ROOT/logs/iq3_18200.pid"
PORT=18200

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

alive_pid() {
    local pid="${1:-}"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

pid="$(listener_pid)"
if ! alive_pid "$pid" && [[ -f "$PIDFILE" ]]; then
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
fi
if ! alive_pid "$pid"; then
    echo "iq3 not running"
    rm -f "$PIDFILE"
    exit 0
fi

echo "stopping pid=$pid"
kill "$pid" 2>/dev/null || true
for i in $(seq 1 40); do
    alive_pid "$pid" || break
    sleep 0.25
done
if alive_pid "$pid"; then
    kill -9 "$pid" 2>/dev/null || true
fi
rm -f "$PIDFILE"
echo "stopped pid=$pid"
