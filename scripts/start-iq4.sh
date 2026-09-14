#!/usr/bin/env bash
# IQ4_XS + CPU BF16 vision; Q5 main KV, F16 MTP KV, 32K / 12K budgets.
# Sampling uses the server's Qwen3.8 defaults, as in the measured configuration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/start-server.py" \
    --recipe iq4 \
    --default-model "$ROOT/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf" \
    --default-mmproj "$ROOT/models/unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf" \
    --default-vision-device cpu --kv q5_0 --budget 32768 --reserve 12288 "$@"
