#!/usr/bin/env bash
# IQ3_S + GPU Q8_0 vision; Q8 main KV, F16 MTP KV, 36K / 16K budgets.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/start-server.py" \
    --recipe iq3 \
    --default-model "$ROOT/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" \
    --default-mmproj "$ROOT/models/unsloth/Qwen3.8-27B-GGUF/mmproj-Q8_0.gguf" \
    --default-vision-device gpu --kv q8_0 --budget 36864 --reserve 16384 \
    --explicit-sampling "$@"
