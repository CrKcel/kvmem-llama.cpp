#!/usr/bin/env bash
# Download Unsloth GGUFs from ModelScope. Never uses Hugging Face.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "missing ${PY}; run: uv venv .venv && uv pip install --python .venv/bin/python modelscope" >&2
    exit 1
fi

# stage: ci | mtp | all
STAGE="${1:-ci}"

download() {
    local repo="$1" file="$2"
    echo "==> ModelScope ${repo} / ${file}"
    "$PY" - "$repo" "$file" "${ROOT}/models/${repo}" <<'PY'
import sys
from modelscope import snapshot_download
repo, fname, dest = sys.argv[1], sys.argv[2], sys.argv[3]
snapshot_download(repo, allow_patterns=[fname], local_dir=dest)
print("ok", dest + "/" + fname)
PY
}

# CI / daily: <27B → later run on RTX 5050 (GPU 0)
download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf
download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q8_0.gguf

# P7 draft-mtp: the daily 0.8B Q8_0 has no nextn; Unsloth's MTP repo does.
if [[ "$STAGE" == "all" || "$STAGE" == "mtp" ]]; then
    download unsloth/Qwen3.5-0.8B-MTP-GGUF Qwen3.5-0.8B-Q8_0.gguf
fi

if [[ "$STAGE" == "all" ]]; then
    download unsloth/Qwen3-1.7B-GGUF Qwen3-1.7B-Q4_K_M.gguf
    download unsloth/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf
    download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf
    # 27B → RTX 5090 (GPU 1) only
    download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf
fi
