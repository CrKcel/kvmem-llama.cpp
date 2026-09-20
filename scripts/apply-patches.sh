#!/usr/bin/env bash
# Replay the Bonsai adapter patch without modifying the submodule pin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${KVMEM_LLAMA_DIR:-$ROOT/llama.cpp}"
PATCH="$ROOT/patches/llama-kvmem-current.patch"
cd "$LLAMA"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "Bonsai KVMem patch already applied"
elif git apply --check "$PATCH"; then
    git apply "$PATCH"
    echo "applied Bonsai KVMem patch to Prism"
else
    echo "Source does not match the Prism baseline; no files changed" >&2
    exit 1
fi
