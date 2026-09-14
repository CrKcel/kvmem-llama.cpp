#!/usr/bin/env bash
# Replay maintained diffs without commits or Git identity changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${KVMEM_LLAMA_DIR:-$ROOT/llama.cpp}"
PATCH="$ROOT/patches/llama-kvmem-current.patch"
UPGRADE="$ROOT/patches/multimodal-upgrade.patch"
cd "$LLAMA"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "KVMem patches already applied"
elif git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "applied current KVMem patch to pinned llama.cpp"
elif git apply --check "$UPGRADE" 2>/dev/null; then
    git apply "$UPGRADE"
    echo "upgraded existing KVMem tree with multimodal support"
else
    echo "llama.cpp differs from the supported pin or KVMem baseline; no files changed" >&2
    echo "inspect local changes before replaying $PATCH" >&2
    exit 1
fi
