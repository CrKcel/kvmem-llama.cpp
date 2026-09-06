#!/usr/bin/env bash
# Apply patches/ onto the pinned llama.cpp submodule.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/llama.cpp"
PATCHDIR="$ROOT/patches"

if [[ ! -d "$LLAMA/.git" && ! -f "$LLAMA/.git" ]]; then
    echo "llama.cpp is not a git checkout: $LLAMA" >&2
    exit 1
fi

shopt -s nullglob
patches=("$PATCHDIR"/*.patch)
if [[ ${#patches[@]} -eq 0 ]]; then
    echo "no patches in $PATCHDIR"
    exit 0
fi

cd "$LLAMA"
if git am --show-current-patch >/dev/null 2>&1; then
    echo "git am already in progress in llama.cpp" >&2
    exit 1
fi

# git am needs an identity even for a local replay.
if [[ -z "$(git config user.name)" ]]; then
    git config user.name kvmem
    git config user.email kvmem@local
fi

# Idempotent: 0004 is the last hunk (CUDA stage-in).
if grep -q 'llama-kvmem-stagein.cu' src/CMakeLists.txt; then
    echo "patches already applied"
    exit 0
fi
# 0001-0003 already on the tree (v0.6.0 checkout); apply 0004 only.
if grep -q 'option(LLAMA_KVMEM' CMakeLists.txt; then
    git am "$PATCHDIR"/0004-kvmem-cuda-stagein.patch
    echo "applied 0004 (CUDA stage-in) on existing 0001-0003"
    exit 0
fi

git am "${patches[@]}"
echo "applied ${#patches[@]} patch(es)"
