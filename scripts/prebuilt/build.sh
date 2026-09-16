#!/usr/bin/env bash
# Run against the supplied source snapshot. Requires CMake, GCC and CUDA Toolkit.
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA 13 Toolkit directory}"
BUILD_DIR=${BUILD_DIR:-"$SOURCE/build-prebuilt"}
CMAKE=${CMAKE:-cmake}
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$CMAKE" -S "$SOURCE" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DKVMEM_BUILD_LLAMA=ON -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT="$SOURCE" \
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON \
  -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=OFF \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  '-DCMAKE_INSTALL_RPATH=$ORIGIN;$ORIGIN/../lib' \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
"$CMAKE" --build "$BUILD_DIR" --target \
  llama-kvmem-server llama-kvmem-cli llama-quantize \
  kvmem-chat-id-test kvmem-reasoning-budget-test kvmem-chat-template-test \
  kvmem_store_test pinned_kv_tier_test nvme_kv_tier_test kvmem_runtime_test raw_kv_store_test \
  -j "${JOBS:-4}"
