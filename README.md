# kvmem_llamacpp

KVMem as a standalone library, attached to llama.cpp through
`llama_memory_i`. See [docs/modification-plan.md](docs/modification-plan.md).

**Current local milestone: [`v0.3.0`](docs/milestones/v0.3.0.md)** (2026-09-04).
llama.cpp can run KVMem. Product default `--kvmem` is retrieval + query-last 64.
Identity and needle revival hold on Qwen3-0.6B / 1.7B (RTX 5050); prefill/decode
through 16k is not badly degraded. P4 (hybrid) is not started.

## Layout

- `kvmem/` — host library (selection, CPU/NVMe tiers, runtime). No llama.cpp.
- `scripts/` — ModelScope downloads and GPU device helpers.
- `models/` — Unsloth GGUFs (gitignored).

## Build (P0 host tests)

```bash
.venv/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DKVMEM_BUILD_LLAMA=OFF
.venv/bin/cmake --build build -j
.venv/bin/ctest --test-dir build --output-on-failure
```

## Build (P1 llama.cpp + CUDA)

Pinned llama.cpp: `b81c99b` (`ggml: avoid KleidiAI buffer type init on dispatch`).
Patches live in `patches/` and are replayed with `scripts/apply-patches.sh`.
A fresh checkout of tag `v0.3.0` is the pin plus patches; run apply-patches
before building.

```bash
git checkout v0.3.0
git submodule update --init
source scripts/gpu.sh small          # RTX 5050; never use GPU 1 for <27B
scripts/apply-patches.sh
scripts/build-cuda.sh                # nvcc from ~/.cu13-env, sm_120
.venv/bin/ctest --test-dir build --output-on-failure
python3 scripts/identity_canary.py   # Qwen3-0.6B Q8_0 greedy on/off
python3 scripts/speed_canary.py      # KVMem-on vs off tok/s on RTX 5050
```

`llama-kvmem-cli` lands in `build/bin/`. This llama.cpp revision gates
`llama-cli` on `LLAMA_BUILD_SERVER`; the upstream greedy binary is
`llama-completion` (`-no-cnv --temp 0`). P1 identity compares
`llama-kvmem-cli` with and without `--kvmem`.

## GPUs on this machine

| nvidia-smi index | Card | VRAM | Use |
|---|---|---|---|
| 0 | RTX 5050 Laptop | 8 GiB | every model **below 27B** |
| 1 | RTX 5090 Laptop | 24 GiB | **27B only** |

CUDA's default device order is FASTEST_FIRST, so CUDA device 0 is the **5090**.
`scripts/gpu.sh` sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and binds the 5050/5090 by UUID.

```bash
source scripts/gpu.sh small   # CUDA_VISIBLE_DEVICES=0
source scripts/gpu.sh 27b     # CUDA_VISIBLE_DEVICES=1
```

## Test models (Unsloth, ModelScope only)

```bash
# venv with modelscope (already created if you used uv)
uv venv .venv
uv pip install --python .venv/bin/python modelscope cmake

scripts/download-test-models.sh ci    # 0.6B + 0.8B
scripts/download-test-models.sh all   # plus 1.7B/4B/27B
```

Daily canaries use Qwen3-0.6B Q8_0 and Qwen3.5-0.8B Q8_0 on GPU 0.
Qwen3.8-27B-UD-Q4_K_M is GPU 1 only.
