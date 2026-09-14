# kvmem_llamacpp

KVMem as a standalone library, attached to llama.cpp through
`llama_memory_i`. See [docs/modification-plan.md](docs/modification-plan.md).

**Current local milestone: [`v0.13.0`](docs/milestones/v0.13.0.md)** (2026-09-14).
IQ3 chat completions accept OpenAI `image_url` (base64 or HTTP) with
`--mmproj`; see [multimodal usage](docs/multimodal-implementation-report-2026-09-14.md).
GPU KV default **q8_0** (`q5_0` needs `GGML_CUDA_FA_ALL_QUANTS`). Orig-pos
cells; restore is packed GPU K/V memcpy (no unrotated raw-K on the product
path). Retrieval uses pre-RoPE mean-K (prefill write + decode running sum
after pin). Full blocks copy packed K/V to host asynchronously during
prefill. `llama-kvmem-server` reuses the GPU prefix across requests; same
last-user tool rounds **skip** retrieval/query-replay and only prefill
`n_new`. Compact/rewrite that leaves LCP far short of the previous cache
**drops reuse** and does a full retrieval. If the post-query tail is longer
than gen-reserve, prefill that tail with offload then retrieve. Query span
is the last ChatML user role block matching last-user text. Persistent GDN
snapshot is at **prefill end / gen-start**, not last query. MTP KV follows
`--kv-dtype` unless `--spec-kv-dtype` overrides. Stream `usage` includes
DeepSeek `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`. Chat
sampling follows the Qwen3.8-27B card (thinking vs non-thinking). Decode/tool
tokens after the query are **recency** (`--kvmem-recent-tokens`, default 0),
not GPU-mandatory. `--kvmem` is retrieval + query-last 64. MTP remains
optional (default **none**). 0.8B `server_smoke` T1–T5 GO. P6 is not started.
Phase D `state_write` is not in this tag.

## Layout

- `kvmem/` — host library (selection, CPU/NVMe tiers, runtime). No llama.cpp.
- `src/adapter/` — `llama_memory_i` wrapper (dense slot-pool + hybrid + MTP follower).
- `tools/` — `llama-kvmem-cli`, `llama-kvmem-server`.
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
A fresh checkout of tag `v0.13.0` is the pin plus the cumulative KVMem patch
(including multimodal). See [patch replay](patches/README.md). Run apply-patches
before building.

```bash
git checkout v0.13.0
git submodule update --init
source scripts/gpu.sh small          # RTX 5050; never use GPU 1 for <27B
scripts/apply-patches.sh
scripts/build-cuda.sh                # nvcc from ~/.cu13-env, sm_120, FA_ALL_QUANTS
.venv/bin/ctest --test-dir build --output-on-failure
python3 scripts/identity_canary.py \
  -m models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
python3 scripts/mtp_canary.py --gpu small   # needs 0.8B-MTP-GGUF (has nextn)
python3 scripts/server_smoke.py      # greedy/stream/retrieval/tools T1–T5
```

`scripts/build-cuda.sh` passes **`-DGGML_CUDA_FA_ALL_QUANTS=ON`**. Without it,
llama.cpp only builds FlashAttention vec kernels for F16 / Q4_0 / Q8_0 / BF16;
`--kv-dtype q5_0` (also q4_1 / q5_1) falls off the native FA path and Qwen3.8
hybrid prefill collapses (tens of tok/s, GPU util stuck low). q8_0 KV does not
need this flag. Reconfigure+rebuild ggml-cuda after flipping it.

`llama-kvmem-cli` and `llama-kvmem-server` land in `build/bin/`. This
llama.cpp revision gates `llama-cli` on `LLAMA_BUILD_SERVER`; the upstream
greedy binary is `llama-completion` (`-no-cnv --temp 0`). P1 identity
compares `llama-kvmem-cli` with and without `--kvmem`.

Independent OpenAI-compatible server (P5, single slot, does not patch
`llama-server`):

```bash
source scripts/gpu.sh small
build/bin/llama-kvmem-server -m models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf \
  --port 8080 -c 2048 --kvmem --kvmem-budget 256 -ngl 99
# POST /v1/chat/completions  (sampling + stream + tools; temperature=0 for greedy)
# last user message becomes the retrieval query span; optional body.kvmem.pin / force_substr
python3 scripts/server_smoke.py
```

Default IQ4 server on the RTX 5060 Ti (bound by GPU UUID):

```bash
scripts/start-iq4.sh             # http://127.0.0.1:18200
scripts/start-iq4.sh --restart   # restart the server on this port
```

Uses `Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf`: IQ4_XS main weights,
Q4_0 MTP weights, Q5_0 target KV, F16 MTP KV, and MTP draft length 2.
Defaults: context 262144, batch 512, retrieval budget 32000, generation
reserve 12000 (11904 after 128-token block alignment), maximum output
16000, thinking enabled with a 4096-token reasoning budget. Logs are
`logs/opencode_kvmem_iq4_256k.{stdout,stderr}.log`; PID is in
`logs/iq4_18200.pid`.

CLI/server accept `--spec-kv-dtype f16` (also `q8_0`, `q5_0`, `q4_0`, `f32`)
to set MTP K/V independently of the target cache. Without this option, MTP
inherits the target K/V types. For example, `--kv-dtype q5_0 --spec-kv-dtype f16`
uses Q5 target KV and F16 MTP KV, as configured in the default IQ4 startup script.
The [controlled experiment](docs/mtp-kv-controlled-experiment-2026-09-13.md)
measured a 42 MiB lower whole-GPU sampled peak with F16 MTP KV than with Q5 MTP KV.

Chat sampling defaults follow the [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices):

| Parameter | Thinking (default IQ4 mode) | Non-thinking |
|---|---:|---:|
| `temperature` | 1.0 | 0.7 |
| `top_p` | 0.95 | 0.80 |
| `top_k` | 20 | 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 0.0 | 1.5 |
| `repetition_penalty` | 1.0 | 1.0 |
| `frequency_penalty` | 0.0 | 0.0 |

`/v1/chat/completions` accepts these fields plus `seed` (uint32; `4294967295`
means random). `repeat_penalty` is an alias for `repetition_penalty`; conflicting
aliases return HTTP 400. Omitted or null fields inherit defaults. Non-numeric,
non-finite, or out-of-range values return HTTP 400: temperature [0,2], top_p/min_p
[0,1], top_k integer >= 0, presence/frequency penalties [-2,2], repetition penalty > 0.
Penalties use llama.cpp's 64-token history window. Explicit `temperature=0`
uses greedy decoding, disables top-k/top-p/min-p filtering, and retains penalties.
To reproduce the previous unpenalized greedy behavior in non-thinking mode,
also send `presence_penalty=0`.

Server flags `--temp`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`,
`--frequency-penalty`, `--repeat-penalty`, and `--seed` override defaults for both
modes. Priority: request fields > explicit server flags > mode defaults. The IQ4
script uses mode defaults, so disabling thinking per request also selects the
non-thinking sampling defaults. `KVMEM_TRACE sampling` logs effective parameters
for every request; both normal generation and MTP use this configuration.
`scripts/start-iq3.sh` explicitly sets the Thinking sampling values as process
defaults; requests can override them, including when disabling thinking.
The 4096 reasoning budget and 16000 output limit remain deployment settings;
this aligns sampling parameters, not the model card's full output/context budgets.
Run `python3 scripts/chat_sampling_canary.py` to verify defaults, overrides,
HTTP validation, and ordinary/MTP generation on the RTX 5050 with the local
0.8B MTP model. This starts isolated test servers and leaves IQ4 running.

Example request fields for an explicit Thinking configuration:

```json
{
  "messages": [{"role": "user", "content": "Write a Python LRU cache."}],
  "enable_thinking": true,
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0,
  "seed": 42,
  "max_tokens": 1024
}
```

## GPUs on this machine

| nvidia-smi index | Card | VRAM | Use |
|---|---|---|---|
| 0 | RTX 5050 Laptop | 8 GiB | every model **below 27B** |
| 1 | RTX 5090 Laptop (24 GiB) or RTX 5060 Ti (16 GiB) | **27B only** |

CUDA's default device order is FASTEST_FIRST unless `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
`scripts/gpu.sh` binds by UUID. Never put 27B on the 5050.

```bash
source scripts/gpu.sh small   # 5050
source scripts/gpu.sh 27b     # 5090 if present, else 5060 Ti
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
