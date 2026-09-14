# KVMem + llama.cpp

## Near-lossless Qwen3.8-27B at a full 256K workspace on a 16 GiB GPU

llama.cpp inference with tiered KV memory for long-running agents.

This repository attaches **KVMem** to a pinned [llama.cpp](https://github.com/ggml-org/llama.cpp) tree. llama.cpp remains the inference engine: GGUF loading, graphs, quantization, Flash Attention, hybrid GDN, sampling, and optional MTP. KVMem turns previously computed attention KV into reusable agent memory: a bounded GPU working set, colder blocks on host RAM, and query-conditioned retrieval of historical blocks. **NVMe offload is not implemented in this llama.cpp port.**

The sibling project [kvmem/kvmem-qw3](https://github.com/kvmem/kvmem-qw3) is a CUDA-native Qwen runtime with the same KVMem idea. That stack is built around **Qwen3.8-27B Q8** and was primarily tested on an **RTX PRO 6000**. This llama.cpp port runs **Qwen3.8-27B in many GGUF quants** (IQ3, IQ4, and denser formats llama.cpp already supports). Hardware coverage is whatever this llama.cpp CUDA build can use: still **NVIDIA only**, **Ampere or newer** (RTX 30 / A100 and later). Ada and Blackwell (including 16 GiB 5060 Ti) work with the recipes below; CUDA 12.8+ is required for SM120.

KVMem’s logical workspace (`-c`) is **not** capped by the model’s native context window. History that does not fit on the GPU is stored as KV on the host. If RAM is large enough, `-c` can go past 256K. Whether quality still holds at those extra lengths has **not** been fully tested in this llama.cpp port, so a larger `-c` is experimental — try it if you want.

The [KVMem paper](https://arxiv.org/abs/2609.04852) shows that, on queries up to 256K, keeping only a **32K GPU-resident active context** is essentially lossless versus the **full 256K** history: **LongMemEval-S** 85.6% vs 86.6% accuracy, **AgentLongBench** 60.9% vs 59.5% task success.

**Speed vs paging the full KV.** [Adaptive KV cache streaming](https://medium.com/@raymond860909/running-qwen-27b-on-16g-vram-with-full-context-length-building-adaptive-kv-cache-streaming-for-bf1e819116e9) also fits Qwen3.8-27B at 256K on 16 GiB, but it still **attends over the entire history** and streams KV from host RAM on every layer. Decode therefore falls as context grows (their RTX 5070 Ti figures: ~50 tok/s at 8K, ~20 tok/s at 176K, ~10 tok/s at 256K). KVMem instead **retrieves a bounded GPU window** (~32–50K). Attention and decode cost stay that of the window, not of the full 256K: on a 5060 Ti 16 GiB we stay around **30–40 tok/s** decode and **~500 tok/s** prefill, including 61K cache-miss prompts. That is the speed tradeoff: streaming keeps exact full-context attention; KVMem keeps near-lossless quality at roughly constant speed.

Paper: [https://arxiv.org/abs/2609.04852](https://arxiv.org/abs/2609.04852)

Current milestone: [`v0.13.0`](docs/milestones/v0.13.0.md).

## Why this repo?

- **llama.cpp as the engine:** model load, CUDA kernels, FA, GDN, MTP, and chat templates stay upstream.
- **KV memory for agents:** keep contextualized KV blocks across GPU and host, then retrieve a bounded working set for the current query.
- **Thin, replayable patches:** llama.cpp is a submodule pin plus `patches/`. Upgrade is bump pin + replay, not a diverged fork.
- **Out-of-tree server:** `llama-kvmem-server` is OpenAI-compatible (`/v1/chat/completions`) and does not patch `llama-server`.
- **Optional vision:** `--mmproj` loads a GGUF projector; `image_url` is base64 or HTTP.

## How KVMem works

KVMem treats an agent’s accumulated KV cache as virtual memory. When the workspace exceeds GPU capacity, it stores completed KV blocks in host memory instead of discarding or summarizing them. At each agent step, KVMem uses the current query to select relevant historical blocks and materializes them, in chronological order, into a bounded GPU-resident execution view. By reusing previously computed KV states and loading only the blocks needed for the current step, KVMem supports large persistent workspaces while keeping GPU memory usage bounded.

![High-level KVMem flow](docs/assets/kvmem-flow.svg)

Core flags:

| Flag | Meaning |
|---|---|
| `-c` | Logical workspace, including history stored off GPU. 256K is the tested default; larger is experimental. |
| `--kvmem-budget` | How many historical tokens retrieval may keep on GPU. |
| `--kvmem-gen-reserve` | GPU slots reserved for new tokens so retrieval cannot fill the pool. |
| `--kv-dtype` | Quantization of the **main** attention KV (e.g. q8_0, q5_0). |
| `--spec-kv-dtype` | Quantization of the **MTP draft** KV (often F16). |
| `--spec-type draft-mtp` | Enable multi-token prediction (`--spec-draft-n-max` is draft length). |
| `--mmproj` | Vision projector GGUF. Omit for text-only. |

GPU KV size is `budget + gen_reserve` (aligned to `--kvmem-block-tokens`). When history exceeds `--kvmem-budget`, retrieval picks blocks for the current last-user query. Clients should send the full `messages` history each turn.

## How KVMem attaches to llama.cpp

```
kvmem/            engine-agnostic library (selection, host store, plans)
                  no #include of llama.cpp headers
src/adapter/      llama_memory_i implementation (slot pool, hybrid, MTP follower)
tools/            llama-kvmem-cli, llama-kvmem-server
llama.cpp/        git submodule, pinned commit
patches/          diffs against that pin (hooks + multimodal)
```

llama.cpp still owns attention. KVMem does **not** patch Flash Attention kernels or repack the window to `[0..W)`. GPU cache is a **bounded slot pool**. Reselect is a plan diff: skip resident blocks, stage in/out only what changed.

Do **not** commit a dirty `llama.cpp` working tree. The submodule pointer is the pin; `scripts/apply-patches.sh` replays `patches/`.

## Requirements

- Linux, x86-64 (current development).
- NVIDIA GPU, Ampere or newer (RTX 30 / A100 and later). 16 GiB is enough for the 27B recipes below.
- CMake 3.18+, C++17; CUDA 12.8+ for Blackwell / SM120.
- Extra host RAM for spilled KV.

Generation is CUDA-only. AMD/ROCm and Metal are not wired here.

## Clone, patch, build

```bash
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp
git checkout v0.13.0
git submodule update --init
scripts/apply-patches.sh
scripts/build-cuda.sh
```

The submodule is ggml-org/llama.cpp at pin `b81c99b`. `scripts/apply-patches.sh` applies `patches/llama-kvmem-current.patch` (or `multimodal-upgrade.patch` on an older KVMem tree). Running it twice is safe. Do **not** apply numbered `0001`–`0004` together with the cumulative patch. See [patches/README.md](patches/README.md).

`scripts/build-cuda.sh` sets `GGML_CUDA_FA_ALL_QUANTS=ON` (needed for `--kv-dtype q5_0` on hybrid models). Binaries: `build/bin/llama-kvmem-server`.

## Recommended settings (16 GiB)

Measured on RTX 5060 Ti (16 GiB). Logical `-c` is 256K; only the KVMem pool sits in VRAM. Sampling follows the Qwen3.8-27B card; requests can override. `temperature=0` is greedy.

| | Thinking (these recipes) | Non-thinking |
|---|---:|---:|
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 0.0 | 1.5 |
| frequency_penalty | 0.0 | 0.0 |
| repetition_penalty | 1.0 | 1.0 |

```bash
scripts/start-iq3.sh    # text + GPU vision + MTP, :18200
scripts/start-iq4.sh    # text + CPU vision + MTP, :18200
```

### IQ3 27B — text + vision, with MTP

`scripts/start-iq3.sh`. ISTA GGUF **as published** (MTP head not requantized). **Main KV q8_0**, MTP KV F16.

- Text: [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) → `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (use the `-mtp` file)
- Vision: [unsloth/Qwen3.8-27B-GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) `mmproj-BF16.gguf`, locally quantized to `mmproj-Q8_0.gguf` (`llama-quantize`; most weights Q8_0, 27 `ffn_down` tensors stay F16)

```text
-m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
--mmproj mmproj-Q8_0.gguf --mmproj-offload --image-max-tokens 512
-c 262144 -n 16384 -b 512 -ngl 99
--kvmem --kvmem-method retrieval
--kvmem-budget 36864 --kvmem-gen-reserve 16384
--kvmem-block-tokens 128 --kv-dtype q8_0
--spec-type draft-mtp --spec-draft-n-max 2 --spec-kv-dtype f16
--enable-thinking --reasoning-budget 4096
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --frequency-penalty 0.0 --repeat-penalty 1.0
```

If GPU vision does not fit, `MMPROJ_DEVICE=cpu`.

### IQ4 27B — text + vision on CPU, with MTP

`scripts/start-iq4.sh`. Projector is **on CPU** by default (`--no-mmproj-offload`) so 16 GiB VRAM stays for the language model. Override with `MMPROJ_DEVICE=gpu` if you have spare VRAM.

Download Unsloth `Qwen3.8-27B-UD-IQ4_XS.gguf` and `mmproj-BF16.gguf` ([ModelScope](https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) / [Hugging Face](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)). Upstream MTP head is Q6_K / Q8_0.

The language GGUF this recipe runs, `Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf`, is **not** an Unsloth release: only MTP (`blk.64`) matmul weights were requantized to **Q4_0**. Vision uses the official **BF16** `mmproj-BF16.gguf` on CPU.

```text
-m Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf
--mmproj mmproj-BF16.gguf --no-mmproj-offload --image-max-tokens 512
-c 262144 -n 12288 -b 512 -ngl 99
--kvmem --kvmem-method retrieval
--kvmem-budget 32768 --kvmem-gen-reserve 12288
--kvmem-block-tokens 128 --kv-dtype q5_0
--spec-type draft-mtp --spec-draft-n-max 2 --spec-kv-dtype f16
--enable-thinking --reasoning-budget 4096
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --frequency-penalty 0.0 --repeat-penalty 1.0
```

Main weights IQ4_XS, MTP head Q4_0. **Main KV q5_0**, MTP KV F16. Max output 12K.

### 5060 Ti results

Same flow for both recipes: ~12K text, then one image, then a short follow-up. Thinking on. Full numbers: [recommended-config-performance.md](docs/recommended-config-performance.md).

| | IQ3 (GPU vision) | IQ4 (CPU vision) |
|---|---:|---:|
| 12K text prefill | 22.0 s | 22.0 s |
| Image encode | **0.30 s** | **9.75 s** |
| Image request (incl. encode) | 4.2 s | 13.9 s |
| Image decode | 37 tok/s | 39 tok/s |
| Follow-up prefill | 0.50 s | 0.57 s |
| Follow-up decode | 38 tok/s | 40 tok/s |
| GPU peak | **15.3 GiB** | **15.2 GiB** |
| Host RSS peak | 11.7 GiB | 13.6 GiB |

### Which one to run

| | IQ3 | IQ4 |
|---|---|---|
| Images | GPU encode **0.3 s** | CPU encode **~10 s** |
| Decode | ~38 tok/s | ~40 tok/s |
| GPU KV window | 36K retrieve / 16K generate | 32K / 12K |
| Main KV | q8_0 | q5_0 |
| MTP weights | Official ISTA `-mtp` | Local Q4_0 requant of Unsloth |
| Host RAM | ~12 GiB RSS | ~14 GiB RSS (BF16 projector on CPU) |

**Recommendation:** **IQ3** if you use images or want the published MTP file and a larger KV window. **IQ4** if the work is mostly text and a ~10 s image encode is acceptable. On this 16 GiB 5060 Ti, default is **IQ3** (`scripts/start-iq3.sh`).

## APIs

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (sampling, stream, tools, optional images)

No auth or TLS. Bind `127.0.0.1`. Stream `usage` includes `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`.

## Documentation

- [v0.13.0 milestone](docs/milestones/v0.13.0.md)
- [Modification plan](docs/modification-plan.md)
- [Architecture](docs/architecture.md)
- [Patch replay](patches/README.md)
- [Recommended 16 GiB performance](docs/recommended-config-performance.md)
- [Multimodal usage](docs/multimodal-implementation-report-2026-09-14.md)
- Native Qwen engine: [kvmem/kvmem-qw3](https://github.com/kvmem/kvmem-qw3)

## Project layout

```
kvmem/            Host KVMem library (no llama.cpp includes)
src/adapter/      llama_memory_i wrapper
tools/            llama-kvmem-cli, llama-kvmem-server, vision helpers
scripts/          apply-patches, CUDA build, GPU bind, start helpers
patches/          Diffs against the llama.cpp pin
docs/             Architecture, milestones, multimodal
llama.cpp/        Submodule (pin only; apply patches after clone)
models/           Local GGUFs (gitignored)
```

## License

Checkpoints are distributed separately and may use different terms. llama.cpp remains under its upstream license. KVMem-qw3 source is Apache-2.0; this port should be treated the same unless a `LICENSE` file is added to this tree.
