# KVMem + llama.cpp

## Near-lossless Qwen3.8-27B at a full 256K workspace on RTX 5060 Ti 16 GiB — decode stays ~30–40 tok/s

llama.cpp inference with tiered KV memory for long-running agents.

**KVMem** adds a bounded GPU KV working set, host-memory storage and query-based retrieval to [llama.cpp](https://github.com/ggml-org/llama.cpp). llama.cpp handles model loading, inference, quantization and MTP. The separate `llama-kvmem-server` provides OpenAI-compatible chat, tools and optional vision. **NVMe offload is not implemented.**

This port supports **Qwen3.8-27B GGUF quants**, including IQ3 and IQ4. The sibling [kvmem-qw3](https://github.com/kvmem/kvmem-qw3) is a CUDA-native runtime focused on Q8, primarily tested on RTX PRO 6000.

The logical workspace (`-c`) can extend beyond 256K using host RAM; quality at those lengths remains experimental.

The [KVMem paper](https://arxiv.org/abs/2609.04852) shows that, on queries up to 256K, keeping only a **32K GPU-resident active context** is essentially lossless versus the **full 256K** history: **LongMemEval-S** 85.6% vs 86.6% accuracy, **AgentLongBench** 60.9% vs 59.5% task success.

**KV streaming vs. KVMem.** Both methods support a full 256K context on a 16 GiB GPU by storing part of the KV cache in host RAM. [Raymond Huang’s adaptive KV-cache streaming](https://medium.com/@raymond860909/running-qwen-27b-on-16g-vram-with-full-context-length-building-adaptive-kv-cache-streaming-for-bf1e819116e9) keeps part of the KV cache in VRAM and stores the rest in host RAM. During decoding, it prefetches the offloaded KV layer by layer through reusable GPU buffers, overlapping transfers with computation. This preserves attention over the entire history, but longer contexts increase both attention work and PCIe traffic, eventually slowing decode.

KVMem retrieves relevant historical blocks into a bounded GPU window, limiting the KV used for attention. On RTX 5060 Ti, the current 256K tool benchmark achieves **30–33 token/s decode**, **437–466 token/s prefill for initial computation** and **243–255 token/s overall prefill**, including input reprocessing and cache management.

Current milestone: [`v0.14.0`](docs/milestones/v0.14.0.md).

## How KVMem works

Completed KV blocks are stored in host RAM. For each agent step, KVMem retrieves relevant blocks using the current query and places them in chronological order in a bounded GPU working set. Previously computed KV is reused across turns.

![High-level KVMem flow](docs/assets/kvmem-flow.svg)

Core flags:

| Flag | Meaning |
|---|---|
| `-c` | Logical workspace, including history stored off GPU. 256K is the tested default; larger is experimental. |
| `--kvmem-budget` | How many historical tokens retrieval may keep on GPU. |
| `--kvmem-gen-reserve` | GPU slots reserved for new tokens so retrieval cannot fill the pool. |
| `--kvmem-query-replay` | `auto` skips a second prefill when the GPU history view did not change; `legacy` always replays. |
| `--kvmem-query-policy` | `user` retrieves from the last real user question and reuses it on tool turns; `legacy` uses the older suffix policy. Recipes use `auto` + `user`. |
| `--kv-dtype` | Quantization of the **main** attention KV (e.g. q8_0, q5_0). |
| `--spec-kv-dtype` | Quantization of the **MTP draft** KV (often F16). |
| `--spec-type draft-mtp` | Enable multi-token prediction (`--spec-draft-n-max` is draft length). |
| `--mmproj` | Vision projector GGUF. Omit for text-only. |

GPU KV size is `budget + gen_reserve` (aligned to `--kvmem-block-tokens`). When history exceeds `--kvmem-budget`, retrieval picks blocks for the current last-user query. Clients should send the full `messages` history each turn.

## How KVMem attaches to llama.cpp

`kvmem/` holds the host store and retrieval logic; `src/adapter/` connects it through llama.cpp’s memory interface. Attention kernels and original positions stay unchanged. Reselection transfers only blocks that changed.

Do **not** commit a dirty `llama.cpp` working tree. The submodule pointer is the pin; `scripts/apply-patches.sh` replays `patches/`.

## Requirements

- Linux, x86-64 (current development).
- NVIDIA GPU. The 27B recipes below are tested on an RTX 5060 Ti with 16 GiB VRAM.
- C++17 compiler, CMake and CUDA toolkit. Tested with CMake 4.4.3 and CUDA 13.2.86.
- Python 3.10+ and `ss` (iproute2) for the startup scripts.
- Extra host RAM for spilled KV.

Generation is CUDA-only. AMD/ROCm and Metal are not wired here.

## Clone, patch, build

```bash
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp
git checkout v0.14.0
git submodule update --init
scripts/apply-patches.sh
scripts/build-cuda.sh
```

The submodule is ggml-org/llama.cpp at pin `b81c99b`. `scripts/apply-patches.sh` applies `patches/llama-kvmem-current.patch` (or `multimodal-upgrade.patch` on an older KVMem tree). Running it twice is safe. Do **not** apply numbered `0001`–`0004` together with the cumulative patch. See [patches/README.md](patches/README.md).

`scripts/build-cuda.sh` sets `GGML_CUDA_FA_ALL_QUANTS=ON` (needed for `--kv-dtype q5_0` on hybrid models). Binaries: `build/bin/llama-kvmem-server`.

The build script defaults to `CMAKE_CUDA_ARCHITECTURES=120a-real` for the tested RTX 5060 Ti. For another GPU, set `CMAKE_CUDA_ARCHITECTURES` to its appropriate target when running the script; other GPU targets have not been tested here.

## Recommended settings (16 GiB)

Both recipes use a 256K workspace and a bounded GPU KV working set. Sampling follows the Qwen3.8-27B card and can be overridden per request; `temperature=0` is greedy.

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
# Choose one recipe; both default to port 18200.
scripts/start-iq3.sh    # text + GPU vision + MTP, :18200
scripts/start-iq4.sh    # text + CPU vision + MTP, :18200

# Switch an existing KVMem service to IQ4.
scripts/start-iq4.sh --restart

# Override GPU and model, or preview the resolved configuration.
CUDA_VISIBLE_DEVICES=0 MODEL=/path/model.gguf scripts/start-iq4.sh
scripts/start-iq4.sh --dry-run
```

GPU selection honors `CUDA_VISIBLE_DEVICES`; otherwise it chooses a 5060 Ti or the only GPU. Ambiguous multi-GPU setups require an explicit selection. `MODEL`, `MMPROJ`, `MMPROJ_DEVICE` and `PORT` can override recipe defaults. CUDA libraries come from the build directory, caller environment or the toolkit recorded during compilation; use `CUDA_HOME` or `LD_LIBRARY_PATH` for a custom installation. An existing matching service is reused; switching configuration requires `--restart`, which only stops this project's server.

### IQ3 27B — text + vision, with MTP

`scripts/start-iq3.sh`. ISTA GGUF **as published** (MTP head not requantized). **Main KV q8_0**, MTP KV F16.

- Text: [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) → `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (use the `-mtp` file)
- Vision: [unsloth/Qwen3.8-27B-GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) `mmproj-BF16.gguf`, locally quantized to `mmproj-Q8_0.gguf` (`llama-quantize`; most weights Q8_0, 27 `ffn_down` tensors stay F16)

After downloading the BF16 projector, run from the repository root:

```bash
model_dir=models/unsloth/Qwen3.8-27B-GGUF
build/bin/llama-quantize --max-buffer-size 256 \
  "$model_dir/mmproj-BF16.gguf" "$model_dir/mmproj-Q8_0.gguf" Q8_0
```

The quantizer automatically falls back to F16 for the 27 incompatible `ffn_down` tensors.

```text
-m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
--mmproj mmproj-Q8_0.gguf --mmproj-offload --image-max-tokens 512
-c 262144 -n 16384 -b 512 -ngl 99
--kvmem --kvmem-method retrieval
--kvmem-query-replay auto --kvmem-query-policy user
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

Download [imatrix_unsloth.gguf](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/imatrix_unsloth.gguf) into the same model directory, then run from the repository root:

```bash
model_dir=models/unsloth/Qwen3.8-27B-GGUF
build/bin/llama-quantize --allow-requantize --max-buffer-size 256 \
  --imatrix "$model_dir/imatrix_unsloth.gguf" \
  --tensor-type-file scripts/quantization/qwen3.8-27b-iq4-xs-mtp-q4_0.types \
  "$model_dir/Qwen3.8-27B-UD-IQ4_XS.gguf" \
  "$model_dir/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf" IQ4_XS
```

The supplied [tensor map](scripts/quantization/qwen3.8-27b-iq4-xs-mtp-q4_0.types) preserves the original model's mixed quantization and changes only eight MTP matrices. The current quantizer requires the imatrix file to accept the existing low-bit tensors.

```text
-m Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf
--mmproj mmproj-BF16.gguf --no-mmproj-offload --image-max-tokens 512
-c 262144 -n 12288 -b 512 -ngl 99
--kvmem --kvmem-method retrieval
--kvmem-query-replay auto --kvmem-query-policy user
--kvmem-budget 32768 --kvmem-gen-reserve 12288
--kvmem-block-tokens 128 --kv-dtype q5_0
--spec-type draft-mtp --spec-draft-n-max 2 --spec-kv-dtype f16
--enable-thinking --reasoning-budget 4096
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --frequency-penalty 0.0 --repeat-penalty 1.0
```

### 5060 Ti results

**Test hardware:** RTX 5060 Ti 16 GiB, Intel Core Ultra 7 255H, 32 GiB RAM. Ubuntu 22.04.5 on WSL2 exposes 16 logical CPUs and 19.53 GiB RAM.

Both tasks use thinking with a 128-token budget and at most 512 output tokens per request. That is the speed-test setting; the start scripts default to `--reasoning-budget 4096`. IQ3 uses GPU Q8_0 vision; IQ4 uses CPU BF16 vision. RAM is runtime process RSS, excluding loading; VRAM is whole-GPU usage.

Task 1: ~12K text, then one image, then code generation.

| Metric | IQ3 | IQ4 |
|---|---:|---:|
| Prefill — initial computation | 549.50 token/s | 582.34 token/s |
| Prefill — overall | 517.99 token/s | 378.67 token/s |
| Image encode | **0.30 s** | **10.70 s** |
| Image decode | 37.35 token/s | 41.25 token/s |
| Code decode (512 tokens) | 38.28 token/s | 41.04 token/s |
| Runtime host RAM peak | **3012.52 MiB** | **3674.52 MiB** |
| VRAM peak | **15639.10 MiB** | **15539.10 MiB** |
| Minimum free VRAM | 412.90 MiB | 512.90 MiB |

Task 2: 32 tool-result rounds plus a base request, reaching **262058 / 262144 tokens** including generation. Both recipes receive identical requests; projectors stay loaded, but no images are sent.

| Metric | IQ3 | IQ4 |
|---|---:|---:|
| Prefill — initial computation | 436.72 token/s | 465.61 token/s |
| Prefill — overall | 243.16 token/s | 254.91 token/s |
| Aggregate tool-round decode | 29.96 token/s | 32.71 token/s |
| Code decode (512 tokens) | 29.33 token/s | 35.30 token/s |
| Runtime host RAM peak | **13444.70 MiB** | **11249.84 MiB** |
| VRAM peak | **15873.10 MiB** | **15931.10 MiB** |
| Minimum free VRAM | 178.90 MiB | 120.90 MiB |

Initial computation measures the first processing of new input. Overall includes any repeated processing, cache management and image encoding. Both rates use **total new input divided by the corresponding total time across the task**, counting visual rows as input positions. Decode includes thinking tokens. Code decode refers to the final request.

[Full results](docs/recommended-config-performance.md) · [Benchmark commands](docs/long-context-benchmark-2026-09-14.md). Summarize saved logs with `python3 scripts/summarize_canary.py <artifact-directory>`.

### Which one to run

| | IQ3 | IQ4 |
|---|---|---|
| Images | GPU encode **0.3 s** | CPU encode **10.7 s** |
| Decode (image / long-context task) | ~38 / ~30 token/s | ~41 / ~33 token/s |
| GPU KV window | 36K retrieve / 16K generate | 32K / 12K |
| Main KV | q8_0 | q5_0 |
| MTP weights | Official ISTA `-mtp` | Local Q4_0 requant of Unsloth |
| Runtime host RAM (image / 256K tool task) | 3012.52 / 13444.70 MiB RSS | 3674.52 / 11249.84 MiB RSS |

**Default: IQ3** for fast image encoding and a larger KV window. **IQ4** suits mostly text workloads with lower long-context RAM use, if ~11 s CPU image encoding is acceptable.

## APIs

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (sampling, stream, tools, optional images)

No auth or TLS. Bind `127.0.0.1`. Stream `usage` includes `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`.

## Documentation

- [v0.14.0 milestone](docs/milestones/v0.14.0.md)
- [Modification plan](docs/modification-plan.md)
- [Architecture](docs/architecture.md)
- [Patch replay](patches/README.md)
- [Recommended 16 GiB performance](docs/recommended-config-performance.md)
- [256K tool benchmark](docs/long-context-benchmark-2026-09-14.md)
- [Query replay](docs/query-replay-implementation-report-2026-09-14.md)
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

## Paper and citation

[KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852)

Copy the BibTeX entry below and cite it with `\cite{chai2026kvmem}`.

```bibtex
@misc{chai2026kvmem,
  title         = {{KVMem}: Virtualizing Million-Token Agent Workspaces on a Consumer {GPU}},
  author        = {Di Chai and Leye Wang and Zeshen Su and Zhiguo Xia and Zhihang Yu},
  year          = {2026},
  eprint        = {2609.04852},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.04852}
}
```
