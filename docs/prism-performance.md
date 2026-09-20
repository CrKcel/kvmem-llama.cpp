# Unmodified Prism 32K performance comparison

Measured locally on 2026-09-20 at the user's request. The original Prism runtime means commit `9a9394a895b96003ca842a6041cb28ac49a108f7`, with no KVMem source patches.

## Reference build

`scripts/windows/build-prism-reference.ps1` exports the pinned Git revision into `build-win-prism-reference/source` and builds its `llama-server`. All **3526 archived source files** were compared byte-for-byte against the archive, with no differences. No installed runtime was replaced.

The reference uses upstream's `LLAMA_USE_SYSTEM_GGML` option to link the existing, unmodified Prism GGML libraries. The adapter has no changes under `ggml`; SHA256 checks confirmed all four installed libraries match the KVMem build's libraries. This keeps CPU/CUDA quantization kernels and CUDA compilation settings identical. An external CMake wrapper supplies the CUDA runtime link dependencies omitted by the static package config; original source files are unchanged.

Both use MSVC 19.44, CUDA 12.9.86, SM120a and Release. Reference server SHA256: `8d5624bfbdb942e3ef7c0e7b1c8f016942d72a68b8095a25b61148e4e4ebd156`.

## Completed 32K run

RTX 5050 Laptop (8151 MiB), Bonsai 2 27B PTQ1_0, Q8 K/V, Flash Attention, `-ngl 99`, batch/ubatch 128, threads/batch threads 4, one slot, no speculation, reasoning disabled. Reference context is **32768**, with full native KV. Memory fitting, server RAM prompt cache and context checkpoints are disabled. A short arithmetic request warmed the model before the measured request. The measured prompt had no cache hits.

| Measurement | Unmodified Prism |
| --- | --- |
| Input | **32035 tokens** |
| Output | **256 tokens** |
| Prefill time | **236.893 s** |
| Prefill rate | **135.23 token/s** |
| Decode time | **29.066 s** |
| Decode rate, server-reported | **8.77 token/s** |
| HTTP request elapsed | **266.029 s** |
| Whole-device peak, 1 s sampling | **7395 MiB (7.22 GiB)** |
| OOM / crash | None |

Native server decode timing excludes the first output token from its rate numerator (255 / 29.066 s); dividing all 256 by the same duration gives 8.81 token/s. The small convention difference does not change the conclusion.

The input reuses the previous 32K archive, with the instruction changed to request a continuous essay. Output was capped at 256 tokens, and finished by length. This is a single warmed run, not a multi-run statistical benchmark.

## Comparison with the previous KVMem runs

| Runtime / retained KV | History size | Prefill token/s | Decode token/s | Peak MiB |
| --- | --- | --- | --- | --- |
| Original Prism / full KV | 32035 | **135.23** | **8.77** | **7395** |
| KVMem / 2K budget + 1K reserve | 32013 initially | 149.67 | 13.54 | 6263 |
| KVMem / 24K budget + 10K reserve | 64653 initially | **129.36** | **9.40** | **7454** |

KVMem measurements are from [the earlier validation](bonsai-validation.md). They include retrieval and state replay. Their 256-token decode measurements were follow-up requests after the initial archive. The reference instead generates directly after its long input. Prompt text, output text, retained KV lengths and cache/replay work differ, so this is a practical speed comparison, **not a controlled estimate of adapter overhead**. The original 32K and KVMem 64K/24K-budget runs are in a similar throughput range. The smaller 2K budget is faster but failed two of three long-history recall checks.

An earlier original 64K full-KV probe could start but showed about 660 MiB shared GPU memory in Windows counters versus about 92 MiB in the 32K run, and only roughly 35-40 prompt tokens/s in the early portion. That probe was intentionally stopped under memory pressure; it is not a completed 64K benchmark. At the user's direction, a subsequent 34K-capacity comparison was also stopped in favor of this 32K run. Its completed 4121-token case measured 155.50 prefill and 13.97 decode token/s, but the planned matched KVMem cases were not run.

## Reproduction and evidence

```powershell
./scripts/windows/build-prism-reference.ps1
python scripts/prism-benchmark.py --profile original-32k --out logs/prism-original-32k `
  --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" `
  --gpu GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29
```

The runner reuses `logs/bonsai-32k-smoke/archive-request.json` from the previous long-context test. Raw results, request command, memory samples, WDDM snapshot and library hashes are under `logs/prism-original-32k/`. Source archive and build are under `build-win-prism-reference/`; build output is in `logs/prism-reference-build.log`. The reference service was stopped after testing.
