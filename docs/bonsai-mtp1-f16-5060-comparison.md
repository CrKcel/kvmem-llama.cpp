# Bonsai MTP1: F16 versus Q8 draft KV on RTX 5060 Ti

Tested September 23, 2026. Only the MTP draft K/V type changes via
`--spec-kv-dtype f16` or `--spec-kv-dtype q8_0`. Target K/V remains Q8_0,
the model remains the PTQ1 trunk with r3 Q4_0 MTP head, and MTP max draft is 1.
The measurements below were taken with explicit overrides before changing defaults.
After this comparison, the user selected F16 as the default for the Bonsai
server, CLI, and launcher. Target KV remains Q8_0. The launcher accepts
`-DraftKvType` for explicit overrides.

Both runs use context limit 131072, KVMem budget 24576 + generation reserve
10240, default 128-token prefix, batch/ubatch 128, four CPU threads, full GPU
offload, FlashAttention, GPU recurrent snapshots, and draft p_min=0.
Thinking is disabled per request; temperature 0 and seed 42.
KVMEM_TRACE is enabled; the all-layer retrieval audit is disabled.

The existing benchmark supplies identical prompts of 4155 and 16226 tokens,
two trials at each length, and exactly 512 output tokens per trial after a
128-token warmup. Each configuration starts a fresh server. Neither length
triggers eviction. This tests ordinary prefill/decode, not 128K retrieval.
Q8 was rerun immediately after F16 to check the unexpected device-memory
result instead of relying only on the earlier MTP 0/1/2/3 benchmark.

## Results

Rates aggregate tokens over measured time across both trials. Acceptance is
accepted draft tokens / proposed draft tokens, excluding bonus tokens.

| Input | Draft KV | Prefill tok/s | Decode tok/s | Acceptance | Mean request wall |
|---|---|---:|---:|---:|---:|
| 4155 | Q8_0 | 399.44 | 52.54 | 74.74% | 20.28 s |
| 4155 | F16 | 399.00 | 54.65 | 75.04% | 19.93 s |
| 16226 | Q8_0 | 400.11 | 44.72 | 70.10% | 52.18 s |
| 16226 | F16 | 399.00 | 43.70 | 70.38% | 52.58 s |

F16 decode differs by +4.02% at 4K and -2.27% at 16K. Prefill changes by
-0.11% and -0.28%. There is no consistent speed improvement. The previous
Q8 MTP1 results were 52.99/43.43 tok/s, versus 52.54/44.72 in this repeat;
the small differences should not be treated as a universal ranking.
Per-trial decode ranges are 52.02–53.07 (Q8 4K), 53.03–56.38 (F16 4K),
44.16–45.29 (Q8 16K), and 43.29–44.13 (F16 16K).

Sampled total-device memory peaks are **7834 MiB with Q8 versus 7766 MiB
with F16**, a 68 MiB reduction, at both lengths. The Q8 peak matches the
earlier benchmark. Sampling occurs approximately once per second and may
miss short-lived peaks; this is not a per-allocation profiler.

All eight requests completed with 512 output tokens and no OOM or crash.
The four corresponding output texts match byte-for-byte across KV types.
Each F16 pair accepts just one more draft token than its Q8 pair, raising
acceptance by about 0.3 percentage points. Only one English prose workload,
thinking disabled, and two trials per configuration are covered. These
results do not explain the much larger IQ3 MTP relative speedup, and do not
justify a default change based on speed alone. The subsequent F16 default
was selected for the measured reduction in total device memory.

## Memory interpretation

Startup logs confirm the target KV allocation is unchanged at 1212153856
bytes. The single-layer draft KV pool has 34816 cells in both runs:

| Draft KV | Draft pool bytes | Draft pool MiB |
|---|---:|---:|
| Q8_0 | 75759616 | 72.25 |
| F16 | 142606336 | 136.00 |

F16 increases the actual draft cache by 63.75 MiB. Total device usage also
includes compute buffers and allocator reservations, so it need not rise by
the same amount. In this source, CUDA FlashAttention can reserve F16 K/V
conversion buffers for non-F16 inputs (`ggml-cuda/fattn-common.cuh`,
`ggml_cuda_flash_attn_ext_get_f16_extra_data`), while native F16 avoids those
buffers. Kernel selection also depends on KV type and tensor shape
(`ggml-cuda/fattn.cu`). This is a plausible explanation for lower total device
usage with F16 here; these runs do not instrument each allocator or kernel,
so the exact allocation difference is not proven.

## Reproduction and evidence

After changing the default, server and CLI rebuilt successfully. The existing
11 CTest checks and Windows launcher dry-run checks passed. A new RTX 5060 Ti
smoke run omitted `--spec-kv-dtype`, confirmed `type_k=f16 type_v=f16` in the
draft-pool log with target K/V Q8, and passed arithmetic, Chinese translation,
and multi-turn identifier recall. It used MTP1 with the same 128K context
limit and 24K+10K pool; evidence is in `logs/bonsai-f16-default-smoke/`.
This short validation is separate from the speed comparison above; the
SHA256 values below identify the pre-change benchmark executable/model.

Run `scripts/bonsai-mtp-context-bench.py` with the same binary/model/GPU
arguments as `bonsai-mtp-0123-5060-short-medium.md`, adding
`--drafts 1 --draft-kv f16` (or `q8_0`) and a fresh output directory.

Raw commands, startup KV type assertions, outputs, timings, acceptance counts,
and one-second device-memory telemetry:

- `logs/bonsai-mtp1-f16-short-medium-5060ti/`
- `logs/bonsai-mtp1-q8-confirm-short-medium-5060ti/`

Binary SHA256: `0256561fd7df564fd4a1cbfee50609d6c978ff9ce8fc6a80d6034acc0274a667`.
Model SHA256: `4b0e0f9b252d1b77ace4e197f913dbb8729a13c2a0174d4eb8b2b80c22918832`.
GPU UUID: `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`.
