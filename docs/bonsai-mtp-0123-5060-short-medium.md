# Bonsai MTP 0/1/2/3 on RTX 5060 Ti: short/medium context

Tested September 23, 2026, on the local `kvmem-bonsai-llama.cpp` branch.
All four modes use the same PTQ1 trunk with the r3 Q4_0 MTP head (F32 norms),
the same server executable, target/draft K/V Q8_0, context limit 131072,
KVMem budget 24576 and generation reserve 10240. The default 128-token pinned
prefix is unchanged. Batch/ubatch 128, CPU threads/batch threads 4, FlashAttention
on, full GPU offload. MTP uses GPU recurrent snapshots and p_min=0.

The actual prompts contain 4155 or 16226 tokens. Neither workload reaches the
24K selection budget; no pressure eviction occurs. These measurements isolate
ordinary prefill/generation performance rather than sparse long-context recall.

Each mode first generates 128 warmup tokens, then runs two trials per prompt
length with 512 output tokens each. Prompts contain a synthetic archive followed
by a request for an English guide to astronomical observatories. Trial markers
differ to avoid substantial prefix-cache reuse; corresponding trials use
identical prompts across all four modes. Cache hits are 2 tokens in each mode's
first short trial, and 0 otherwise. Temperature 0, seed 42, presence/frequency
penalties 0, thinking disabled. The server reasoning budget remains 4096 but
does not apply to these non-thinking requests.

## Results

Throughputs aggregate token counts over measured time across two trials.
Acceptance is accepted draft tokens / proposed draft tokens, excluding bonus
tokens. Timings include the server's real generation path, not a kernel-only
microbenchmark. KVMEM_TRACE is enabled equally in all modes for timing and
acceptance counters; the expensive all-layer KV audit is disabled.

| Input | MTP max draft | Prefill tok/s | Decode tok/s | Decode gain vs off | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| 4155 | Off | 408.37 | 46.90 | — | — |
| 4155 | 1 | 400.30 | 52.99 | +13.0% | 74.7% |
| 4155 | 2 | 399.95 | 52.15 | +11.2% | 56.3% |
| 4155 | 3 | 398.13 | 47.98 | +2.3% | 41.8% |
| 16226 | Off | 410.71 | 40.55 | — | — |
| 16226 | 1 | 401.32 | 43.43 | +7.1% | 70.1% |
| 16226 | 2 | 399.36 | 44.59 | +10.0% | 54.5% |
| 16226 | 3 | 400.68 | 43.20 | +6.5% | 40.5% |

| MTP max draft | Mean request wall, 4155 + 512 tokens | Mean request wall, 16226 + 512 tokens | Sampled device VRAM peak |
|---|---:|---:|---:|
| Off | 21.22 s | 52.30 s | 7208 MiB |
| 1 | 20.17 s | 52.39 s | 7834 MiB |
| 2 | 20.34 s | 52.28 s | 7984 MiB |
| 3 | 21.21 s | 52.52 s | 8134 MiB |

MTP1 gives the best measured short-context decode result. MTP2 is about 2.7%
above MTP1 at 16K, with 150 MiB more device memory. MTP3 does not improve either
context over the best shorter draft length. Prefill is about 2–3% slower with
MTP; at 16K and only 512 output tokens, that offsets the decode benefit and
whole-request latency stays approximately unchanged. At 4K, MTP1 reduces
whole-request latency by about 5%.

The draft-length histogram confirms the requested lengths are actually used:
MTP2 proposes two tokens per normal verification and MTP3 proposes three;
one-token proposals occur only near the output cap. Average output tokens per
verification are about 1.75/2.12/2.25 at 4K and 1.70/2.09/2.21 at 16K for
MTP1/2/3. The modest additional committed tokens at MTP3 do not offset its
additional work in this workload. Host checkpoint restore counters remain zero;
this does not mean every draft was accepted, since rejected suffixes use GPU
recurrent snapshot rollback.

All 16 measured requests produce exactly 512 tokens and finish by the output
limit. No CUDA OOM or server crash. MTP3's 4K trials vary more (49.44 and 46.60
tok/s); two trials are insufficient to establish a universal ranking. Outputs
are not byte-identical between all modes even at temperature zero, so this is a
same-prompt service comparison, not a forced-identical-completion benchmark or
a proof of speculative decoding equivalence. It covers one English prose task,
not code, Chinese, thinking-enabled workloads, or long-context retrieval.

Retain MTP1 as the balanced default for now; MTP2 is worth trying for longer
generation contexts. These runs alone do not justify increasing the default to
3. No launcher default was changed.

## Reproduce and evidence

```powershell
python scripts/bonsai-mtp-context-bench.py --binary build-win-bonsai-release/bin/llama-kvmem-server.exe --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf" --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c --out logs/bonsai-mtp-0123-rerun
```

Raw responses, commands, timings, per-request acceptance counts, prompts and
GPU telemetry: `logs/bonsai-mtp-0123-short-medium-5060ti/`.
Aggregate measurements: `summary.json` in that directory.

Binary SHA256: `0256561fd7df564fd4a1cbfee50609d6c978ff9ce8fc6a80d6034acc0274a667`.
Model SHA256: `4b0e0f9b252d1b77ace4e197f913dbb8729a13c2a0174d4eb8b2b80c22918832`.
GPU UUID: `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`.
