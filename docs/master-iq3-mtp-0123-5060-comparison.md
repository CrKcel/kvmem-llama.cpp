# Master IQ3 vs Bonsai: MTP 0/1/2/3 on RTX 5060 Ti

Tested September 23, 2026. The IQ3 executable uses the fetched GitHub master
snapshot `7eb4d52ef90f9cf8412f9ee6e1fb72f82cd848de`, with upstream llama.cpp pin
`b81c99b479d4c24e5eeca10de99032ebd343ef8f` and master's maintained patch.
It is not the older local branch named master or the Bonsai/Prism executable.

## Build provenance

The master tree was exported into `../master-iq3-mtp-5060/source`. Its build
inputs were compared against the frozen CUDA 12.9 release source. All 3579
relevant frozen inputs (including patched upstream files) matched their recorded
SHA256 manifest; master's upstream patch was unchanged. The changed KVMem/MTP
adapter translation units, store implementation and server were recompiled.
Unchanged libraries from `release-rc3-cuda129/build` were reused, with hashes
recorded. The store/runtime tests were rebuilt and both passed. All project
source changes affecting the server were covered; the CMake delta only affects
the standalone MTP KV test target. The existing Bonsai checkout was preserved.

Both compared builds use CUDA 12.9.86 and native SM120a code on the 5060 Ti.
The master release libraries also include older GPU targets, which are not used
on this GPU. Full compile commands, input verification and library hashes are
recorded in `../master-iq3-mtp-5060/build-provenance.json` and `build.cmd`.

## Matched protocol

The same benchmark script and byte-identical prompts as the Bonsai test are
used. Actual input lengths match exactly: 4155 and 16226 tokens. Each mode has
one 128-token warmup and two measured 512-token generations per context.

- Context limit 131072; KVMem budget 24576 + reserve 10240; default prefix128.
- Target and draft K/V Q8_0; full GPU offload; FlashAttention; batch/ubatch128;
  CPU threads/batch threads4; snapshot MTP; draft p_min=0.
- Temperature0, seed42, presence/frequency penalties0, thinking disabled.
- The same synthetic archive and English observatory prose task. Trial markers
  differ to avoid substantial prompt reuse; corresponding trials match across
  models. Cache hits: 2 tokens in each mode's first short trial, 0 otherwise.
- Trace enabled for counters; expensive packed-KV audit disabled. No eviction
  occurs because input and output fit within the selection budget.

Master defaults its draft KV to F16, unlike the Bonsai inheritance fix. The
IQ3 commands explicitly use `--spec-kv-dtype q8_0`; startup logs confirm Q8/Q8.

The IQ3 model is `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`. Direct GGUF inspection
finds eight MTP matrix tensors in **Q6_K** and seven norm tensors in F32.
Bonsai uses the grafted community r3 **Q4_0** MTP matrices with F32 norms.
These are different checkpoints and heads as well as different runtime baselines;
the comparison cannot isolate the effect of trunk quantization or Prism alone.

## Decode results

Rates aggregate output tokens over measured generation time. Gains compare each
model to its own MTP-off baseline; they are not IQ3-versus-Bonsai absolute gains.

| Input | MTP | IQ3 decode tok/s | IQ3 gain | Bonsai decode tok/s | Bonsai gain |
|---|---:|---:|---:|---:|---:|
| 4155 | Off | 27.69 | — | 46.90 | — |
| 4155 | 1 | 35.53 | +28.3% | 52.99 | +13.0% |
| 4155 | 2 | 38.62 | +39.5% | 52.15 | +11.2% |
| 4155 | 3 | 38.14 | +37.7% | 47.98 | +2.3% |
| 16226 | Off | 25.81 | — | 40.55 | — |
| 16226 | 1 | 31.68 | +22.7% | 43.43 | +7.1% |
| 16226 | 2 | 35.19 | +36.3% | 44.59 | +10.0% |
| 16226 | 3 | 35.02 | +35.7% | 43.20 | +6.5% |

| MTP | IQ3 acceptance, 4K / 16K | Bonsai acceptance, 4K / 16K | IQ3 peak device memory |
|---|---:|---:|---:|
| Off | — | — | 12810 MiB |
| 1 | 77.2% / 79.6% | 74.7% / 70.1% | 13540 MiB |
| 2 | 62.1% / 61.6% | 56.3% / 54.5% | 13690 MiB |
| 3 | 50.2% / 50.2% | 41.8% / 40.5% | 13840 MiB |

Acceptance counts accepted draft tokens divided by proposed tokens, excluding
bonus tokens. Histograms verify that MTP2/3 normally propose two/three tokens;
shorter drafts appear near the output limit. Host-checkpoint restore counters
remain zero; rejected suffixes use GPU recurrent snapshots.

## Prefill and request latency

| MTP | IQ3 prefill tok/s, 4K / 16K | IQ3 request seconds, 4K / 16K | Bonsai request seconds, 4K / 16K |
|---|---:|---:|---:|
| Off | 723.20 / 732.05 | 24.35 / 42.16 | 21.22 / 52.30 |
| 1 | 697.47 / 699.22 | 20.49 / 39.55 | 20.17 / 52.39 |
| 2 | 696.25 / 698.75 | 19.33 / 37.93 | 20.34 / 52.28 |
| 3 | 697.86 / 697.56 | 19.50 / 38.05 | 21.21 / 52.52 |

For this 512-output workload, IQ3 MTP2 reduces whole-request time about 20.6%
at 4K and 10.0% at 16K. Bonsai remains faster during generation, while IQ3's
faster prefill can make its total request faster. These are distinct metrics.

## Interpretation and limits

The user's observation is confirmed for this workload: IQ3 gains substantially
more from MTP relative to its own baseline. A plausible contributor is that its
slower ordinary decode leaves more target-model work to amortize across accepted
draft tokens; the draft/verification overhead is proportionally more expensive
against Bonsai's already-fast baseline. IQ3 also has higher acceptance, especially
for longer drafts. These measurements do not separately establish the effects of
native kernel scaling, checkpoint/head quality, or head quantization.

MTP2 has the highest two-trial mean for IQ3 at both lengths; MTP3 is close and
uses another 150 MiB. The 16K MTP3 trials span 33.82–36.31 tok/s, so its small
mean difference from MTP2 is not a statistically established ranking. For these
results, MTP2 is the better provisional IQ3 tradeoff; retain the prior Bonsai
MTP1 recommendation. No defaults were changed.

All 16 measured IQ3 requests finish with exactly 512 output tokens and no OOM or
server crash. The two models generate different text, and outputs are not
byte-identical across all MTP modes even within a model. This is a same-prompt
service benchmark, not forced-identical output or a proof of MTP equivalence.
It does not test code, Chinese, thinking, long-context recall or a full 10K output.

## Reproduce / raw evidence

```powershell
python scripts/bonsai-mtp-context-bench.py --binary ../master-iq3-mtp-5060/bin/llama-kvmem-server.exe --model "$env:LOCALAPPDATA/KVMem/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c --draft-kv q8_0 --out logs/master-iq3-mtp-rerun
```

Raw results: `logs/master-iq3-mtp-0123-short-medium-5060ti/`, including
`summary.json`, `comparison-with-bonsai.json`, `iq3-head.json` and build provenance.

Master executable SHA256:
`42d6da3750f72125d124af9604589a630bd47dbfdfd1648b1d7b8e416d95a84d`.
IQ3 model SHA256:
`58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f`.
Bonsai baseline and hashes: `bonsai-mtp-0123-5060-short-medium.md`.
