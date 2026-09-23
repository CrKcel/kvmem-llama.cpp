# Bonsai PTQ1 CUDA kernel comparison

Local experiment on September 23, 2026. This does not replace the packaged release.

## Sources and build

The measured baseline was the pre-integration KVMem/MTP working tree on branch
`kvmem-bonsai-llama.cpp`, with Prism pinned to
`9a9394a895b96003ca842a6041cb28ac49a108f7`.
The optimized build adds the community PTQ1 CUDA changes from
[Prism PR #218](https://github.com/PrismML-Eng/llama.cpp/pull/218), including fixes
through `285542d98d37d0f07f491cd206aefa31f1848f33`.
Patch provenance and hashes are in `patches/prism-bonsai-ptq1-kernels.json`.

The kernel patch changes eight CUDA/test files. It was initially tested in an
isolated source tree, and is now part of the cumulative rc3-prism.2 patch. The
original baseline executable remains available locally. The source snapshot and its hashes
are under `build-win-bonsai-kernel/`; the snapshot also exposes the existing
upstream `test-backend-ops` target for numerical validation.

Build configuration: MSVC, CUDA 12.9.86, SM120a, static libraries, all FlashAttention
KV quantizations, same options as the baseline. This local build targets the two
Blackwell GPUs being measured; it is not the 30/40/50-series distribution package.

To prepare a fresh snapshot (the preparation script refuses to overwrite one):

```powershell
python scripts/prepare-bonsai-kernel-build.py
./scripts/windows/build.ps1 -SourceDir ./build-win-bonsai-kernel/source -BuildDir "$PWD/build-win-bonsai-kernel/build" -CudaPath 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9' -ExperimentalCuda129 -CudaArchitectures '120a-real' -Jobs 4
```

## Method

Both backends use the same PTQ1 model with the community r3 Q8 MTP head appended:
`Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf`.
Model SHA256: `d020556899ccb7879bab6c2356cd6e633e14ddf12a4fb42eb7820b24d21220d9`.

- Context 8192; target KV budget 2048 plus generation reserve 1024.
- Q8 target/draft KV; batch and ubatch 128; GPU offload 99; FlashAttention on.
- MTP off, draft=1 and draft=2; GPU GDN snapshots; temperature 0 and seed 42.
- `GGML_CUDA_BATCH_INVARIANT` unset in all comparison processes.
- Each case runs four 128-token prose requests. The first is warmup; report the
  median and range of the remaining three server decode timings.
- Each case also checks arithmetic, Chinese translation, multi-turn recall,
  repeated output, output limits, a 4108-token archive exceeding the 3072-cell
  pool, archive recall, a thinking budget, and cancellation/recovery.
- Compare backends consecutively for each draft length, reversing order for
  draft=1. GPU memory/utilization are checked before each case. GPU telemetry and
  exact commands are retained in the result JSON. No other GPU service is stopped.

Reproduce on one idle GPU:

```powershell
python scripts/bonsai-kernel-compare.py --baseline build-win-bonsai/bin/llama-kvmem-server.exe --optimized build-win-bonsai-kernel/build/bin/llama-kvmem-server.exe --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf" --gpu GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29 --out logs/bonsai-kernel-compare-5050
```

These short-prompt decode measurements and 4K archive checks do not establish
performance or stability at 32K/64K/128K actual input lengths or at the default
24K + 10K pools. Repeated text within a case is a determinism check, not a broad
quality benchmark or a proof of bitwise equivalence between backends.

## Execution result

The user disabled Smart App Control manually, after which the same binaries
started normally. The previous WinError 4551 was a Windows application-control block.
The optimized build passed all 11 CTest checks. PTQ1 MUL_MAT and fused mat-vec
numerical tests passed 85/85 against CPU on each GPU, including the shared-memory
boundary cases.

The previously blocked MTP KV suite also passed on the final optimized build on
RTX 5060 Ti: ten target/draft KV combinations passed GPU save/restore, cyclic
overwrite and partial-block checks; three replay cases (Q8/Q8, Q5/Q5 and Q8/Q4)
matched both repeated replay and single-pass logits exactly in this fixture.
Logs: `build-win-bonsai-kernel/mtp-kv-pdlfix-5060.log` and
`build-win-bonsai-kernel/final-build-test.log`.

### Required PDL synchronization fix

The first model-level run exposed a real defect in the community kernel on
SM120a: four identical greedy prose requests produced different text. Functional
answers and isolated numerical tests still passed, so those checks alone did not
catch it. The initial timing data is excluded from the comparison below.

The new `mul_mat_vec_ptq1_0_pt` uses the generic PDL-enabled launcher, but did not
call `ggml_cuda_pdl_sync()` before reading activations. Its producer,
`quantize_q8_1`, signals launch completion early with `ggml_cuda_pdl_lc()`.
The consumer could therefore read data before the quantizer had finished.
[NVIDIA requires the dependent kernel to synchronize before consuming results](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

Diagnosis: setting `GGML_CUDA_PDL=0` on the unmodified experimental binary restored
identical output across four requests. The local fix adds the dependency wait at
entry to the new kernel. With that fix, PDL remains enabled by default and every
comparison case below passed the repeated-output check. This PDL path is relevant
to SM90+; an RTX 3060 test does not exercise it.

`patches/prism-bonsai-ptq1-pdl-sync.patch` contains the local fix, separate from
the community patch; the preparation script applies both. The original kernel
binary is retained under `build-win-bonsai-kernel/unfixed/` for reproduction only.
Diagnostic logs: `logs/bonsai-kernel-compare-5050/` (failed determinism) and
`logs/bonsai-kernel-pdl-off-5050/` (PDL disabled, passed).

### Paired measurements after the fix

All 12 cases passed every functional check. Each generated four 128-token prose
responses; the first was warmup and the following three supply the timing range
and median. Text is identical within each case. Different backends and draft modes
can produce different greedy prose; this is not cross-backend bitwise equivalence
or a broad quality evaluation.

| GPU | Draft | Original tok/s | Optimized tok/s | Same-mode gain | Extra MTP gain over optimized off |
|---|---:|---:|---:|---:|---:|
| RTX 5050 | 0 | 13.26 | 20.42 | +54.1% | +0.0% |
| RTX 5050 | 1 | 13.61 | 23.43 | +72.1% | +14.7% |
| RTX 5050 | 2 | 11.38 | 21.47 | +88.6% | +5.1% |
| RTX 5060 | 0 | 38.79 | 46.04 | +18.7% | +0.0% |
| RTX 5060 | 1 | 35.29 | 54.52 | +54.5% | +18.4% |
| RTX 5060 | 2 | 28.81 | 48.65 | +68.8% | +5.7% |

| GPU | Backend | Draft | Decode range tok/s | 4K archive prefill tok/s | Peak device MiB | Prose acceptance |
|---|---|---:|---:|---:|---:|---|
| RTX 5050 | baseline | 0 | 13.22-13.39 | 136.64 | 6218 | n/a |
| RTX 5050 | optimized | 0 | 20.31-20.51 | 136.28 | 6224 | n/a |
| RTX 5050 | baseline | 1 | 13.37-13.68 | 134.60 | 6891 | 64.1% |
| RTX 5050 | optimized | 1 | 23.36-23.59 | 134.11 | 6883 | 68.4% |
| RTX 5050 | baseline | 2 | 11.33-11.40 | 134.26 | 6997 | 47.0% |
| RTX 5050 | optimized | 2 | 21.34-21.47 | 134.24 | 6980 | 51.6% |
| RTX 5060 | baseline | 0 | 38.58-38.81 | 356.70 | 6120 | n/a |
| RTX 5060 | optimized | 0 | 45.47-46.15 | 359.52 | 6120 | n/a |
| RTX 5060 | baseline | 1 | 34.71-35.40 | 351.11 | 6708 | 73.0% |
| RTX 5060 | optimized | 1 | 54.30-55.00 | 350.87 | 6708 | 73.0% |
| RTX 5060 | baseline | 2 | 28.59-29.08 | 350.61 | 6858 | 51.6% |
| RTX 5060 | optimized | 2 | 48.00-49.65 | 351.20 | 6858 | 51.6% |

5050 is the 8GB Laptop model; 5060 is the 16GB Ti. Device memory is the peak over
the whole integration run and includes any display/background allocations.
The fixed model has an appended MTP head in all cases, with loading controlled
by the draft mode. The 4K archive prefill is a single sample per case.

Draft=1 is fastest in this measured workload on both GPUs. It adds about 15-18%
over the optimized backend with MTP off, at roughly 0.6 GiB additional device
memory. Draft=2 is slower than draft=1 and uses more memory. Prefill is effectively
unchanged; these gains concern decode. Keeping MTP off still benefits from the
new PTQ1 kernel without the head/snapshot memory overhead.

Complete commands, binary SHA256 hashes, GPU telemetry, responses and timings:

- `logs/bonsai-kernel-compare-pdlfix-5050/`
- `logs/bonsai-kernel-compare-pdlfix-5060/`
- `build-win-bonsai-kernel/validation-summary.json`

### Local startup

The measurements below used the isolated build. The rc3-prism.2 branch now
includes these kernels in its normal build. To reproduce the isolated run:

```powershell
./scripts/windows/start-bonsai.ps1 -BuildDir ./build-win-bonsai-kernel/build -Gpu 0 -Mtp -DraftTokens 1 -Context 8192 -Budget 2048 -Reserve 1024 -ReasoningBudget 256
```

Use `-NoMtp` for the original PTQ1 model without the appended head; that is a
different model file from the controlled comparison above. These measurements
precede release packaging; they do not certify a later multiarch release binary.
