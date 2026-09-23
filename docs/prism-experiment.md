# rc3 Prism baseline experiment

Historical initial validation: no-MTP adapter port implemented; Windows CUDA build, 11 tests and initial 4K text tests pass. On 8GB, 32K with a 2K retrieval budget recalls 1/3 codes; 64K with a 24K retrieval budget plus 10K reserve recalls 3/3 at a 7454 MiB peak. These are synthetic tests. For current release preparation, see [rc3-prism.2 notes](milestones/v0.16.0-rc3-prism.2.md). See [validation results](bonsai-validation.md).

September 23 update: snapshot MTP is now implemented and tested separately
with the community r3 head. Its smaller-pool results and limitations are in
[Bonsai MTP validation](bonsai-mtp-validation.md); the 64K measurements above remain no-MTP results.

## Reproducible baseline

- Branch: `kvmem-bonsai-llama.cpp`
- KVMem parent: `1734a2809bb0422da842d03a4734771ad9439ade`
- Preserved rc3 snapshot commit: `a0365fd`
- Snapshot: `rc3-cuda129-stage/source-manifest.json`, version `0.16.0-rc3`
- Snapshot manifest SHA256: `e8cde0b6b8c8396f501d426fd885138e8b974fa8005830f6e8dfb12acb928b72`
- All 190 non-submodule snapshot files were checked against the manifest before copying.
- Previous llama.cpp pin: `b81c99b479d4c24e5eeca10de99032ebd343ef8f`
- New repository: https://github.com/PrismML-Eng/llama.cpp
- New pin: `9a9394a895b96003ca842a6041cb28ac49a108f7` (resolved from `prism` on 2026-09-20)

The gitlink pins the exact revision. Do not use `git submodule update --remote` for reproducible builds.

## Implemented scope

- KVMem memory factory, Q/K/V capture, logical token positions, sparse KV selection, host spill and stage-in.
- Prism's native recurrent state and host checkpoints for KVMem query replay.
- No rollback planes when MTP is off (`n_rs_seq=0`). Opt-in `draft-mtp` uses a follower KV pool and GPU snapshot planes. GDN Record/Fold buffers and kernels are not included.
- PTQ1 CUDA small-batch kernels include the community optimization and local PDL fix. Weight formats, Hadamard metadata and GDN computations remain unchanged.
- KV constructor and mtmd helper calls adapted to the pinned Prism interfaces. Vision is not validated by this experiment.

`patches/llama-kvmem-current.patch` is the cumulative Prism integration patch (including MTP and CUDA kernels). The old rc3 cumulative patch is retained as `llama-kvmem-rc3-reference.patch`. Historical numbered patches must not be applied to this baseline. Patch replay was checked on a fresh export of the pinned revision, including repeat application.

## Build and run

From this worktree, with Visual Studio C++ Build Tools and CUDA 12.9.86 installed:

```powershell
./scripts/windows/build.ps1 -BuildDir "$PWD/build-win-bonsai" `
  -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9' `
  -ExperimentalCuda129 -CudaArchitectures '120a-real' -Jobs 4
./scripts/windows/start-bonsai.ps1 -Gpu 0
```

The measured build targets SM120a (RTX 50 series); use suitable architecture flags and revalidate for other cards. CUDA runtime DLLs must be on PATH. The launcher accepts a GPU UUID to avoid index ambiguity.

The launcher uses the downloaded model under `%LOCALAPPDATA%/KVMem/models`, a 131072-token logical context, 24576-token retrieval budget, 10240-token generation reserve, Q8 KV, batch/ubatch 128, thinking enabled with a 4096-token reasoning budget, MTP draft=1 and no projector. It selects the merged r3 model by default; use `-NoMtp` for the original PTQ1 model. It listens on localhost port 18202. The later [128K MTP1 test](bonsai-128k-mtp1-validation.md) completed without OOM but recalled only1/3 codes. It is not a passing long-context quality validation. The earlier 32K test used the old 2048+1024 pool and recalled one of three codes; the 64K test used 24576+10240 and recalled all three. See the measured results for their precise scope.

```powershell
python scripts/bonsai-smoke.py --long `
  --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" `
  --gpu GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29
```

Replace the UUID for another machine. Results and server traces are written under `logs/bonsai-smoke`. Existing installed runtime files are not replaced. Host-only configuration remains available with `-DKVMEM_BUILD_LLAMA=OFF`.

## Remaining validation

Compare logits against an independently built, unmodified Prism runtime; extend long-context and tool-use coverage; run ordinary Qwen regressions and validate other GPUs/platforms. MTP's separate validation boundaries are documented above. The earlier compatibility analyses are planning records, not test results.
