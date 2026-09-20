# rc3 Prism baseline experiment

Status: source baseline switched; KVMem adapter migration pending. This branch is not a validated KVMem runtime release.

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

## Integration check

`git -C llama.cpp apply --check ../patches/llama-kvmem-current.patch` fails on these 10 files:

- `ggml/src/ggml-cpu/ops.cpp`
- `ggml/src/ggml-cuda/gated_delta_net.cu`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `src/llama-context.cpp`
- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`
- `src/llama-kv-cells.h`
- `src/llama-memory-recurrent.cpp`
- `src/llama-model.cpp`
- `src/models/delta-net-base.cpp`

The check did not modify the submodule. The original rc3 patches remain intact as migration inputs; they are not Prism-compatible patches. No rejected hunks or partial patch application are included.

## Next integration steps

1. Port the memory factory and Q/K/V capture hooks, then KV storage/retrieval.
2. Port ReplaySSM record/fold and the CPU/CUDA GDN interfaces.
3. Regenerate the maintained patch against the exact Prism pin and update patch replay instructions.
4. Compile and compare plain model logits with unmodified Prism, then test retrieval, recurrent replay and ordinary Qwen regressions.
5. Validate MTP separately before enabling it in this experimental runtime.

Top-level inference configuration stops when the KVMem factory header is absent. Host-only configuration remains available with `-DKVMEM_BUILD_LLAMA=OFF`. Unmodified Prism can be built independently from `llama.cpp`, but that does not provide KVMem features.

No model inference, CUDA build or runtime deployment has been performed for this baseline switch. Existing rc3 validation results do not validate this branch.
