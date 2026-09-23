# Prism patch replay

`llama-kvmem-current.patch` targets Prism commit `9a9394a895b96003ca842a6041cb28ac49a108f7`.
It includes the KVMem adapter, rc3 server/template fixes, snapshot MTP restoration,
PTQ1 small-batch CUDA optimization and the required PDL synchronization fix.
The submodule stays pinned to upstream; commit changes as this cumulative patch,
not an unpublished submodule commit. GDN Record/Fold is not included.

Run `scripts/apply-patches.sh` after initializing a fresh submodule. The Windows
build script performs the same full-patch checks and accepts an already patched tree.
An older partially patched working tree must first have its old patch reversed;
preserve any local changes before updating. Do not stack a new cumulative patch
on the old cumulative patch.

## Kernel provenance

The kernel changes originate from [Prism PR #218](https://github.com/PrismML-Eng/llama.cpp/pull/218),
including SM120 compilation, shared-memory guards and four-column dispatch fixes
through `285542d98d37d0f07f491cd206aefa31f1848f33`. Imported patch revisions and hashes
are recorded in `prism-bonsai-ptq1-kernels.json`.

`prism-bonsai-ptq1-kernels.patch` and `prism-bonsai-ptq1-pdl-sync.patch` are retained
for attribution and independent reproduction. **Both are already included in the
cumulative patch; do not apply them again.** The local PDL fix waits for activation
quantization before the new PTQ1 mat-vec reads its input on SM90+.
See [measurements and numerical checks](../docs/bonsai-kernel-comparison.md).

`python scripts/prepare-bonsai-kernel-build.py` snapshots the already integrated
working sources to `build-win-bonsai-kernel/source` and adds the upstream numerical
test target. It checks cumulative-patch presence and refuses to overwrite snapshots.

`prism-bonsai-mtp-embedding.patch` is only for the independent Prism comparison;
do not apply it on top of the cumulative patch. `build-prism-reference.ps1` builds
its own pristine GGML so that the reference never reuses optimized KVMem libraries.

`llama-kvmem-rc3-reference.patch`, numbered patches and `*-upgrade.patch` target
older baselines; do not apply them to this Prism checkout.
