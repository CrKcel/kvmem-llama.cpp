# Prism patch replay

`llama-kvmem-current.patch` targets Prism commit `9a9394a895b96003ca842a6041cb28ac49a108f7`.
It adds the KVMem memory factory, capture hooks, logical positions, sparse KV handling and rc3 server/template fixes.
It does not add GDN Record/Fold or modify Prism's quantization and Hadamard kernels.

Run `scripts/apply-patches.sh` after `git submodule update --init`.
The script checks the full patch before changing files and accepts an already patched tree.
The Windows build script performs the same checks.

`llama-kvmem-rc3-reference.patch` preserves the original rc3 patch for reference.
The numbered patches and `*-upgrade.patch` files also target earlier baselines; do not apply them to Prism.

The submodule stays pinned to the upstream Prism commit. Commit adapter changes as a regenerated patch here, not as an unpublished submodule commit.
