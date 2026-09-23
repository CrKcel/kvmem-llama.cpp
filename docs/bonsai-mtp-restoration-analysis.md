# Bonsai MTP restoration analysis

Date: 2026-09-22. Reviewed branch: `kvmem-bonsai-llama.cpp`, HEAD `c7485c2`.
Prism pin: `9a9394a895b96003ca842a6041cb28ac49a108f7`.
This records the pre-implementation assessment. For the subsequent local snapshot
implementation and measured results, see [Bonsai MTP validation](bonsai-mtp-validation.md).
The existing uncommitted launcher log-forwarding change was left intact.

## Model candidates

1. [ProCreations trained Bonsai MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP):
   r3-mtp checkpoint, BF16 head in `model_mtp.safetensors` (about 849 MB),
   plus a combined PQ2 target/Q8 head GGUF (about 7.66 GB). Independent community
   work, not an official Prism release. Preferred trained-head candidate; for
   our 8GB goal, extract/convert its head and merge into PTQ1 instead of adopting
   the larger PQ2 trunk. That PTQ1 combination still needs independent validation.
2. [decent-jawfish graft](https://huggingface.co/decent-jawfish/bonsai-2-27b-mtp):
   PQ2 Bonsai plus an untrained/transplanted Qwen3.8 MTP block. Useful reference
   and alternative donor, with a minimal runtime embedding patch.
3. [sudoingX PTQ1 graft recipe](https://github.com/sudoingX/bonsai2-small-gpu/blob/main/graft/recipe.txt):
   explicitly starts from the exact PTQ1 SHA256 we downloaded. Its lean variant
   appends 15 head tensors without duplicating embedding weights: total file
   6,297,658,848 bytes versus our 5,946,648,928 bytes, an increase of about 351 MB
   (335 MiB). This is a local assembly recipe, not proof of a ready-made PTQ1
   download or a trained Bonsai head. Avoid its default larger embedding-copy
   variant for the small-VRAM experiment.

The graft tools preserve tensor byte spans. Before using them, audit custom
PTQ1/PQ2 handling, offsets/alignment, metadata, tensor dimensions and tokenizer
identity; verify all 851 target tensor payloads are unchanged. Append `blk.64.*`,
set block_count to 65 and nextn_predict_layers to 1, and preserve the original
Hadamard metadata. Do not mark unrotated donor-head tensors as Hadamard weights.
Do not requantize the ternary trunk via a generic quantizer. Preserve original
weights and write a separate derived GGUF with pinned donor provenance.

## Required code changes

| Area | Current state | Proposed change |
| --- | --- | --- |
| `llama.cpp/src/models/qwen35.cpp`, `graph_mtp` | Direct `ggml_get_rows` lacks the target's embedding inverse | Port the community Hadamard inverse after lookup, before normalization; apply only when that weight is in the inverse map, with rotation/sign order matching normal embedding lookup. Keep the graph verifier enabled. |
| `tools/kvmem-spec.cpp` | Startup returns false; both generation overloads throw | Recover the rc3 setup/generation logic from `a0365fd`, adapting to current Prism APIs. Initially exclude Record/Fold and use actual snapshot rollback. Preserve sampling verification, hidden rows, rejection, cancellation and state cleanup. |
| CLI/server entry points | MTP rejected; n_rs_seq forced to zero | Restore opt-in `draft-mtp`, model-head validation, n_outputs_max/per_seq and rollback depth derived from draft count. Retain a working no-MTP fallback. |
| `src/adapter/llama-memory-kvmem.cpp` factory | MTP context throws | Restore the target lookup through ctx_other and instantiate the existing MTP follower. Restore truthful recurrent/rollback allocation reporting. |
| `src/adapter/llama-memory-kvmem-hybrid.cpp` | Recurrent constructor receives literal zero | Pass the correct snapshot depth. Audit suffix deletion and recurrent rollback separately from query-replay holes. Existing rollback code below the constructor is currently dormant. |
| `src/adapter/llama-memory-kvmem-mtp.cpp` | Compiled but not instantiated | Validate block-64 dimensions, shared weights, bounded KV allocation, packed K/V transfers and follower moves on retrieval/reuse. Keep the adapted Prism KV constructor. |
| `common/speculative.cpp`, batch/graph/capture hooks | Several required hooks already retained | Audit ctx_other, embd_nextn, target hidden-row selection and accepted-token alignment. Do not blindly reapply the entire rc3 patch or duplicate existing hooks. |
| CMake/tests/launcher | MTP tests absent from current CMake; launcher hardcodes none | Re-enable relevant existing tests in stages; expose an optional MTP mode, draft count and draft KV type. Preserve current no-MTP defaults until validated. |

Community embedding fix:
[minimal patch](https://huggingface.co/decent-jawfish/bonsai-2-27b-mtp/blob/main/0001-qwen35-mtp-hadamard-inverse.patch).
It addresses the model graph, not KVMem's memory policy or speculative state lifecycle.

## State handling and memory

Snapshot mode is the shortest route to correctness because pinned Prism already
implements recurrent snapshots and partial rollback. It does not require the
custom rc3 GGML RECORD operation. Use draft=1 initially, then measure draft=2.
The target and follower must see identical accepted logical positions after
zero, partial and full acceptance, cancellation and subsequent requests.

Measured no-MTP GDN allocation in `logs/bonsai-64k-24k-10k/kvmem.log`:
150,994,944 bytes of recurrent state plus 5,898,240 bytes of convolution state.
At the same single-sequence allocation, each additional snapshot plane costs
about 149.625 MiB (`n_rows = mem_size * (1 + n_rs_seq)`). Draft=1/2 therefore
adds approximately 150/299 MiB of recurrent storage, excluding graph buffers.

For a one-layer MTP head with the same 4 KV heads and 256-dimensional keys/values,
a 34,816-cell follower needs approximately 72.25 MiB at Q8/Q8, or 136 MiB at F16/F16.
Confirm the donor dimensions before applying these estimates. If the follower
accidentally allocates the full 131,072 logical context, those numbers become
272/512 MiB; bounded follower allocation is important.

The lean donor adds approximately 335 MiB of weight bytes; the trained Q8 head
is roughly 430 MiB. Together with one snapshot and Q8 follower this is roughly
557-652 MiB extra before additional compute buffers. Our previous 64K no-MTP
peak was 7,454 MiB out of 8,151 MiB: this leaves very little room. These are
budget estimates using the old 64K baseline, not a measured 128K MTP peak.
PQ2 combined models increase trunk memory further and are poor first choices
for full-GPU 8GB serving. Prototype on the available 16GB GPU, then tune 8GB.

If snapshots cannot meet the 8GB target, port rc3 Record/Fold as a separate
second phase. This crosses recurrent memory, delta-net graph construction,
GGML op definitions/dispatch and CUDA record/fold kernels. Prism raw-gate and
row-indexed-state paths must not be fed into the old record representation
without adaptation. Initially disable those optimizations specifically on
ReplaySSM, preserve ordinary Prism execution, and validate accepted-prefix
state plus convolution history. See `prism-compatibility-analysis.md`.

## Performance and numerical risks

PTQ1 multi-token verification can be expensive enough to erase acceptance gains.
[Community PTQ1 measurements](https://github.com/sudoingX/bonsai2-small-gpu/blob/main/results/KERNEL_REPORT.md)
separate a faster mat-vec kernel from MTP gains and report batch-size-dependent
floating-point output changes. The tested batch-invariance option has a speed
tradeoff, particularly at longer contexts. Those RTX 3060 results and older
Prism baseline cannot predict our RTX 5050/5060 Ti performance.

Keep Hadamard correctness, state rollback and optional small-batch kernel tuning
as separate changes. Verify greedy output and teacher-forced logits; distinguish
numerical near-tie changes from state corruption. For sampled output, inspect
the verifier/sampling semantics as well as acceptance. Do not label the port
lossless based only on coherent output or one matching greedy sample.

## Implementation gates

1. Pin head/patch revisions, inspect metadata and construct a separate PTQ1+head
   file. Compare all trunk tensor payloads and no-MTP output/logits.
2. Validate the model with plain Prism plus the embedding patch, initially on
   16GB with short actual input. This separates donor/runtime issues from KVMem.
3. Restore KVMem snapshot MTP and its existing MTP KV test. Exercise zero/partial/
   full acceptance, draft=1/2, cancellation and 4096-token thinking-budget endings.
4. Cross the GPU pool boundary and test eviction, restoration, query replay,
   prefix reuse and continued decode. Recheck no-MTP behavior.
5. Benchmark identical prompts/model/hardware with MTP off/on, first 4K then
   actual 32K/64K histories. Record wall-clock prefill/decode, accepted/proposed
   drafts, verification/fold time and peak VRAM. Configured 128K is not proof of
   an actual 128K test. Retest on 8GB and reduce KV/draft depth if necessary.
6. Only if required, implement Record/Fold and optional PTQ1 kernel changes,
   with their own state-equivalence and CUDA backend tests before packaging.

Assessment: medium effort for a snapshot-based functional prototype; medium-high
for reliable KVMem retrieval plus MTP; high for an optimized 8GB/large-pool port
requiring Record/Fold. The small public patch solves only one of these layers.
No model weights downloaded, runtime modified, build performed or service restarted
for this analysis. A prior Windows application-control block affected the new
multiarch binary; actual validation will need a runnable build environment.
