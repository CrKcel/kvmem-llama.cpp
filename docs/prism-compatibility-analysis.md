# rc3 / Prism compatibility analysis

Scope update: the first Bonsai release disables MTP and omits Record/Fold. See [the current no-MTP analysis](bonsai-no-mtp-analysis.md). The analysis below preserves findings for the broader, original scope.

Date: 2026-09-20. Branch: `kvmem-bonsai-llama.cpp`.
Prism pin: `9a9394a895b96003ca842a6041cb28ac49a108f7`.

## Scope and evidence

This is a source review, not a compiled or model-validated port. The live submodule was not patched. A separate `prism-compat-audit` directory contains copies of the 38 affected files, `apply.log` and rejected hunks from a trial application.

The rc3 cumulative patch has 38 files and 107 hunks. Trial application accepted 91 hunks and rejected 16 hunks in 10 files; 28 files applied completely. Clean application is not proof of semantic compatibility.

## Conflict inventory and proposed changes

| File | Rejected hunks | Proposed modification | Risk |
| --- | ---: | --- | --- |
| `src/llama-context.cpp` | 1 | Restore the conditional KVMem hook include at the current include block. Preserve Prism graph verification and check capture reuse/harvest ordering. | Low textual; medium integration |
| `src/llama-model.cpp` | 1 | Add the factory include without restoring absent upstream headers. Factory insertion itself applies. | Low |
| `src/llama-kv-cache.h` | 1 | Declare `get_v_storage` and `seq_rm_logical` at their current API locations. | Low |
| `src/llama-kv-cells.h` | 1 | Add `logical_pos` to the x/y-only extension. Explicitly restore its -1 sentinel after `reset()` zeroes the structure. | Medium |
| `src/llama-kv-cache.cpp` | 1 | Populate logical position for text and embedding batches, preserving x/y for M-RoPE. Retain the hole-aware purge/mask changes that apply cleanly. | High correctness impact |
| `src/llama-memory-recurrent.cpp` | 4 | Allocate `(2 + replay ? extra-record-count : 0)` conceptually, expressed unambiguously as `2u + (replay ? REPLAY_COUNT : 0)`, tensors per layer. Add replay arrays/buffers around existing r/s allocations. Do not import rc3 PLE/p_l code absent from this baseline. Preserve lifecycle, poison, save/load and n_rs_seq behavior. | Medium-high |
| `ggml/src/ggml-cpu/ops.cpp` | 1 | Accept K=0 and allocate/use scratch for K!=1, keeping Prism raw-gate and row-indexed input support. Audit all output-state writes for K=0. | Medium |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 1 | Add RECORD to both execution and support dispatch; preserve rejection of row-indexed src[6] on CUDA. | Medium-high |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | 3 | Port K=0 output suppression and matching FP32 update helpers into the current multi-column kernel. Preserve RAW, G_PRECOMPUTED and PDL behavior. | High |
| `src/models/delta-net-base.cpp` | 2 | Separate ReplaySSM record/fold from ordinary snapshot/rows paths; derive replay K=0/1 and suppress fused state writes when recording. | High |

## Semantic risks beyond patch rejects

### RECORD and raw gates

Prism `ggml_gated_delta_net_set_raw_gates()` asserts `gdn->op == GGML_OP_GATED_DELTA_NET`. The rc3 patch changes K=0 nodes to `GGML_OP_GATED_DELTA_NET_RECORD`. Retaining the current raw-gates call on a recording node therefore trips that assertion when raw gates are active, despite the ggml.c patch applying cleanly.

The lowest-risk first implementation disables raw gates specifically on ReplaySSM paths and records activated g/b, matching the fold kernel contract. Do not merely relax the assertion: a fully fused implementation also needs a clear record representation and numerical agreement between verification and fold.

### Recurrent rows and snapshots

Prism passes optional `state_rows`, with state potentially represented as a 2D cache instead of a 4D gathered tensor. CUDA currently rejects this variant; CPU and Metal can use it. Keep the existing gathered 4D path for ReplaySSM and disable rows there explicitly. Ordinary non-replay Prism paths should retain their current optimizations.

K=0 must write attention output only, not mutate live recurrent state. Acceptance commits only the accepted prefix; rejection of every draft token must preserve the original state. Both regular and fused graph paths need this invariant.

### CUDA arithmetic and supported shapes

Prism's kernel has a multi-column register layout on DGX Spark plus RAW/G_PRECOMPUTED variants. The old single-column helper substitution must be moved inside the column loop rather than replacing the whole kernel. Verification and fold should use consistent FP32 formulas and reduction order; long recurrence can amplify drift.

The existing fold kernel is specialized to state dimension 128, key heads 16, value heads 48, inner dimension 6144, convolution width 4, one sequence, one CUDA device and draft length 1-5. `use_gdn_replay()` already checks these conditions and only enables replay in explicit mode 2; automatic mode retains snapshots. Preserve these guards. Actual Bonsai GGUF architecture and dimensions still need model-load verification. Weight-format support does not establish MTP availability.

### KV position and transform semantics

Prism's cell extension currently contains only x/y, and `reset()` zeroes it. Adding a default member initializer alone does not preserve logical_pos=-1 after reset. Text batches currently skip extension initialization unless positions are 2D, so logical positions must be set on every relevant write.

Preserve separate logical and RoPE positions, sparse restored blocks and correct attention masks. Validate deletion/reuse and query replay, including embedding inputs. Existing hole handling applying cleanly does not validate cache behavior.

Prism's weight Hadamard transforms are applied around weight matmuls/embeddings; the qwen35 Q/K/V capture hunks apply cleanly. This supports keeping capture after projection/normalization and before the existing RoPE placement. It does not establish numerical equivalence without a model test. Do not combine weight-basis rotation with KVMem's separate KV quantization rotation.

Prism also has optional K-cache mean centering in `cpy_k()`. No activation of that feature was found in the current KVMem tools/adapter. Keep it disabled initially: direct KVMem stage-in can bypass cpy_k and would require the same centering contract if the feature is exposed later.

## Recommended implementation order

1. Port factory, capture, logical positions, sparse KV and basic hybrid memory. Validate Bonsai text inference with MTP disabled and ordinary recurrent handling.
2. Validate retrieval and query replay across slot eviction/restoration, comparing unchanged resident execution where applicable. Run ordinary Qwen regression cases as well.
3. Port RECORD/Fold with raw gates and rows disabled only for ReplaySSM. Reuse `kvmem-gdn-replay-test` to check zero/partial/full acceptance, state values, convolution history and continued decode; ensure CUDA tests actually run rather than skip.
4. Validate a compatible MTP model/drafter separately with `kvmem-mtp-kv-test`; do not assume Bonsai ships the required MTP tensors.
5. Re-enable compatible fusion optimizations one at a time, measuring correctness and throughput against the conservative path. Broader backend support is separate work.
6. Regenerate the cumulative patch against the Prism pin, update replay scripts/docs and verify fresh application plus idempotent reapplication. The current CMake header guard only detects an unapplied adapter, not a complete compatible port.

## Assessment

Windows/CUDA text plus KVMem retrieval is medium-risk engineering with no identified fundamental architecture conflict. ReplaySSM/MTP is high-risk until state-equivalence tests pass. Cross-backend support and simultaneous fusion optimization increase scope significantly.

Planning estimate, not a measured schedule: 2-4 engineering days for a first compilable basic port, 3-7 additional days for recurrence/MTP regression and fixes, and a further 2-5 days if optimizing fused paths. Compile/API failures outside the patch set and actual GGUF metadata can change this estimate.

No build, model download, inference, performance measurement or production deployment was performed by this analysis.
