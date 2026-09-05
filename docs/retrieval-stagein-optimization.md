# Retrieval Stage-In / Layout Optimization (llama.cpp KVMem)

| Field | Value |
|---|---|
| **Title** | Next staged plan after harvest: `apply_retrieval` layout + cold stage-in |
| **Date** | 2026-09-06 |
| **Status** | Active (rev 1). Harvest IO is landed; this document supersedes it as the roadmap. |
| **Repo** | `/home/leye/kvmem_llamacpp` (v0.5.0) |
| **Reference (read-only)** | `/home/leye/kvmem_qw3` |
| **Prior plan** | `docs/prefill-harvest-optimization.md` (Stage 1 harvest/NVMe; do not implement further harvest PRs from it) |

Do not patch `ggml_flash_attn_ext`. Do not restart qw3. ISTA IQ3_S: MTP off (`--spec-type none`).

---

## Overview

ISTA IQ3_S 128k on the RTX 5090 (`logs/ista_iq3_128k_speed_harvest.log`) after the harvest thread + RAM mean-K cache:

| | Before harvest PRs | After |
|---|---|---|
| off prefill | 745 tok/s / 176 s | 738 / 178 s |
| retr prefill | 380 / 345 s | 371 / 353 s |
| **CLI `retrieval_ms`** | 234 s | **229 s** |
| decode off / retr | 32.2 / 40.6 | 32.0 / 40.9 |

`KVMEM_HARVEST_SUM`: `commit_us=0` per ubatch, `pack_ms=15.5`, `nvme_ms=0`. Pack is off the compute thread. GPU during `apply_retrieval`: ~16% SM, HBM ~0%.

The remaining KVMem-local bill is **`llama_kvmem_apply_retrieval`**, not cosine mean-K and not NVMe. Prefill still ~2× stock because harvest **host-waits `compute_done`** so ubatch N+1 `set_input` (`cudaStreamPerThread`) cannot overlap graph N, plus a 4087-node ggml graph. That prefill gap is **out of this plan** (see Non-Goals).

This plan is: time the 229 s, then stop using host as a scratchpad for **already-baked GPU KV**, then batch cold H2D and optionally GPU-RoPE at **orig_pos**. It is not qw3 compact-window assembly.

---

## Diagnosis

`apply_retrieval` today:

1. `harvest_flush` (should be cheap after prefill).
2. `score_retrieval` — cached mean memcpy + CPU dots. Mean is captured on every K write. Not expected to be 229 s, but **unmeasured**; `score_retrieval` also heap-allocates a per-head vector inside the block×layer×head loop.
3. `apply_plan_to_kv` — `stage_out` harvests V; no `wait_writes` on this path.
4. **`layout_gpu_slots_by_orig_pos`** — if GPU slots are not already orig_pos order:
   - **resident** blocks: per-token `ggml_backend_tensor_get` (D2H) into a host payload, reset slots, per-token `tensor_set` (H2D) into the new slot. Bytes unchanged; only the slot index changes. Each get/set **synchronizes `cudaStreamPerThread`**.
   - **cold** blocks: `write_block_to_gpu`.
5. If layout is a no-op: cold `stage_in` via `write_block_to_gpu` only.
6. `write_block_to_gpu`: `copy_k` (F16→F32) + host `rope_neox_apply` at **`orig_pos_start`** + per-token F32→F16 + `tensor_set`.

Two different jobs share the 229 s wall:

| Job | Data source | RoPE | Purpose |
|---|---|---|---|
| resident layout | GPU cache (already baked) | none | pack slots `[0..N)` by orig_pos |
| cold stage-in | `RawKvStore` unrotated K | host, orig_pos | admit retrieved blocks |

Raw-K is already immutable (pre-RoPE FP16). qw3 additionally bakes GPU working K at **compact window** positions and remaps via page tables. llama.cpp Q RoPE and `ggml_flash_attn_ext` stay on **orig `llama_pos`**. Compact-window re-RoPE is incompatible with the FA freeze.

PR 0 must split the 229 s **before** choosing whether PR 1 (batch get/set), PR 2 (D2D layout), or a score fix goes first.

---

## Goals & Non-Goals

### Goals

- Drive 128k `retrieval_ms` from ~229 s toward seconds (record; never auto-fail).
- Keep raw-K as unrotated host FP16 authority. GPU working K stays baked at **orig_pos**. Slot ≠ window RoPE coordinate.
- `skip` / already-resident blocks: no copy, no RoPE.
- Cold `stage_in`: bounded batched H2D; GPU RoPE at orig_pos is PR 3, not PR 1.
- If slots must be packed: **GPU D2D**, not host round-trip of baked KV.
- MTP uses the same helpers (0.8B canary). IQ3 recipes stay `--spec-type none`.
- Identity PASS; retrieval BLUEBIRD-42 with `--no-think` is GO/NO-GO.

### Non-Goals

- Do not patch `ggml_flash_attn_ext` / FlashInfer / NVFP4 / GDN port.
- Do not switch Q/K to compact window coordinates.
- Do not remove harvest’s host-wait of `compute_done` (N+1 `set_input` race).
- Do not move `set_input` onto the compute stream.
- Do not promise kvmem prefill → stock 738 tok/s.
- Do not retest IQ3 MTP. Do not restart qw3.

---

## Key Decisions

1. **Measure first (PR 0).** `KVMEM_PERF=1` prints `KVMEM_RETR_SUM`. If `score_ms` dominates, fix scoring allocations before any copy PR.
2. **Resident layout and cold stage-in are separate PRs.** Host payload for baked KV is a missing D2D helper, not “immutable-K assemble.”
3. **D2D over H2D for resident.** Staging GPU memory is **per layer** (~60k × 2 KiB ≈ 120 MiB K), never a 3–4 GiB all-layers arena (5090 also holds the 27B weights).
4. **Cold path: batch copies first, GPU orig_pos RoPE second.** PR 1 keeps host `rope_neox_apply` but one `tensor_set` per `(block, layer, K\|V)` of `n_tokens` contiguous rows. PR 3 H2Ds unrotated F16 and RoPEs on GPU at orig_pos.
5. **Bounded staging.** Full-window 16 attn layers × K+V × 60k × F16 ≈ 4 GiB. Chunk by layer or 8–32 blocks. Pinned host staging. Partial last block writes `n_tokens` rows only.
6. **One fence before query replay / decode.** Today correctness is “every token syncs.” Async copies must host-wait (or stream-wait the compute stream) before `apply_retrieval` returns.
7. **CUDA graphs:** copies sit at the prefill→decode boundary. Use a copy stream; do not inject into compute-stream capture. Wait before the first decode graph.
8. **Coordinate system stays orig_pos.** Same convention as graph Q RoPE and cell mask. No qw3 window remap.
9. **Prefill 2× is not this plan.** Optional later: `-b 2048 -ub 2048` after copies are gone; default stays 512.

---

## Proposed Design (after PR 2–3)

```text
apply_retrieval
  harvest_flush
  score_retrieval          // mean memcpy + dots
  apply_plan_to_kv         // stage_out V enqueue; no wait_writes
  layout only if slots are not orig_pos-ordered
     resident: GPU D2D slot → slot (scratch one layer if overlap)
     cold:     unrotated K H2D → GPU RoPE(orig_pos) → dest slot
               V H2D, no RoPE
  occupy_block_cells       // n_tokens only
  fence copy/RoPE stream
  MTP follow (same helpers)
```

`skip`: no D2D, no RoPE, no H2D.

---

## PR Plan

Independently reviewable. Do not mix compact-window RoPE into copy PRs.

### PR 0 — `kvmem: time apply_retrieval (KVMEM_RETR_SUM)`

- **Files:** `src/adapter/llama-memory-kvmem.cpp`, `.h`
- **Depends on:** none
- **Changes:** `KVMEM_PERF=1` prints one line per `apply_retrieval`, no behavior change. Fields: `total_ms`, `flush_ms`, `score_ms`, `plan_ms`, `stage_out_ms`, `layout_d2h_ms`, `layout_h2d_ms`, `copy_ms`, `rope_ms`, `set_ms`, `mtp_ms`, `n_move`, `n_raw`, `n_skip`, `n_stage_in`, `laid_out`. TRACE `dump_kv_compare` / `trace_working_set` timed as `dump_ms` so they cannot hide inside layout. `write_block_to_gpu` always adds `copy_ms`/`rope_ms`/`set_ms` (including when called from layout).
- **Gate:** 0.8B identity; retrieval needle still GO/NO-GO; one 0.8B (or 16k 27B) run with `KVMEM_PERF=1` that **prints** `KVMEM_RETR_SUM`. Speed not a fail. **This line decides whether PR 1, PR 2, or a score fix is next.**

### PR 1 — `kvmem: batched tensor_get/set for a contiguous slot span`

- **Files:** `read_gpu_block` (`harvest_gpu_v`), `copy_gpu_block_to_host` / `from_host`, `write_block_to_gpu`; MTP `harvest_v` / `write_block_to_gpu`
- **Depends on:** PR 0
- **Changes:** cells `slot*bt .. slot*bt+nt-1` are contiguous rows. One `ggml_backend_tensor_{get,set}(..., n * row)` per `(layer, K|V)`. Host RoPE unchanged. Bit-identical. **16k IQ3 SUM (budget 8192): `stage_out` 47% is `read_gpu_block`; omitting it leaves the largest bucket.**
- **Gate:** identity; retrieval needle; 16k `KVMEM_RETR_SUM`. Expect `stage_out`/`set`/`layout_*` to drop; resident layout may still pay PCIe.

### PR 2 — `kvmem: layout resident blocks with GPU D2D`

- **Files:** `layout_gpu_slots_by_orig_pos`
- **Depends on:** PR 0 numbers; PR 1’s contiguous-span assumption
- **Changes:** resident moves use `cudaMemcpyAsync` of `nt*row` per layer. Overlap → per-layer GPU scratch (not a full-window multi-layer arena). Stop `copy_gpu_block_to_host` of baked K. Cold still `write_block_to_gpu`. Fence before return.
- **Gate:** identity (full-budget often skips layout — **insufficient**). 0.8B retrieval needle + 16k or 128k 27B retr. Optional `rebuild_vs_gpu cos≈1`.
- **Hypothesis:** 128k `n_move` is the bulk of 229 s.

### PR 3 — `kvmem: cold stage-in batched H2D + GPU RoPE at orig_pos`

- **Files:** `write_block_to_gpu`; small adapter CUDA or ggml RoPE helper (`src/adapter/` only; `kvmem/` stays llama-free)
- **Depends on:** PR 1. Parallel with PR 2
- **Changes:** H2D unrotated F16 K (no F32 round-trip if `copy_k` can be skipped). Pinned staging, chunked. GPU NeoX (Qwen3.5: same convention as today’s host path that already matches MRoPE empirically). V H2D only. Partial blocks: `n_tokens` rows. Fence. `v_trans_` (non-FA) does not use this batch shape.
- **Gate:** identity; retrieval GO/NO-GO; 0.8B Qwen3.5 dump cos≈1 vs GPU cache.

### PR 4 — `kvmem: MTP uses the same D2D / batch / fence helpers`

- **Files:** `llama-memory-kvmem-mtp.cpp`
- **Depends on:** PR 2–3 helpers (no third per-token loop)
- **Gate:** `mtp_canary.py` 0.8B; `n_no_raw=0` hard fail. No IQ3 MTP.

### PR 5 (optional) — document `-b 2048` if VRAM-safe

- Default remains 512. Does not close the 4087-node / `compute_done` prefill gap.

---

## Sequence

```text
PR 0 timers
  └─ if score_ms dominates: scoring fix (unplanned PR, not FA)
  └─ else:
        PR 1 batched get/set
          ├─ PR 2 layout D2D          // n_move
          └─ PR 3 GPU orig RoPE H2D   // n_raw; can parallel PR 2
        PR 4 MTP
        PR 5 optional ubatch
```

---

## Acceptance

| Test | GPU | When |
|---|---|---|
| `raw_kv_store_test` | CPU | every copy PR |
| identity 0.8B | 5050 | every PR |
| retrieval BLUEBIRD-42 `--no-think` | 5050 | PR 1–4 |
| recency must miss | 5050 | every PR |
| MTP canary `n_no_raw=0` | 5050 | PR 4 only |
| 16k IQ3 + `KVMEM_RETR_SUM` | 5090 | after PR 0, then after each copy PR |
| 128k IQ3 `retr_ms` + SUM | 5090 | after PR 2 and PR 3 (milestone, not auto-fail) |

Hardware: 5050 UUID `GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29` for **< 27B**; 5090 UUID `GPU-58a7c28b-e698-307f-c149-24d4ecd88bf4` for **27B only**. `source scripts/gpu.sh small|27b`.

---

## Open Questions

None that block PR 0. After PR 0, `KVMEM_RETR_SUM` chooses the next PR; do not start PR 1–3 until that line exists in-tree (16k is enough to see which bucket is non-zero; 128k remains the magnitude check).
