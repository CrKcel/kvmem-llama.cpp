# Architecture (P0)

KVMem is a block-sparse KV working-set manager. llama.cpp will own inference;
this library owns selection, tiering, and (later) window assembly.

```
kvmem/          host policy + CPU/NVMe  (this repo, no llama.cpp headers)
src/adapter/    llama_memory_i wrapper  (P1+)
llama.cpp/      submodule + thin patches (P1+)
```

GPU attention cache is a **bounded block-slot pool** of size
`budget + gen_reserve`. Product default GPU KV type is llama.cpp **q8_0**
for K and V (`--kv-dtype q8_0`; `f16` or `q4_0` to override). Cell `pos` is
the original monotonic token position (slot index is not a RoPE coordinate).
Restore is packed GPU-format memcpy at that orig pos — no unrotated raw-K
and no re-RoPE on the product path. Retrieval scores **mean-K** (F32,
pre-RoPE, captured at first write). Packed K/V for a full block are
copied to host asynchronously when the block fills, overlapping later
prefill; eviction then skips if the copy exists. After retrieval pin,
decode keeps a GPU running sum of pre-RoPE K and writes mean-K when a
block fills (accepted tokens only; MTP drafts are not counted). Cold
stage-in is `copy_k_gpu` / `copy_v_gpu` + slab H2D. `llama-kvmem-server`
reuses the GPU prefix across requests (token LCP); the retrieval query
is the last `role=user` span. Chat tools reuse llama.cpp `common/chat`
+ `common_sampler`; the server does not execute tools. Packed transfers use a **32 MiB**
GPU slab. GPU-format CPU/NVMe scratch is allocated only when those tiers
are on. Each logical block occupies one slot of `block_tokens` cells.
Reselect is a `KvMemPlan` diff: resident selected blocks stay in their
slot; only `stage_out` cells are `seq_rm`'d. Flash Attention is not
modified. P1 recency does not re-RoPE and does not resurrect dropped
blocks.

Hardware split on this machine: RTX 5050 (GPU 0) for models < 27B;
RTX 5090 (GPU 1) for 27B. Details in `scripts/gpu.sh` and
`docs/modification-plan.md`.

## Known v1 limit: generation length vs `gen_reserve`

GPU pool = `budget` (working set) + `gen_reserve` (decode slack). After
retrieval the working set is **pinned**: decode must not recency-reselect
and drop resurrected blocks.

If the last GPU block is full and there is no free slot left in
`gen_reserve`, `prepare_working_set` fails with `no free GPU slot for
block N` and `llama-kvmem-cli` exits (`llama_decode(gen) failed rc=1`).
It does **not** evict pinned retrieval blocks. Long continuations can
therefore hit a hard stop.

Workaround today: raise `--kvmem-gen-reserve` (CLI default 256 tokens).

Follow-up feature (not v1): when gen slots are exhausted, recycle **only**
generation-occupied slots (or spill those gen blocks to CPU/NVMe) and
keep the pinned retrieval working set. Recency decode still has the
related `block_count() > budget` mis-trigger; same class of issue.

## P7: MTP shares the slot-pool (plan B)

Logically long context must not grow a full-length MTP KV (plan A). The
draft context (`LLAMA_CONTEXT_TYPE_MTP`) gets a **follower** slot-pool
the same size as the target attention cache, same block IDs and slot
indices, original `pos` on cells. Speculative decoding stays in
llama.cpp `draft-mtp`. Details: `docs/kvmem-mtp-plan.md`.
