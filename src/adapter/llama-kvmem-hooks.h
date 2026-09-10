#pragma once

// Public-ish KVMem controls. Compiled into libllama when LLAMA_KVMEM=ON.
// llama.cpp itself only sees llama_memory_kvmem_maybe_create via the factory
// hook header in the submodule patch.

#include "llama.h"

#include <stdint.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_backend_sched;

#ifdef __cplusplus
extern "C" {
#endif

struct llama_kvmem_params {
    bool enabled;
    uint32_t block_tokens;  // 0 → 32
    uint32_t budget;        // tokens; 0 → use n_ctx (identity / no sparse)
    uint32_t gen_reserve;   // tokens; 0 → 256
    uint32_t sink_tokens;   // 0 → one block
    uint32_t recent_tokens; // 0 → pressure uses newest tail only
    int32_t  method;        // 0 recency, 1 retrieval (CLI default 1)
    int32_t  query_begin;   // original token pos, -1 = unset
    int32_t  query_end;     // exclusive, -1 = end of prompt
    int32_t  force_pos;     // include block containing this orig pos; -1 = none
    float    gpu_memory_ratio;     // 0 → 0.50; cap pool vs device VRAM
    float    gpu_high_watermark;   // 0 → 0.95; trigger prefill offload
    float    gpu_low_watermark;    // 0 → 0.85; documented target after offload
    uint64_t cpu_bytes;            // CPU spill arena; 0 = disabled
    uint64_t nvme_bytes;           // NVMe spill file; 0 = disabled
    const char * nvme_dir;         // directory for the ephemeral NVMe file
    bool     raw_k_nvme;           // put raw-K/V authority on NVMe (qw3-style)
    bool     harvest_v;            // prefill D2H V with K (default off; not implied by raw_k_nvme)
};

// Call before llama_init_from_model. A null pointer resets to defaults
// (disabled).
LLAMA_API void llama_kvmem_set_params(const struct llama_kvmem_params * params);
LLAMA_API const struct llama_kvmem_params * llama_kvmem_get_params(void);

// Legacy eval-callback entry. Always returns false so ggml does not split the
// graph. Capture harvest runs after the full ubatch compute instead.
LLAMA_API bool llama_kvmem_eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

// Graph-build / post-compute capture (LLAMA_KVMEM process_ubatch hook).
LLAMA_API void llama_kvmem_register_capture(struct ggml_tensor * t, int il, char which);
// is_mtp: 1 when the llama_context building this graph is the MTP draft.
// Trunk and MTP must not clear each other's capture pending.
LLAMA_API void llama_kvmem_capture_on_new_graph(int is_mtp);
LLAMA_API void llama_kvmem_harvest_ubatch(struct ggml_backend_sched * sched, int is_mtp);
// True when this ubatch overlaps [query_begin, query_end) and Q nodes should
// be present. n_pos is the per-token position stride (1 for 1-D RoPE).
LLAMA_API bool llama_kvmem_ubatch_needs_q_capture(uint32_t n_tokens, uint32_t n_pos,
                                                  const llama_pos * pos);
// False if the last built graph's Q-capture topology would not match this ubatch.
// MTP graphs do not capture Q; is_mtp=1 always allows reuse from KVMem's side.
LLAMA_API bool llama_kvmem_capture_can_reuse(uint32_t n_tokens, uint32_t n_pos,
                                             const llama_pos * pos, int is_mtp);
// True while prefill/query-span harvest is still allowed (retrieval, not pinned).
// MTP verify is n>1 after pin; callers must not harvest those ubatches into raw-K.
LLAMA_API bool llama_kvmem_want_prefill_capture(void);
// Q capture for retrieval scoring. True for query-span ubatches before pin,
// including replay of a cached query (T5).
LLAMA_API bool llama_kvmem_want_q_capture(uint32_t n_tokens, uint32_t n_pos,
                                          const llama_pos * pos);
LLAMA_API void llama_kvmem_reset_query(void);
// True after retrieval pin: capture pre-RoPE K for decode mean-K (n=1 and MTP verify).
LLAMA_API bool llama_kvmem_want_decode_mean(void);
// Close prefill harvest before decode / speculative verify.
LLAMA_API void llama_kvmem_end_prefill_capture(void);
// Start a follow-up turn that keeps the GPU prefix. Call seq_rm(n_past,-1)
// on trunk + MTP after this; then prefill only the suffix.
LLAMA_API void llama_kvmem_begin_cached_turn(void);
// Same last-user continuation: keep the already-captured Q for top-k.
LLAMA_API void llama_kvmem_begin_cached_turn_keep_query(void);
// Same-query skip: do not recency-reselect; keep the retrieved GPU window.
LLAMA_API void llama_kvmem_keep_selected(void);
// Pin after skip prefill so decode mean-K uses gen_reserve slots.
LLAMA_API void llama_kvmem_pin_working_set(void);
LLAMA_API uint32_t llama_kvmem_free_slots(void);
LLAMA_API void llama_kvmem_truncate_cached(uint32_t n_past);
LLAMA_API uint32_t llama_kvmem_store_n_tokens(void);
LLAMA_API llama_pos llama_kvmem_recr_pos_max(void);
// After MTP verify: keep the first n_keep batch tokens in the running mean (0 = discard).
LLAMA_API void llama_kvmem_decode_mean_commit(uint32_t n_keep);
LLAMA_API void llama_kvmem_decode_mean_discard(void);
// Write any partial-block running mean to the host store (end of turn / evict).
LLAMA_API void llama_kvmem_decode_mean_flush(void);

// Score + reselect + stage-in raw-K/V for retrieval. No-op if method is recency.
LLAMA_API void llama_kvmem_apply_retrieval(struct llama_context * ctx);
// True when sink + [query_begin, prompt_end) fits the GPU selection budget
// (qw3 kvmem_replay_capacity). False → skip query seq_rm/replay.
LLAMA_API bool llama_kvmem_query_replay_fits(uint32_t query_begin, uint32_t prompt_end);
LLAMA_API void llama_kvmem_set_replay(bool replay);
LLAMA_API void llama_kvmem_trace_cells(struct llama_context * ctx, const char * tag);
// True when the active KVMem memory is hybrid (attn slot-pool + stock GDN).
LLAMA_API bool llama_kvmem_has_recurrent(void);
// Update query span / force_pos on the live memory (after llama_init_from_model).
LLAMA_API void llama_kvmem_set_request_span(int32_t query_begin, int32_t query_end, int32_t force_pos);
// Compare never-evicted GPU KV vs host-rebuild from raw-K. block_id -1 = force_pos block.
LLAMA_API void llama_kvmem_dump_kv_compare(struct llama_context * ctx, int32_t block_id);
LLAMA_API void llama_kvmem_dump_kv_writeback(struct llama_context * ctx, int32_t block_id);

#ifdef __cplusplus
}
#endif
