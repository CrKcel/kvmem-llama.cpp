#pragma once

#include "llama-kv-cache.h"
#include "llama-memory.h"

#include "kvmem/kvmem_runtime.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"

#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;
struct llama_cparams;
struct llama_memory_params;
struct ggml_tensor;
class llama_memory_recurrent;

// Bounded block-slot pool over a llama_kv_cache.
//
// GPU attention cache size = min(n_ctx, budget + gen_reserve). Each logical
// KVMem block occupies one slot of `block_tokens` cells. Reselect never packs
// cells into [0, W); resident blocks keep their slot. init_batch returns a
// llama_kv_cache_context so llama-graph.cpp can static_cast as usual.
class llama_memory_kvmem : public llama_memory_i {
public:
    llama_memory_kvmem(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams,
            llama_kv_cache * ext_kv = nullptr);

    ~llama_memory_kvmem() override;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_kv_cache * get_kv() { return kv_; }
    kvmem::KvMemRuntime & runtime() { return *runtime_; }
    const kvmem::KvMemRuntime & runtime() const { return *runtime_; }

    uint32_t kv_size() const { return kv_size_; }
    uint32_t block_tokens() const { return block_tokens_; }
    uint32_t n_slots() const { return n_slots_; }

    // Slot-pool prepare used by both the dense KVMem memory and the hybrid
    // wrapper (attn half). Fills per-ubatch slot_info and the capture pos queue.
    bool prepare_ubatches(
            const std::vector<llama_ubatch> & ubatches,
            uint32_t n_new_tokens,
            llama_kv_cache::slot_info_vec_t & sinfos);
    void reset_policy();

    int32_t alloc_slot();
    void free_slot(int32_t slot);

    void note_ubatch_pos(const std::vector<llama_pos> & pos);
    void reset_query_acc();
    void register_capture(struct ggml_tensor * t, int il, char which);
    void capture_on_new_graph();
    bool capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) const;
    void harvest_pending(struct ggml_backend_sched * sched);
    void harvest_flush();
    void harvest_capture(struct ggml_tensor * t, int il, char which);
    void apply_retrieval();
    void dump_kv_compare(int32_t block_id, bool writeback_test = false);
    void trace_working_set(const char * tag) const;
    void set_replay(bool replay) { replay_ = replay; }
    bool replay() const { return replay_; }
    void set_recurrent(llama_memory_recurrent * recr) { recr_ = recr; }
    bool has_recurrent() const { return recr_ != nullptr; }
    void set_query_span(int32_t begin, int32_t end) {
        query_begin_ = begin;
        query_end_ = end;
    }
    void set_force_pos(int32_t pos) { force_pos_ = pos; }

    kvmem::RawKvStore & raw() { return *raw_; }

private:
    struct SlotBackend : public kvmem::KvMemBackend {
        llama_memory_kvmem * owner = nullptr;
        int32_t alloc_gpu_slot() override { return owner->alloc_slot(); }
        void free_gpu_slot(int32_t slot) override { owner->free_slot(slot); }
        void copy_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                void * host, uint64_t bytes) override {
            owner->copy_gpu_block_to_host(block_id, gpu_slot, host, bytes);
        }
        void copy_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                  const void * host, uint64_t bytes) override {
            owner->copy_gpu_block_from_host(block_id, gpu_slot, host, bytes);
        }
    };

    uint32_t resident_tokens() const;
    bool prepare_working_set(uint32_t n_new_tokens);
    void apply_plan_to_kv(const kvmem::KvMemPlan & plan);
    // Place GPU-resident blocks into slots 0..N-1 in orig_pos order.
    // Resident KV is copied slot-to-slot; only cold blocks are rebuilt from raw.
    bool layout_gpu_slots_by_orig_pos();
    bool gpu_kv_already_resident(uint32_t block_id) const;
    void occupy_block_cells(uint32_t block_id);
    void reset_slots();
    void trace_plan(const char * tag, const kvmem::KvMemPlan & plan) const;
    void write_block_to_gpu(uint32_t block_id);
    void harvest_gpu_v(uint32_t block_id);
    void copy_gpu_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                void * host, uint64_t bytes);
    void copy_gpu_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                  const void * host, uint64_t bytes);
    void score_retrieval();
    bool read_gpu_block(uint32_t block_id, uint32_t il, bool is_k, std::vector<float> & out) const;
    static void kv_stats(const char * tag, const float * a, const float * b, size_t n);
    static void tensor_to_f32_token_major(const struct ggml_tensor * t, std::vector<float> & out);
    static void bytes_to_f32_token_major(const uint8_t * data, ggml_type type,
                                         int64_t d, int64_t h, int64_t n,
                                         size_t nb0, size_t nb1, size_t nb2,
                                         std::vector<float> & out);
    void harvest_from_host(int il, char which, const uint8_t * host,
                           ggml_type type, int64_t d, int64_t h, int64_t n,
                           size_t nb0, size_t nb1, size_t nb2);
    bool d2h_init();
    void d2h_free();
    void d2h_commit(int slot);
    bool d2h_submit(struct ggml_backend * be);

    const llama_model & model_;
    uint32_t block_tokens_ = 128;
    uint32_t kv_size_ = 0;
    uint32_t n_slots_ = 0;
    bool trace_ = false;

    std::unique_ptr<llama_kv_cache> kv_owned_;
    llama_kv_cache * kv_ = nullptr;
    llama_memory_recurrent * recr_ = nullptr;
    SlotBackend backend_;
    std::unique_ptr<kvmem::KvMemRuntime> runtime_;
    std::unique_ptr<kvmem::RawKvStore> raw_;
    std::vector<int32_t> free_slots_;

    kvmem::RopeConfig rope_{};
    uint32_t n_layer_ = 0;
    uint32_t n_embd_k_ = 0;
    uint32_t n_embd_v_ = 0;
    uint32_t n_head_ = 0;
    uint32_t n_head_kv_ = 0;
    uint32_t n_embd_head_ = 0;
    ggml_type type_k_ = GGML_TYPE_F16;
    ggml_type type_v_ = GGML_TYPE_F16;
    bool v_trans_ = false;
    bool replay_ = false;
    bool retrieval_pinned_ = false;
    int32_t method_ = 0;
    int32_t query_begin_ = -1;
    int32_t query_end_ = -1;
    int32_t force_pos_ = -1;

    std::vector<std::vector<llama_pos>> pos_queue_;
    std::vector<llama_pos> cur_pos_;
    struct CaptureNode {
        ggml_tensor * t = nullptr;
        int il = 0;
        char which = 0;
    };
    std::vector<CaptureNode> pending_capture_;
    bool graph_has_q_ = false;
    struct CaptureD2hPipe;
    std::unique_ptr<CaptureD2hPipe> d2h_;
    std::vector<std::vector<float>> q_sum_;
    std::vector<uint32_t> q_count_;
};

// GPU attn-cache cell count for a KVMem slot pool (budget + gen_reserve,
// clamped to n_ctx on identity). Hybrid uses this as llama_memory_hybrid's
// attn kv_size so the two halves agree.
uint32_t llama_kvmem_pool_cells(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams);
