#pragma once

#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-memory-kvmem.h"

#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"

struct ggml_backend_sched;

#include <cstdint>
#include <memory>
#include <vector>

// MTP draft KV as a lockstep follower of the target slot-pool.
//
// kv_size is copied from target (budget + gen_reserve), never recomputed
// from draft n_ctx. The same block_id maps to the same slot index and the
// same original pos on the cell. Follower does not alloc/free slots.
class llama_memory_kvmem_mtp : public llama_memory_i {
public:
    llama_memory_kvmem_mtp(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams,
            llama_memory_kvmem * target);

    ~llama_memory_kvmem_mtp() override;

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

    uint32_t kv_size() const { return kv_size_; }
    llama_kv_cache * get_kv() { return kv_.get(); }
    llama_memory_kvmem * target() { return target_; }

    bool is_mtp_layer(int il) const { return il >= 0 && (uint32_t) il >= n_layer_trunk_; }
    void register_capture(struct ggml_tensor * t, int il, char which);
    void capture_on_new_graph();
    void harvest_pending(struct ggml_backend_sched * sched);
    void harvest_flush();
    uint32_t harvest_perf_n_ubatch() const { return perf_n_ubatch_; }
    int64_t harvest_perf_sync_us() const { return perf_sync_us_; }
    uint64_t harvest_perf_nvme_bytes() const { return perf_nvme_bytes_; }
    uint64_t harvest_perf_nvme_syscalls() const { return perf_nvme_syscalls_; }
    void on_stage_out(uint32_t block_id);
    void harvest_resident_v();
    void follow_retrieval();
    void detach_target() { target_ = nullptr; }

    // Retrieval layout: keep native MTP GPU KV for blocks that still sit
    // at src_slot, D2D them to dst_slot (same permutation as the trunk).
    struct LayoutMove {
        int32_t src_slot = -1;
        int32_t dst_slot = -1;
        uint32_t n_tokens = 0;
    };
    bool slot_holds(int32_t slot, uint32_t orig_pos) const;
    bool layout_d2d(const LayoutMove * moves, size_t n_moves);
    void occupy_block(uint32_t block_id);

private:
    bool fill_from_target(const llama_ubatch & ubatch, llama_kv_cache::slot_info & out);
    void write_block_to_gpu(uint32_t block_id);
    void harvest_k(uint32_t block_id);
    void harvest_v(uint32_t block_id);

    const llama_model & model_;
    llama_memory_kvmem * target_ = nullptr;
    uint32_t kv_size_ = 0;
    uint32_t block_tokens_ = 32;
    uint32_t n_layer_trunk_ = 0;
    uint32_t il_graph_ = 0;
    uint32_t n_embd_k_ = 0;
    uint32_t n_embd_v_ = 0;
    ggml_type type_k_ = GGML_TYPE_F16;
    ggml_type type_v_ = GGML_TYPE_F16;
    bool v_trans_ = false;
    bool trace_ = false;
    kvmem::RopeConfig rope_{};
    std::unique_ptr<llama_kv_cache> kv_;
    std::unique_ptr<kvmem::RawKvStore> raw_;

    struct CaptureNode {
        ggml_tensor * t = nullptr;
        char which = 0;
    };
    std::vector<CaptureNode> pending_capture_;
    std::vector<std::vector<llama_pos>> pos_queue_;
    std::vector<llama_pos> cur_pos_;
    uint32_t perf_n_ubatch_ = 0;
    int64_t perf_sync_us_ = 0;
    uint64_t perf_nvme_bytes_ = 0;
    uint64_t perf_nvme_syscalls_ = 0;
};
