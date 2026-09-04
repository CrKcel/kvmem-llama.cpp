#pragma once

#include "llama-memory-hybrid.h"
#include "llama-memory-kvmem.h"

#include <memory>

// llama_memory_hybrid whose attention half is a KVMem slot-pool and whose
// recurrent half is stock llama_memory_recurrent.
//
// Graph still static_cast's to llama_memory_hybrid_context. GDN sees every
// token; pressure reselect / seq_rm of attn holes must not roll back GDN.
class llama_memory_kvmem_hybrid : public llama_memory_hybrid {
public:
    llama_memory_kvmem_hybrid(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams);

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    llama_memory_kvmem * attn_kvmem() { return attn_kvmem_.get(); }

private:
    std::unique_ptr<llama_memory_kvmem> attn_kvmem_;
};
