#include "llama-memory-kvmem-hybrid.h"

#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>

llama_memory_kvmem_hybrid::llama_memory_kvmem_hybrid(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) :
    llama_memory_hybrid(
            model,
            params.type_k,
            params.type_v,
            !cparams.flash_attn,
            llama_kvmem_pool_cells(model, params, cparams),
            /* n_pad */ 1,
            model.hparams.n_swa,
            model.hparams.swa_type,
            GGML_TYPE_F32,
            GGML_TYPE_F32,
            std::max((uint32_t) 1, cparams.n_seq_max),
            cparams.n_seq_max,
            cparams.n_rs_seq,
            cparams.offload_kqv,
            /* unified */ true,
            [&](int32_t il) {
                return il < (int32_t) model.hparams.n_layer() && !model.hparams.is_recr(il);
            },
            [&](int32_t il) {
                return il < (int32_t) model.hparams.n_layer() && model.hparams.is_recr(il);
            }) {
    attn_kvmem_ = std::make_unique<llama_memory_kvmem>(
            model, params, cparams, get_mem_attn());
    attn_kvmem_->set_recurrent(get_mem_recr());
    LLAMA_LOG_INFO("%s: KVMem hybrid (attn=slot-pool recr=stock) n_rs_seq=%u\n",
            __func__, cparams.n_rs_seq);
}

llama_memory_context_ptr llama_memory_kvmem_hybrid::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;
            if (embd_all) {
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Keep GDN rollback snapshots valid: trailing (1 + n_rs_seq)
                // tokens of each seq stay in one ubatch.
                const bool unified = (get_mem_attn()->get_n_stream() == 1);
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;
                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (!get_mem_recr()->prepare(ubatches)) {
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        llama_kv_cache::slot_info_vec_t sinfos;
        if (!attn_kvmem_->prepare_ubatches(ubatches, balloc.get_n_tokens(), sinfos)) {
            LLAMA_LOG_ERROR("%s: failed to prepare KVMem attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

void llama_memory_kvmem_hybrid::clear(bool data) {
    llama_memory_hybrid::clear(data);
    if (attn_kvmem_) {
        attn_kvmem_->reset_policy();
    }
}

bool llama_memory_kvmem_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const llama_pos p0n = p0 < 0 ? 0 : p0;
    const bool full = seq_id <= 0 && p0n == 0 && p1 < 0;
    // Partial attn eviction (holes / query replay of the attn cache) must not
    // roll back GDN. Recurrent seq_rm of a suffix is a one-shot snapshot
    // rollback and otherwise fails.
    if (!full) {
        return attn_kvmem_ ? attn_kvmem_->seq_rm(seq_id, p0, p1)
                           : get_mem_attn()->seq_rm(seq_id, p0, p1);
    }
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return attn_kvmem_ ? attn_kvmem_->seq_rm(seq_id, p0, p1)
                       : get_mem_attn()->seq_rm(seq_id, p0, p1);
}
