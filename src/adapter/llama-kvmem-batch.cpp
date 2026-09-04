#include "llama-kvmem-batch.h"

#include "llama-impl.h"

bool kvmem_fill_slot_info(
        const kvmem::KvMemStore & store,
        uint32_t block_tokens,
        uint32_t kv_size,
        const llama_ubatch & ubatch,
        llama_kv_cache::slot_info & out) {
    if (ubatch.n_tokens == 0 || block_tokens == 0 || !ubatch.pos) {
        return false;
    }
    // n_pos > 1 is M-RoPE. Dim 0 is the sequential token position (text
    // tokens broadcast that value across the other dims).

    out.s0 = 0;
    out.s1 = 0;
    out.resize(1);
    out.strm[0] = 0;
    out.idxs[0].clear();
    out.idxs[0].reserve(ubatch.n_tokens);

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id && ubatch.n_seq_id[i] > 1) {
            LLAMA_LOG_ERROR("%s: KVMem P1 is single-sequence only\n", __func__);
            return false;
        }
        const llama_pos pos = ubatch.pos[i];
        if (pos < 0) {
            LLAMA_LOG_ERROR("%s: negative pos at token %u\n", __func__, i);
            return false;
        }
        const int32_t bid = store.block_id_containing(static_cast<uint32_t>(pos));
        if (bid < 0) {
            LLAMA_LOG_ERROR("%s: no KVMem block for pos %d\n", __func__, (int) pos);
            return false;
        }
        const kvmem::KvMemBlock & blk = store.blocks()[static_cast<uint32_t>(bid)];
        if (blk.gpu_slot < 0) {
            LLAMA_LOG_ERROR("%s: block %u has no GPU slot (pos %d)\n", __func__, blk.block_id, (int) pos);
            return false;
        }
        const uint32_t off = static_cast<uint32_t>(pos) - blk.orig_pos_start;
        if (off >= blk.n_tokens || off >= block_tokens) {
            LLAMA_LOG_ERROR("%s: pos %d out of block %u range\n", __func__, (int) pos, blk.block_id);
            return false;
        }
        const uint32_t cell = static_cast<uint32_t>(blk.gpu_slot) * block_tokens + off;
        if (cell >= kv_size) {
            LLAMA_LOG_ERROR("%s: cell %u >= kv_size %u (slot %d)\n",
                    __func__, cell, kv_size, (int) blk.gpu_slot);
            return false;
        }
        out.idxs[0].push_back(cell);
    }

    return out.idxs[0].size() == ubatch.n_tokens;
}
