#pragma once

#include "llama-batch.h"
#include "llama-kv-cache.h"

#include "kvmem/kvmem_store.hpp"

#include <cstdint>

// Map each ubatch token to a cell in its block's GPU slot.
// cell = gpu_slot * block_tokens + (orig_pos - orig_pos_start)
// Slot number is not the window RoPE coordinate; cell.pos stays the original
// (monotonic) token position. Dim 0 of M-RoPE batches is that sequential pos.
bool kvmem_fill_slot_info(
        const kvmem::KvMemStore & store,
        uint32_t block_tokens,
        uint32_t kv_size,
        const llama_ubatch & ubatch,
        llama_kv_cache::slot_info & out);
