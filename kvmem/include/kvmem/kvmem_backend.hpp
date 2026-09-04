#pragma once

// Engine-agnostic storage/copy hooks. P0 is host-only: the runtime updates
// tier metadata without touching device memory. llama.cpp adapter (P1+)
// implements copies via ggml_backend_tensor_get/set.

#include <cstdint>

namespace kvmem {

class KvMemBackend {
public:
    virtual ~KvMemBackend() = default;

    virtual int32_t alloc_gpu_slot() { return -1; }
    virtual void free_gpu_slot(int32_t /*slot*/) {}

    // Byte copies. P0 stubs are no-ops so unit tests stay GPU-free.
    virtual void copy_block_to_host(uint32_t /*block_id*/,
                                    int32_t /*gpu_slot*/,
                                    void * /*host*/,
                                    uint64_t /*bytes*/) {}
    virtual void copy_block_from_host(uint32_t /*block_id*/,
                                      int32_t /*gpu_slot*/,
                                      const void * /*host*/,
                                      uint64_t /*bytes*/) {}
};

} // namespace kvmem
