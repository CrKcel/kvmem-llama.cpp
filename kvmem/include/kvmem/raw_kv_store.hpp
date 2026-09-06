#pragma once

// CPU/NVMe immutable raw-K (pre-RoPE). Indexed by logical block, then layer.
//
// K is FP16 authority. Mean-K is captured on every K write so RAM scoring is
// a memcpy of the cached mean. When NVMe is configured, completed blocks are
// flushed to SSD and dropped from RAM; the mean stays in RAM.
// GPU-format V (qw3-style): stage-out copies packed cache rows as-is
// (`v_gpu_row_bytes`); optional FP16 V is the harvest-v / F32 path.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace kvmem {

class NvmeKvTier;

struct RawKvStoreConfig {
    uint32_t n_layer = 0;
    uint32_t n_embd_k = 0;
    uint32_t n_embd_v = 0;
    uint32_t block_tokens = 128;
    uint64_t v_gpu_row_bytes = 0;  // 0 = no packed GPU V; else bytes per token row
    uint64_t nvme_bytes = 0;
    std::string nvme_dir;
    std::string nvme_file = "kvmem_raw_k.bin";
};

class RawKvStore {
public:
    explicit RawKvStore(RawKvStoreConfig cfg);
    ~RawKvStore();

    RawKvStore(const RawKvStore &) = delete;
    RawKvStore &operator=(const RawKvStore &) = delete;

    const RawKvStoreConfig &config() const { return cfg_; }
    bool nvme_enabled() const;

    void ensure_blocks(uint32_t block_count);

    void write_layer_tokens(uint32_t pos0, uint32_t n, uint32_t il,
                            const float * k, const float * v);
    void write_layer_tokens_f16(uint32_t pos0, uint32_t n, uint32_t il,
                                const uint16_t * k, const uint16_t * v);
    // Packed GPU-cache V rows (same dtype/layout as the working V tensor).
    void write_layer_v_gpu(uint32_t pos0, uint32_t n, uint32_t il, const uint8_t * v);

    bool has_block(uint32_t block_id) const;
    bool has_k(uint32_t block_id, uint32_t il) const;
    bool has_v(uint32_t block_id, uint32_t il) const;
    uint32_t n_tokens(uint32_t block_id) const;

    bool copy_k(uint32_t block_id, uint32_t il, float * out) const;
    bool copy_v(uint32_t block_id, uint32_t il, float * out) const;
    bool copy_v_gpu(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;

    void mean_k(uint32_t block_id, uint32_t il, float * out) const;

    size_t bytes_k() const;
    size_t bytes_v() const;

    uint64_t nvme_bytes_written() const;
    uint64_t nvme_syscalls() const;
    uint64_t nvme_wait_ns() const;

    void wait_writes();
    void clear();

private:
    struct LayerBlk {
        uint32_t n_tokens = 0;
        std::vector<uint16_t> k;
        std::vector<uint16_t> v;
        std::vector<uint8_t> v_gpu;
        std::vector<float> mean;
        bool k_on_nvme = false;
        bool v_on_nvme = false;
        bool v_gpu_fmt = false;  // NVMe/RAM V is packed GPU rows, not FP16
        bool k_flushing = false;
        bool v_flushing = false;
    };
    struct BlockRaw {
        std::vector<LayerBlk> layers;
    };

    uint32_t nvme_key(uint32_t block_id, uint32_t il, bool is_v) const;
    uint64_t k_slot_bytes() const;
    uint64_t v_slot_bytes() const;
    uint64_t v_gpu_slot_bytes() const;
    void maybe_flush_k(uint32_t block_id, uint32_t il);
    void maybe_flush_v(uint32_t block_id, uint32_t il);
    void capture_mean(LayerBlk & lb) const;
    bool load_k_nvme(uint32_t block_id, uint32_t il, uint16_t * dst) const;
    bool load_v_nvme(uint32_t block_id, uint32_t il, uint16_t * dst) const;
    bool load_v_gpu_nvme(uint32_t block_id, uint32_t il, uint8_t * dst) const;
    void enqueue_flush(uint32_t key, std::vector<uint8_t> && data, uint64_t bytes,
                       uint32_t block_id, uint32_t il, bool is_v);
    void io_loop();
    bool io_sync_inline() const;

    struct IoJob {
        uint32_t key = 0;
        uint32_t block_id = 0;
        uint32_t il = 0;
        bool is_v = false;
        uint64_t bytes = 0;
        std::vector<uint8_t> data;
    };

    RawKvStoreConfig cfg_;
    std::vector<BlockRaw> blocks_;
    std::unique_ptr<NvmeKvTier> nvme_;
    mutable std::vector<uint16_t> io_;
    mutable std::vector<uint8_t> io8_;
    size_t nvme_k_bytes_ = 0;
    size_t nvme_v_bytes_ = 0;
    uint64_t nvme_syscalls_ = 0;
    uint64_t nvme_wait_ns_ = 0;
    mutable std::mutex mu_;
    mutable std::condition_variable cv_;
    std::vector<IoJob> q_;
    size_t inflight_ = 0;
    std::thread io_thread_;
    std::atomic<bool> stop_io_{false};
};

} // namespace kvmem
