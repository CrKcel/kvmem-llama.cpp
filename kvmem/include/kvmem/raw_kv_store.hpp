#pragma once

// CPU-side immutable raw-K (pre-RoPE) and V. Indexed by logical block, then
// layer. Adapter writes captured ubatches here and reads them back on stage-in.

#include <cstdint>
#include <vector>

namespace kvmem {

struct RawKvStoreConfig {
    uint32_t n_layer = 0;
    uint32_t n_embd_k = 0; // n_embd_head_k * n_head_kv
    uint32_t n_embd_v = 0;
    uint32_t block_tokens = 128;
};

class RawKvStore {
public:
    explicit RawKvStore(RawKvStoreConfig cfg);

    const RawKvStoreConfig &config() const { return cfg_; }

    void ensure_blocks(uint32_t block_count);

    // Write `n` tokens starting at original position `pos0` for one layer.
    // k/v are [n, n_embd_*] token-major f32. Partial writes extend the block.
    void write_layer_tokens(uint32_t pos0, uint32_t n, uint32_t il,
                            const float * k, const float * v);

    bool has_block(uint32_t block_id) const;
    uint32_t n_tokens(uint32_t block_id) const;

    // Pointer to [n_tokens, n_embd_k] for (block, layer). Null if missing.
    const float * k(uint32_t block_id, uint32_t il) const;
    const float * v(uint32_t block_id, uint32_t il) const;

    // Mean-K over tokens in the block, per layer. out size n_embd_k.
    void mean_k(uint32_t block_id, uint32_t il, float * out) const;

    void clear();

private:
    struct LayerBlk {
        uint32_t n_tokens = 0;
        std::vector<float> k;
        std::vector<float> v;
    };
    struct BlockRaw {
        std::vector<LayerBlk> layers;
    };

    RawKvStoreConfig cfg_;
    std::vector<BlockRaw> blocks_;
};

} // namespace kvmem
