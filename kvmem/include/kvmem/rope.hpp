#pragma once

// Host RoPE (NeoX layout). Used to bake raw-K into a cache slot without
// going through the model graph. Matches ggml GGML_ROPE_TYPE_NEOX.

#include <cstdint>
#include <vector>

namespace kvmem {

struct RopeConfig {
    uint32_t n_rot = 128;
    uint32_t n_embd_head = 128;
    uint32_t n_head_kv = 8;
    float freq_base = 1000000.0f;
    float freq_scale = 1.0f;
};

// k_prerope: [n_tokens, n_head_kv, n_embd_head] row-major (token major).
// out: same layout, rotated at positions pos0, pos0+1, ...
void rope_neox_apply(const RopeConfig & cfg,
                     const float * k_prerope,
                     uint32_t n_tokens,
                     int32_t pos0,
                     float * out);

} // namespace kvmem
