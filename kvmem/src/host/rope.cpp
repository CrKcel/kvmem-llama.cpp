#include "kvmem/rope.hpp"

#include <cmath>
#include <vector>

namespace kvmem {

void rope_neox_apply(const RopeConfig & cfg,
                     const float * k_prerope,
                     uint32_t n_tokens,
                     int32_t pos0,
                     float * out) {
    const uint32_t n_rot = cfg.n_rot;
    const uint32_t n_embd = cfg.n_embd_head;
    const uint32_t n_head = cfg.n_head_kv;
    const uint32_t half = n_rot / 2;

    std::vector<float> theta(half);
    for (uint32_t i = 0; i < half; ++i) {
        theta[i] = std::pow(cfg.freq_base, -2.0f * static_cast<float>(i) / static_cast<float>(n_rot));
        theta[i] *= cfg.freq_scale;
    }

    const uint32_t head_stride = n_embd;
    const uint32_t tok_stride = n_head * n_embd;

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float pos = static_cast<float>(pos0 + static_cast<int32_t>(t));
        for (uint32_t h = 0; h < n_head; ++h) {
            const float * src = k_prerope + t * tok_stride + h * head_stride;
            float * dst = out + t * tok_stride + h * head_stride;
            for (uint32_t i = 0; i < half; ++i) {
                const float c = std::cos(pos * theta[i]);
                const float s = std::sin(pos * theta[i]);
                const float x0 = src[i];
                const float x1 = src[i + half];
                dst[i] = x0 * c - x1 * s;
                dst[i + half] = x0 * s + x1 * c;
            }
            for (uint32_t i = n_rot; i < n_embd; ++i) {
                dst[i] = src[i];
            }
        }
    }
}

} // namespace kvmem
