#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__,         \
                         __LINE__, #cond);                                     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

int main() {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 2;
    cfg.n_embd_k = 4;
    cfg.n_embd_v = 4;
    cfg.block_tokens = 4;
    kvmem::RawKvStore raw(cfg);

    std::vector<float> k(8, 0.0f);
    std::vector<float> v(8, 1.0f);
    for (int i = 0; i < 8; ++i) {
        k[i] = static_cast<float>(i);
    }
    raw.write_layer_tokens(0, 2, 0, k.data(), v.data());
    CHECK(raw.n_tokens(0) == 2);
    CHECK(raw.k(0, 0)[0] == 0.0f);
    CHECK(raw.k(0, 0)[5] == 5.0f);

    std::vector<float> mean(4, 0.0f);
    raw.mean_k(0, 0, mean.data());
    CHECK(std::fabs(mean[0] - 2.0f) < 1e-5f); // (0+4)/2
    CHECK(std::fabs(mean[1] - 3.0f) < 1e-5f); // (1+5)/2

    kvmem::RopeConfig rc;
    rc.n_rot = 4;
    rc.n_embd_head = 4;
    rc.n_head_kv = 1;
    rc.freq_base = 10000.0f;
    std::vector<float> src(4, 0.0f);
    src[0] = 1.0f;
    src[2] = 1.0f;
    std::vector<float> dst(4, 0.0f);
    kvmem::rope_neox_apply(rc, src.data(), 1, 0, dst.data());
    // pos=0 → identity rotation
    CHECK(std::fabs(dst[0] - 1.0f) < 1e-5f);
    CHECK(std::fabs(dst[2] - 1.0f) < 1e-5f);

    kvmem::rope_neox_apply(rc, src.data(), 1, 1, dst.data());
    CHECK(std::fabs(dst[0] - src[0]) > 1e-6f || std::fabs(dst[2] - src[2]) > 1e-6f);

    // Hybrid: only attention layers are captured; layer 0 may stay empty.
    kvmem::RawKvStore raw_h(cfg);
    raw_h.write_layer_tokens(0, 2, 1, k.data(), v.data());
    CHECK(raw_h.has_block(0));
    CHECK(raw_h.n_tokens(0) == 2);
    CHECK(raw_h.k(0, 0) == nullptr);
    CHECK(raw_h.k(0, 1)[0] == 0.0f);
    return 0;
}
