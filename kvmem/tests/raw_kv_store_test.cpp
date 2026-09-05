#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    raw.write_layer_tokens(0, 2, 0, k.data(), nullptr);
    CHECK(raw.n_tokens(0) == 2);
    CHECK(raw.has_k(0, 0));
    CHECK(!raw.has_v(0, 0));
    // Mean-K is captured at write so RAM scoring does not copy_k.
    std::vector<float> got(8, -1.0f);
    CHECK(raw.copy_k(0, 0, got.data()));
    CHECK(got[0] == 0.0f);
    CHECK(got[5] == 5.0f);
    CHECK(raw.bytes_v() == 0);
    CHECK(raw.bytes_k() > 0);

    raw.write_layer_tokens(0, 2, 0, nullptr, v.data());
    CHECK(raw.has_v(0, 0));
    std::vector<float> gv(8, 0.0f);
    CHECK(raw.copy_v(0, 0, gv.data()));
    CHECK(std::fabs(gv[0] - 1.0f) < 1e-3f);

    std::vector<float> mean(4, 0.0f);
    raw.mean_k(0, 0, mean.data());
    CHECK(std::fabs(mean[0] - 2.0f) < 1e-3f); // (0+4)/2
    CHECK(std::fabs(mean[1] - 3.0f) < 1e-3f); // (1+5)/2

    std::vector<float> k2(8, 0.0f);
    for (int i = 0; i < 8; ++i) {
        k2[i] = static_cast<float>(i + 8);
    }
    raw.write_layer_tokens(2, 2, 0, k2.data(), nullptr);
    CHECK(raw.n_tokens(0) == 4);
    raw.mean_k(0, 0, mean.data());
    CHECK(std::fabs(mean[0] - 6.0f) < 1e-3f); // (0+4+8+12)/4
    CHECK(std::fabs(mean[1] - 7.0f) < 1e-3f); // (1+5+9+13)/4

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
    raw_h.write_layer_tokens(0, 2, 1, k.data(), nullptr);
    CHECK(raw_h.has_block(0));
    CHECK(raw_h.n_tokens(0) == 2);
    CHECK(!raw_h.has_k(0, 0));
    CHECK(raw_h.has_k(0, 1));
    std::vector<float> hgot(8, -1.0f);
    CHECK(raw_h.copy_k(0, 1, hgot.data()));
    CHECK(hgot[0] == 0.0f);

    kvmem::RawKvStoreConfig ncfg = cfg;
    ncfg.nvme_dir = "/tmp/kvmem_raw_k_test";
    ncfg.nvme_file = "raw.bin";
    ncfg.nvme_bytes = 4ull * 1024ull * 1024ull;
    kvmem::RawKvStore rawn(ncfg);
    CHECK(rawn.nvme_enabled());
    std::vector<float> kfull(16, 0.0f);
    for (int i = 0; i < 16; ++i) {
        kfull[i] = static_cast<float>(i);
    }
    rawn.write_layer_tokens(0, 4, 0, kfull.data(), nullptr);
    CHECK(rawn.has_k(0, 0));
    std::vector<float> nout(16, -1.0f);
    CHECK(rawn.copy_k(0, 0, nout.data()));
    CHECK(nout[0] == 0.0f);
    CHECK(nout[15] == 15.0f);
    std::vector<float> nmean(4, 0.0f);
    rawn.mean_k(0, 0, nmean.data());
    CHECK(std::fabs(nmean[0] - 6.0f) < 1e-2f); // (0+4+8+12)/4

    // F16 write: 0x3c00 is 1.0 in IEEE half.
    kvmem::RawKvStore raw16(cfg);
    std::vector<uint16_t> ones(8, 0x3c00);
    raw16.write_layer_tokens_f16(0, 2, 0, ones.data(), nullptr);
    std::vector<float> f16out(8, 0.0f);
    CHECK(raw16.copy_k(0, 0, f16out.data()));
    CHECK(std::fabs(f16out[0] - 1.0f) < 1e-3f);
    CHECK(std::fabs(f16out[7] - 1.0f) < 1e-3f);
    std::vector<float> m16(4, 0.0f);
    raw16.mean_k(0, 0, m16.data());
    CHECK(std::fabs(m16[0] - 1.0f) < 1e-3f);
    CHECK(std::fabs(m16[3] - 1.0f) < 1e-3f);
    return 0;
}
