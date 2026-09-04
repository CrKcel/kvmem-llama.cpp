#include "kvmem/raw_kv_store.hpp"

#include <algorithm>
#include <cstring>

namespace kvmem {

RawKvStore::RawKvStore(RawKvStoreConfig cfg) : cfg_(cfg) {}

void RawKvStore::ensure_blocks(uint32_t block_count) {
    if (blocks_.size() >= block_count) {
        return;
    }
    const size_t old = blocks_.size();
    blocks_.resize(block_count);
    for (size_t i = old; i < blocks_.size(); ++i) {
        blocks_[i].layers.resize(cfg_.n_layer);
    }
}

void RawKvStore::write_layer_tokens(uint32_t pos0, uint32_t n, uint32_t il,
                                    const float * k, const float * v) {
    if (n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);
        LayerBlk & lb = blocks_[bid].layers[il];
        if (lb.k.size() < static_cast<size_t>(bt) * cfg_.n_embd_k) {
            lb.k.assign(static_cast<size_t>(bt) * cfg_.n_embd_k, 0.0f);
            lb.v.assign(static_cast<size_t>(bt) * cfg_.n_embd_v, 0.0f);
        }
        if (k) {
            std::memcpy(lb.k.data() + off * cfg_.n_embd_k,
                        k + done * cfg_.n_embd_k,
                        take * cfg_.n_embd_k * sizeof(float));
        }
        if (v) {
            std::memcpy(lb.v.data() + off * cfg_.n_embd_v,
                        v + done * cfg_.n_embd_v,
                        take * cfg_.n_embd_v * sizeof(float));
        }
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        done += take;
    }
}

bool RawKvStore::has_block(uint32_t block_id) const {
    if (block_id >= blocks_.size()) {
        return false;
    }
    // Hybrid models only capture attention layers; layer 0 may be GDN.
    for (const auto & lb : blocks_[block_id].layers) {
        if (lb.n_tokens > 0 && !lb.k.empty()) {
            return true;
        }
    }
    return false;
}

uint32_t RawKvStore::n_tokens(uint32_t block_id) const {
    if (block_id >= blocks_.size()) {
        return 0;
    }
    uint32_t n = 0;
    for (const auto & lb : blocks_[block_id].layers) {
        n = std::max(n, lb.n_tokens);
    }
    return n;
}

const float * RawKvStore::k(uint32_t block_id, uint32_t il) const {
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return nullptr;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return lb.k.empty() ? nullptr : lb.k.data();
}

const float * RawKvStore::v(uint32_t block_id, uint32_t il) const {
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return nullptr;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return lb.v.empty() ? nullptr : lb.v.data();
}

void RawKvStore::mean_k(uint32_t block_id, uint32_t il, float * out) const {
    const float * src = k(block_id, il);
    const uint32_t nt = n_tokens(block_id);
    if (!src || nt == 0 || !out) {
        return;
    }
    std::fill(out, out + cfg_.n_embd_k, 0.0f);
    for (uint32_t t = 0; t < nt; ++t) {
        const float * row = src + t * cfg_.n_embd_k;
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
            out[d] += row[d];
        }
    }
    const float inv = 1.0f / static_cast<float>(nt);
    for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
        out[d] *= inv;
    }
}

void RawKvStore::clear() {
    blocks_.clear();
}

} // namespace kvmem
