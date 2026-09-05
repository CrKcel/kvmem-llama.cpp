#include "kvmem/raw_kv_store.hpp"
#include "kvmem/nvme_kv_tier.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <time.h>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace {
uint64_t monotonic_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}
} // namespace

namespace kvmem {
namespace {

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if ((x & 0x7fffffffu) == 0) {
        return static_cast<uint16_t>(sign);
    }
    if (((x >> 23) & 0xffu) == 0xffu) {
        return static_cast<uint16_t>(sign | 0x7c00u | (man ? 0x200u : 0));
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        man |= 0x800000u;
        const uint32_t t = static_cast<uint32_t>(14 - exp);
        uint32_t half = man >> t;
        const uint32_t rem = man & ((1u << t) - 1u);
        if (rem > (1u << (t - 1)) || (rem == (1u << (t - 1)) && (half & 1u))) {
            half += 1;
        }
        return static_cast<uint16_t>(sign | half);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint32_t half = (static_cast<uint32_t>(exp) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
        half += 1;
    }
    return static_cast<uint16_t>(sign | half);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x3ffu;
            out = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

void pack_f32(const float * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f32_to_f16(src[i]);
    }
}

void unpack_f16(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f16_to_f32(src[i]);
    }
}

} // namespace

RawKvStore::RawKvStore(RawKvStoreConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.nvme_bytes > 0 && !cfg_.nvme_dir.empty() && cfg_.block_tokens > 0) {
        NvmeKvTierConfig ncfg;
        ncfg.dir = cfg_.nvme_dir;
        ncfg.file_name = cfg_.nvme_file.empty() ? "kvmem_raw_k.bin" : cfg_.nvme_file;
        ncfg.total_bytes = cfg_.nvme_bytes;
        ncfg.slot_bytes = std::max(k_slot_bytes(), v_slot_bytes());
        ncfg.drop_page_cache = true;
        if (ncfg.slot_bytes == 0) {
            return;
        }
        nvme_ = std::make_unique<NvmeKvTier>(ncfg);
        if (nvme_->enabled()) {
            std::fprintf(stderr,
                         "KVMEM_RAW_NVME dir=%s file=%s bytes=%llu slots=%u slot_bytes=%llu\n",
                         cfg_.nvme_dir.c_str(), ncfg.file_name.c_str(),
                         (unsigned long long) cfg_.nvme_bytes,
                         nvme_->slot_count(),
                         (unsigned long long) ncfg.slot_bytes);
            if (!io_sync_inline()) {
                io_thread_ = std::thread([this] { io_loop(); });
            }
        }
    }
}

RawKvStore::~RawKvStore() {
    stop_io_.store(true, std::memory_order_release);
    cv_.notify_all();
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
}

bool RawKvStore::io_sync_inline() const {
    const char * e = std::getenv("KVMEM_HARVEST_SYNC");
    return e && e[0] != '\0' && e[0] != '0';
}

bool RawKvStore::nvme_enabled() const {
    return nvme_ && nvme_->enabled();
}

uint32_t RawKvStore::nvme_key(uint32_t block_id, uint32_t il, bool is_v) const {
    return block_id * (cfg_.n_layer * 2u + 2u) + il * 2u + (is_v ? 1u : 0u);
}

uint64_t RawKvStore::k_slot_bytes() const {
    return static_cast<uint64_t>(cfg_.block_tokens) * cfg_.n_embd_k * sizeof(uint16_t);
}

uint64_t RawKvStore::v_slot_bytes() const {
    return static_cast<uint64_t>(cfg_.block_tokens) * cfg_.n_embd_v * sizeof(uint16_t);
}

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

void RawKvStore::capture_mean(LayerBlk & lb) const {
    if (lb.k.empty() || cfg_.n_embd_k == 0) {
        return;
    }
    const uint32_t nt = lb.n_tokens;
    if (nt == 0) {
        return;
    }
    lb.mean.assign(cfg_.n_embd_k, 0.0f);
    for (uint32_t t = 0; t < nt; ++t) {
        const uint16_t * row = lb.k.data() + t * cfg_.n_embd_k;
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
            lb.mean[d] += f16_to_f32(row[d]);
        }
    }
    const float inv = 1.0f / static_cast<float>(nt);
    for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
        lb.mean[d] *= inv;
    }
}

void RawKvStore::enqueue_flush(uint32_t key, std::vector<uint16_t> && data,
                               uint64_t bytes, uint32_t block_id, uint32_t il,
                               bool is_v) {
    IoJob job;
    job.key = key;
    job.block_id = block_id;
    job.il = il;
    job.is_v = is_v;
    job.bytes = bytes;
    job.data = std::move(data);
    q_.push_back(std::move(job));
    cv_.notify_one();
}

void RawKvStore::maybe_flush_k(uint32_t block_id, uint32_t il) {
    if (!nvme_enabled() || block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return;
    }
    LayerBlk & lb = blocks_[block_id].layers[il];
    if (lb.k_on_nvme || lb.k_flushing || lb.k.empty() ||
        lb.n_tokens < cfg_.block_tokens) {
        return;
    }
    capture_mean(lb);
    const uint64_t nbytes = k_slot_bytes();
    if (lb.k.size() * sizeof(uint16_t) < nbytes) {
        return;
    }
    if (io_sync_inline() || !io_thread_.joinable()) {
        const uint64_t t0 = monotonic_ns();
        nvme_->write_block(nvme_key(block_id, il, false), lb.k.data(), nbytes);
        nvme_wait_ns_ += monotonic_ns() - t0;
        nvme_syscalls_ += 1;
        nvme_k_bytes_ += nbytes;
        lb.k.clear();
        lb.k.shrink_to_fit();
        lb.k_on_nvme = true;
        return;
    }
    lb.k_flushing = true;
    enqueue_flush(nvme_key(block_id, il, false), std::move(lb.k), nbytes,
                  block_id, il, false);
    lb.k.clear();
    lb.k.shrink_to_fit();
}

void RawKvStore::maybe_flush_v(uint32_t block_id, uint32_t il) {
    if (!nvme_enabled() || block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return;
    }
    LayerBlk & lb = blocks_[block_id].layers[il];
    if (lb.v_on_nvme || lb.v_flushing || lb.v.empty() ||
        lb.n_tokens < cfg_.block_tokens) {
        return;
    }
    const uint64_t nbytes = v_slot_bytes();
    if (lb.v.size() * sizeof(uint16_t) < nbytes) {
        return;
    }
    if (io_sync_inline() || !io_thread_.joinable()) {
        const uint64_t t0 = monotonic_ns();
        nvme_->write_block(nvme_key(block_id, il, true), lb.v.data(), nbytes);
        nvme_wait_ns_ += monotonic_ns() - t0;
        nvme_syscalls_ += 1;
        nvme_v_bytes_ += nbytes;
        lb.v.clear();
        lb.v.shrink_to_fit();
        lb.v_on_nvme = true;
        return;
    }
    lb.v_flushing = true;
    enqueue_flush(nvme_key(block_id, il, true), std::move(lb.v), nbytes,
                  block_id, il, true);
    lb.v.clear();
    lb.v.shrink_to_fit();
}

void RawKvStore::io_loop() {
    while (true) {
        std::vector<IoJob> batch;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] {
                return stop_io_.load(std::memory_order_acquire) || !q_.empty();
            });
            if (q_.empty() && stop_io_.load(std::memory_order_acquire)) {
                break;
            }
            batch.swap(q_);
            inflight_ = batch.size();
        }
        if (batch.empty() || !nvme_) {
            continue;
        }
        std::vector<NvmeIoSpan> spans;
        spans.reserve(batch.size());
        uint64_t slab_bytes = 0;
        for (auto & job : batch) {
            auto p = nvme_->place_block(job.key);
            if (p.slot < 0) {
                continue;
            }
            NvmeIoSpan sp;
            sp.slot = p.slot;
            sp.buffer_offset = slab_bytes;
            sp.bytes = job.bytes;
            spans.push_back(sp);
            slab_bytes += job.bytes;
        }
        std::vector<size_t> order(spans.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return spans[a].slot < spans[b].slot;
        });
        std::vector<uint8_t> slab(slab_bytes);
        std::vector<NvmeIoSpan> sorted;
        sorted.reserve(spans.size());
        uint64_t off = 0;
        for (size_t idx : order) {
            NvmeIoSpan sp = spans[idx];
            std::memcpy(slab.data() + off, batch[idx].data.data(),
                        static_cast<size_t>(batch[idx].bytes));
            sp.buffer_offset = off;
            sorted.push_back(sp);
            off += batch[idx].bytes;
        }
        NvmeBatchIoStats st;
        const uint64_t t0 = monotonic_ns();
        nvme_->write_spans(sorted, slab.data(), slab.size(), &st);
        const uint64_t dt = monotonic_ns() - t0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            nvme_wait_ns_ += st.duration_ns ? st.duration_ns : dt;
            nvme_syscalls_ += st.syscalls ? st.syscalls : sorted.size();
            for (auto & job : batch) {
                if (job.block_id >= blocks_.size() || job.il >= cfg_.n_layer) {
                    continue;
                }
                LayerBlk & lb = blocks_[job.block_id].layers[job.il];
                if (job.is_v) {
                    lb.v_flushing = false;
                    lb.v_on_nvme = true;
                    nvme_v_bytes_ += job.bytes;
                } else {
                    lb.k_flushing = false;
                    lb.k_on_nvme = true;
                    nvme_k_bytes_ += job.bytes;
                }
            }
            inflight_ = 0;
            cv_.notify_all();
        }
    }
}

void RawKvStore::wait_writes() {
    if (!nvme_enabled() || io_sync_inline() || !io_thread_.joinable()) {
        return;
    }
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return q_.empty() && inflight_ == 0; });
}

void RawKvStore::write_layer_tokens(uint32_t pos0, uint32_t n, uint32_t il,
                                    const float * k, const float * v) {
    std::unique_lock<std::mutex> lk(mu_);
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
        if (k) {
            const size_t need = static_cast<size_t>(bt) * cfg_.n_embd_k;
            if (lb.k.size() < need) {
                if (lb.k_on_nvme) {
                    lb.k.assign(need, 0);
                    load_k_nvme(bid, il, lb.k.data());
                    lb.k_on_nvme = false;
                } else {
                    lb.k.assign(need, 0);
                }
            }
            pack_f32(k + done * cfg_.n_embd_k,
                     lb.k.data() + off * cfg_.n_embd_k,
                     static_cast<size_t>(take) * cfg_.n_embd_k);
        }
        if (v) {
            const size_t need = static_cast<size_t>(bt) * cfg_.n_embd_v;
            if (lb.v.size() < need) {
                if (lb.v_on_nvme) {
                    lb.v.assign(need, 0);
                    load_v_nvme(bid, il, lb.v.data());
                    lb.v_on_nvme = false;
                } else {
                    lb.v.assign(need, 0);
                }
            }
            pack_f32(v + done * cfg_.n_embd_v,
                     lb.v.data() + off * cfg_.n_embd_v,
                     static_cast<size_t>(take) * cfg_.n_embd_v);
        }
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        if (k) {
            capture_mean(lb);
        }
        maybe_flush_k(bid, il);
        maybe_flush_v(bid, il);
        done += take;
    }
}

void RawKvStore::write_layer_tokens_f16(uint32_t pos0, uint32_t n, uint32_t il,
                                        const uint16_t * k, const uint16_t * v) {
    std::unique_lock<std::mutex> lk(mu_);
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
        if (k) {
            const size_t need = static_cast<size_t>(bt) * cfg_.n_embd_k;
            if (lb.k.size() < need) {
                if (lb.k_on_nvme) {
                    lb.k.assign(need, 0);
                    load_k_nvme(bid, il, lb.k.data());
                    lb.k_on_nvme = false;
                } else {
                    lb.k.assign(need, 0);
                }
            }
            std::memcpy(lb.k.data() + off * cfg_.n_embd_k,
                        k + done * cfg_.n_embd_k,
                        static_cast<size_t>(take) * cfg_.n_embd_k * sizeof(uint16_t));
        }
        if (v) {
            const size_t need = static_cast<size_t>(bt) * cfg_.n_embd_v;
            if (lb.v.size() < need) {
                if (lb.v_on_nvme) {
                    lb.v.assign(need, 0);
                    load_v_nvme(bid, il, lb.v.data());
                    lb.v_on_nvme = false;
                } else {
                    lb.v.assign(need, 0);
                }
            }
            std::memcpy(lb.v.data() + off * cfg_.n_embd_v,
                        v + done * cfg_.n_embd_v,
                        static_cast<size_t>(take) * cfg_.n_embd_v * sizeof(uint16_t));
        }
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        if (k) {
            capture_mean(lb);
        }
        maybe_flush_k(bid, il);
        maybe_flush_v(bid, il);
        done += take;
    }
}

bool RawKvStore::has_block(uint32_t block_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size()) {
        return false;
    }
    for (const auto & lb : blocks_[block_id].layers) {
        if (lb.n_tokens > 0 &&
            (!lb.k.empty() || lb.k_on_nvme || lb.k_flushing)) {
            return true;
        }
    }
    return false;
}

bool RawKvStore::has_k(uint32_t block_id, uint32_t il) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return !lb.k.empty() || lb.k_on_nvme || lb.k_flushing;
}

bool RawKvStore::has_v(uint32_t block_id, uint32_t il) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return !lb.v.empty() || lb.v_on_nvme || lb.v_flushing;
}

uint32_t RawKvStore::n_tokens(uint32_t block_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size()) {
        return 0;
    }
    uint32_t n = 0;
    for (const auto & lb : blocks_[block_id].layers) {
        n = std::max(n, lb.n_tokens);
    }
    return n;
}

bool RawKvStore::load_k_nvme(uint32_t block_id, uint32_t il, uint16_t * dst) const {
    if (!nvme_enabled() || !dst) {
        return false;
    }
    try {
        nvme_->read_block(nvme_key(block_id, il, false), dst, k_slot_bytes());
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool RawKvStore::load_v_nvme(uint32_t block_id, uint32_t il, uint16_t * dst) const {
    if (!nvme_enabled() || !dst) {
        return false;
    }
    try {
        nvme_->read_block(nvme_key(block_id, il, true), dst, v_slot_bytes());
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool RawKvStore::copy_k(uint32_t block_id, uint32_t il, float * out) const {
    if (!out) {
        return false;
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    cv_.wait(lk, [&] {
        const LayerBlk & x = blocks_[block_id].layers[il];
        return !x.k_flushing;
    });
    const LayerBlk & lb = blocks_[block_id].layers[il];
    if (lb.k.empty() && !lb.k_on_nvme) {
        return false;
    }
    const uint32_t nt = lb.n_tokens;
    const uint16_t * src = nullptr;
    if (!lb.k.empty()) {
        src = lb.k.data();
    } else {
        io_.assign(static_cast<size_t>(cfg_.block_tokens) * cfg_.n_embd_k, 0);
        if (!load_k_nvme(block_id, il, io_.data())) {
            return false;
        }
        src = io_.data();
    }
    unpack_f16(src, out, static_cast<size_t>(nt) * cfg_.n_embd_k);
    return true;
}

bool RawKvStore::copy_v(uint32_t block_id, uint32_t il, float * out) const {
    if (!out) {
        return false;
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    cv_.wait(lk, [&] {
        const LayerBlk & x = blocks_[block_id].layers[il];
        return !x.v_flushing;
    });
    const LayerBlk & lb = blocks_[block_id].layers[il];
    if (lb.v.empty() && !lb.v_on_nvme) {
        return false;
    }
    const uint32_t nt = lb.n_tokens;
    const uint16_t * src = nullptr;
    if (!lb.v.empty()) {
        src = lb.v.data();
    } else {
        io_.assign(static_cast<size_t>(cfg_.block_tokens) * cfg_.n_embd_v, 0);
        if (!load_v_nvme(block_id, il, io_.data())) {
            return false;
        }
        src = io_.data();
    }
    unpack_f16(src, out, static_cast<size_t>(nt) * cfg_.n_embd_v);
    return true;
}

void RawKvStore::mean_k(uint32_t block_id, uint32_t il, float * out) const {
    if (!out) {
        return;
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    if (lb.k.empty() && !lb.k_on_nvme && !lb.k_flushing) {
        return;
    }
    if (lb.mean.size() == cfg_.n_embd_k) {
        std::memcpy(out, lb.mean.data(), cfg_.n_embd_k * sizeof(float));
        return;
    }
    cv_.wait(lk, [&] {
        const LayerBlk & x = blocks_[block_id].layers[il];
        return !x.k_flushing;
    });
    if (lb.mean.size() == cfg_.n_embd_k) {
        std::memcpy(out, lb.mean.data(), cfg_.n_embd_k * sizeof(float));
        return;
    }
    const uint32_t nt = lb.n_tokens;
    if (nt == 0 || cfg_.n_embd_k == 0) {
        return;
    }
    const uint16_t * src = nullptr;
    if (!lb.k.empty()) {
        src = lb.k.data();
    } else {
        io_.assign(static_cast<size_t>(cfg_.block_tokens) * cfg_.n_embd_k, 0);
        if (!load_k_nvme(block_id, il, io_.data())) {
            return;
        }
        src = io_.data();
    }
    std::fill(out, out + cfg_.n_embd_k, 0.0f);
    for (uint32_t t = 0; t < nt; ++t) {
        const uint16_t * row = src + t * cfg_.n_embd_k;
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
            out[d] += f16_to_f32(row[d]);
        }
    }
    const float inv = 1.0f / static_cast<float>(nt);
    for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
        out[d] *= inv;
    }
}

size_t RawKvStore::bytes_k() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = nvme_k_bytes_;
    for (const auto & b : blocks_) {
        for (const auto & lb : b.layers) {
            n += lb.k.size() * sizeof(uint16_t);
            n += lb.mean.size() * sizeof(float);
        }
    }
    return n;
}

uint64_t RawKvStore::nvme_bytes_written() const {
    std::lock_guard<std::mutex> lk(mu_);
    return nvme_k_bytes_ + nvme_v_bytes_;
}

uint64_t RawKvStore::nvme_syscalls() const {
    std::lock_guard<std::mutex> lk(mu_);
    return nvme_syscalls_;
}

uint64_t RawKvStore::nvme_wait_ns() const {
    std::lock_guard<std::mutex> lk(mu_);
    return nvme_wait_ns_;
}

size_t RawKvStore::bytes_v() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = nvme_v_bytes_;
    for (const auto & b : blocks_) {
        for (const auto & lb : b.layers) {
            n += lb.v.size() * sizeof(uint16_t);
        }
    }
    return n;
}

void RawKvStore::clear() {
    wait_writes();
    std::lock_guard<std::mutex> lk(mu_);
    blocks_.clear();
    nvme_k_bytes_ = 0;
    nvme_v_bytes_ = 0;
    nvme_syscalls_ = 0;
    nvme_wait_ns_ = 0;
    if (nvme_) {
        nvme_->clear();
    }
}

} // namespace kvmem
