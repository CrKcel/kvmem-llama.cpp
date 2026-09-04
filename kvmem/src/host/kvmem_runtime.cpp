#include "kvmem/kvmem_runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace kvmem {

KvMemRuntime::KvMemRuntime(KvMemRuntimeConfig cfg, KvMemBackend *backend)
    : cfg_(std::move(cfg)),
      store_(cfg_.store),
      backend_(backend ? backend : &null_backend_) {
    trace_ = std::getenv("KVMEM_TRACE") != nullptr;
    slot_bytes_ = cfg_.store.estimated_block_bytes;
    if (cfg_.cpu_bytes > 0 && slot_bytes_ > 0) {
        PinnedKvTierConfig pcfg;
        pcfg.total_bytes = cfg_.cpu_bytes;
        pcfg.slot_bytes = slot_bytes_;
        cpu_tier_ = std::make_unique<PinnedKvTier>(pcfg);
        if (cpu_tier_->enabled()) {
            cpu_arena_.assign(
                static_cast<size_t>(cpu_tier_->slot_count()) * slot_bytes_, 0);
        }
    }
    if (cfg_.nvme_bytes > 0 && slot_bytes_ > 0 && !cfg_.nvme_dir.empty()) {
        NvmeKvTierConfig ncfg;
        ncfg.dir = cfg_.nvme_dir;
        ncfg.total_bytes = cfg_.nvme_bytes;
        ncfg.slot_bytes = slot_bytes_;
        nvme_tier_ = std::make_unique<NvmeKvTier>(ncfg);
    }
    if (slot_bytes_ > 0) {
        scratch_.assign(static_cast<size_t>(slot_bytes_), 0);
    }
    if (cpu_tier_ || nvme_tier_) {
        fprintf(stderr,
                "KVMEM_TIERS cpu_bytes=%llu cpu_slots=%u nvme_bytes=%llu "
                "nvme_slots=%u nvme_dir=%s slot_bytes=%llu\n",
                (unsigned long long) cfg_.cpu_bytes,
                cpu_tier_ && cpu_tier_->enabled() ? cpu_tier_->slot_count() : 0u,
                (unsigned long long) cfg_.nvme_bytes,
                nvme_tier_ && nvme_tier_->enabled() ? nvme_tier_->slot_count() : 0u,
                cfg_.nvme_dir.empty() ? "-" : cfg_.nvme_dir.c_str(),
                (unsigned long long) slot_bytes_);
    }
}

uint8_t *KvMemRuntime::cpu_ptr(int32_t slot) {
    if (slot < 0 || cpu_arena_.empty()) {
        return nullptr;
    }
    return cpu_arena_.data() + static_cast<size_t>(slot) * slot_bytes_;
}

const uint8_t *KvMemRuntime::cpu_ptr(int32_t slot) const {
    if (slot < 0 || cpu_arena_.empty()) {
        return nullptr;
    }
    return cpu_arena_.data() + static_cast<size_t>(slot) * slot_bytes_;
}

void KvMemRuntime::trace_tier(const char *tag, uint32_t block_id, int32_t slot) const {
    if (!trace_) {
        return;
    }
    fprintf(stderr, "KVMEM_TRACE %s block=%u slot=%d\n", tag, block_id, slot);
}

void KvMemRuntime::register_append(uint32_t n_tokens) {
    store_.register_append(n_tokens);
}

std::vector<KvMemDroppedBlock> KvMemRuntime::truncate_to(uint32_t token_pos) {
    wait_prefetch();
    prefetch_buf_.clear();
    auto dropped = store_.truncate_to(token_pos);
    for (const auto &d : dropped) {
        if (cpu_tier_ && d.cpu_slot >= 0) {
            cpu_tier_->release_block(d.block_id);
        }
        if (nvme_tier_ && d.nvme_slot >= 0) {
            nvme_tier_->release_block(d.block_id);
        }
        if (d.gpu_slot >= 0) {
            backend_->free_gpu_slot(d.gpu_slot);
        }
    }
    return dropped;
}

KvMemPlan KvMemRuntime::prepare_reselect(
        const std::vector<uint32_t> &mandatory, bool force_raw_refresh) {
    last_plan_ = store_.set_selection(store_.pick_topk_blocks(mandatory),
                                      force_raw_refresh);
    pending_ = true;
    start_prefetch();
    return last_plan_;
}

KvMemPlan KvMemRuntime::prepare_prefill_pressure(
        const std::vector<uint32_t> &mandatory) {
    last_plan_ = store_.set_selection(
        store_.pick_prefill_pressure_blocks(mandatory));
    pending_ = true;
    start_prefetch();
    return last_plan_;
}

bool KvMemRuntime::maybe_offload_during_prefill(
        uint32_t incoming_tokens,
        uint32_t resident_tokens,
        uint32_t pool_tokens,
        const std::vector<uint32_t> &mandatory) {
    if (!store_.prefill_needs_offload(resident_tokens, incoming_tokens, pool_tokens)) {
        return false;
    }
    prepare_prefill_pressure(mandatory);
    return true;
}

void KvMemRuntime::start_prefetch() {
    wait_prefetch();
    prefetch_buf_.clear();
    if (!nvme_tier_ || !nvme_tier_->enabled() || slot_bytes_ == 0) {
        return;
    }
    for (uint32_t id : last_plan_.stage_in) {
        if (id >= store_.block_count()) {
            continue;
        }
        const KvMemBlock &b = store_.blocks()[id];
        if (b.nvme_slot < 0 && b.tier != KvTier::SSD) {
            continue;
        }
        auto buf = std::make_shared<std::vector<uint8_t>>(
            static_cast<size_t>(slot_bytes_), 0);
        prefetch_buf_[id] = buf;
        NvmeKvTier *tier = nvme_tier_.get();
        const uint64_t nbytes = slot_bytes_;
        prefetch_futs_.push_back(std::async(std::launch::async,
            [tier, id, buf, nbytes]() {
                tier->read_block(id, buf->data(), nbytes);
            }));
    }
    if (trace_ && !prefetch_buf_.empty()) {
        fprintf(stderr, "KVMEM_TRACE prefetch_nvme n=%zu\n", prefetch_buf_.size());
    }
}

void KvMemRuntime::wait_prefetch() {
    std::exception_ptr err;
    for (auto &fut : prefetch_futs_) {
        if (!fut.valid()) {
            continue;
        }
        try {
            fut.get();
        } catch (...) {
            if (!err) {
                err = std::current_exception();
            }
        }
    }
    prefetch_futs_.clear();
    if (err) {
        std::rethrow_exception(err);
    }
}

bool KvMemRuntime::spill_bytes_to_nvme(uint32_t block_id, const void *data) {
    if (!nvme_tier_ || !nvme_tier_->enabled() || !data || slot_bytes_ == 0) {
        return false;
    }
    auto np = nvme_tier_->place_block_evicting(block_id);
    if (np.slot < 0) {
        return false;
    }
    if (np.evicted_block >= 0) {
        store_.set_block_ssd_backing(static_cast<uint32_t>(np.evicted_block), -1, false);
        trace_tier("nvme_evict", static_cast<uint32_t>(np.evicted_block), -1);
    }
    if (data && cfg_.store.select_method == KvMemMethod::Retrieval) {
        nvme_tier_->write_slot(np.slot, data, slot_bytes_);
    }
    store_.set_block_ssd_backing(block_id, np.slot, true);
    trace_tier("stage_out_nvme", block_id, np.slot);
    return true;
}

void KvMemRuntime::spill_outgoing() {
    if (!pending_) {
        return;
    }
    wait_prefetch();
    pending_gpu_frees_.clear();
    for (uint32_t id : last_plan_.stage_out) {
        stage_out(id);
    }
}

void KvMemRuntime::admit_incoming() {
    if (!pending_) {
        return;
    }
    for (int32_t slot : pending_gpu_frees_) {
        backend_->free_gpu_slot(slot);
    }
    pending_gpu_frees_.clear();
    for (uint32_t id : last_plan_.stage_in) {
        stage_in(id);
    }
    prefetch_buf_.clear();
    pending_ = false;
}

void KvMemRuntime::finish_reselect() {
    spill_outgoing();
    admit_incoming();
}

void KvMemRuntime::stage_out(uint32_t block_id) {
    if (block_id >= store_.block_count()) return;
    const int32_t gpu_slot = store_.blocks()[block_id].gpu_slot;
    const int32_t prev_nvme = store_.blocks()[block_id].nvme_slot;

    const bool persist = cfg_.store.select_method == KvMemMethod::Retrieval;
    if (persist && gpu_slot >= 0 && slot_bytes_ > 0 && !scratch_.empty()) {
        backend_->copy_block_to_host(block_id, gpu_slot, scratch_.data(), slot_bytes_);
    }

    if (cpu_tier_ && cpu_tier_->enabled() && slot_bytes_ > 0 && !cpu_arena_.empty()) {
        auto p = cpu_tier_->place_block_evicting(block_id);
        if (p.slot >= 0) {
            if (p.evicted_block >= 0) {
                const uint32_t vic = static_cast<uint32_t>(p.evicted_block);
                (void) spill_bytes_to_nvme(vic, cpu_ptr(p.slot));
                store_.set_block_cpu_copy(vic, -1);
                if (store_.blocks()[vic].tier == KvTier::CPU) {
                    const int32_t vs = store_.blocks()[vic].nvme_slot;
                    store_.set_block_tier(vic, vs >= 0 ? KvTier::SSD : KvTier::CPU, -1, vs);
                }
            }
            if (!scratch_.empty()) {
                std::memcpy(cpu_ptr(p.slot), scratch_.data(), static_cast<size_t>(slot_bytes_));
            }
            store_.set_block_tier(block_id, KvTier::CPU, p.slot, prev_nvme);
            store_.set_block_cpu_copy(block_id, p.slot);
            if (gpu_slot >= 0) {
                pending_gpu_frees_.push_back(gpu_slot);
            }
            trace_tier("stage_out_cpu", block_id, p.slot);
            return;
        }
    }
    if (spill_bytes_to_nvme(block_id, scratch_.empty() ? nullptr : scratch_.data())) {
        store_.set_block_tier(block_id, KvTier::SSD, -1,
                              store_.blocks()[block_id].nvme_slot);
        if (gpu_slot >= 0) {
            pending_gpu_frees_.push_back(gpu_slot);
        }
        return;
    }
    // No lower-tier capacity: drop GPU residency.
    store_.set_block_tier(block_id, KvTier::CPU, -1, prev_nvme);
    if (gpu_slot >= 0) {
        pending_gpu_frees_.push_back(gpu_slot);
    }
}

void KvMemRuntime::stage_in(uint32_t block_id) {
    if (block_id >= store_.block_count()) return;
    const KvMemBlock b = store_.blocks()[block_id];
    if (b.tier == KvTier::GPU && b.gpu_slot >= 0) {
        return;
    }
    const int32_t gpu = backend_->alloc_gpu_slot();
    const uint8_t *src = nullptr;
    bool from_nvme = false;
    auto pit = prefetch_buf_.find(block_id);
    if (pit != prefetch_buf_.end() && pit->second) {
        src = pit->second->data();
        from_nvme = true;
    } else if (b.cpu_slot >= 0) {
        src = cpu_ptr(b.cpu_slot);
    } else if (nvme_tier_ && nvme_tier_->enabled() &&
               (b.nvme_slot >= 0 || b.tier == KvTier::SSD) &&
               slot_bytes_ > 0 && !scratch_.empty()) {
        nvme_tier_->read_block(block_id, scratch_.data(), slot_bytes_);
        src = scratch_.data();
        from_nvme = true;
    }
    if (src && gpu >= 0 && slot_bytes_ > 0) {
        backend_->copy_block_from_host(block_id, gpu, src, slot_bytes_);
    }
    if (cpu_tier_ && b.cpu_slot >= 0) {
        cpu_tier_->release_block(block_id);
        store_.set_block_cpu_copy(block_id, -1);
    }
    store_.set_block_tier(block_id, KvTier::GPU, -1, store_.blocks()[block_id].nvme_slot);
    store_.set_block_gpu_slot(block_id, gpu);
    if (from_nvme) {
        trace_tier("stage_in_nvme", block_id, gpu);
    }
}

} // namespace kvmem
