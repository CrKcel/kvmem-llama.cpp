#include "llama-memory-kvmem.h"

#include "llama-kvmem-batch.h"
#include "llama-kvmem-capture.h"
#include "llama-kvmem-factory.h"
#include "llama-kvmem-hooks.h"

#include "llama-arch.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

static llama_kvmem_params g_kvmem_params = {};

struct llama_memory_kvmem::CaptureD2hPipe {
    struct Item {
        int il = 0;
        char which = 0;
        size_t offset = 0;
        size_t nbytes = 0;
        int64_t d = 0;
        int64_t h = 0;
        int64_t n = 0;
        size_t nb0 = 0;
        size_t nb1 = 0;
        size_t nb2 = 0;
        ggml_type type = GGML_TYPE_F16;
        bool from_gpu = false;
    };
    struct Slot {
        uint8_t * gpu = nullptr;
        uint8_t * pin = nullptr;
        size_t cap = 0;
        cudaEvent_t done = nullptr;
        bool inflight = false;
        std::vector<Item> items;
        std::vector<llama_pos> pos;
    };
    cudaStream_t stream = nullptr;
    cudaEvent_t snap = nullptr;
    Slot slots[2];
    int next = 0;
    bool ok = false;
};

static bool kvmem_cuda_ok(cudaError_t e, const char * what) {
    if (e == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "KVMEM D2H %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

void llama_kvmem_set_params(const struct llama_kvmem_params * params) {
    if (params) {
        g_kvmem_params = *params;
    } else {
        g_kvmem_params = {};
    }
}

const struct llama_kvmem_params * llama_kvmem_get_params(void) {
    return &g_kvmem_params;
}

static uint32_t kvmem_align_tokens(uint32_t tokens, uint32_t block_tokens) {
    if (block_tokens == 0) {
        return 0;
    }
    if (tokens < block_tokens) {
        return block_tokens;
    }
    return (tokens / block_tokens) * block_tokens;
}

static double kvmem_default_ratio(float v, double fallback) {
    return v > 0.0f ? static_cast<double>(v) : fallback;
}

static uint64_t kvmem_first_gpu_total_bytes() {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t free_m = 0;
            size_t total_m = 0;
            ggml_backend_dev_memory(dev, &free_m, &total_m);
            return static_cast<uint64_t>(total_m);
        }
    }
    return 0;
}

static kvmem::KvMemRuntimeConfig make_runtime_cfg(
        uint32_t block_tokens,
        uint32_t budget,
        uint32_t sink_tokens,
        uint32_t recent_tokens,
        uint64_t block_bytes) {
    kvmem::KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = block_tokens;
    cfg.store.select_budget = budget;
    cfg.store.prefill_budget = budget;
    cfg.store.select_method = g_kvmem_params.method == 1
        ? kvmem::KvMemMethod::Retrieval
        : kvmem::KvMemMethod::Recency;
    cfg.store.sink_blocks = sink_tokens == 0
        ? 1u
        : std::max(1u, sink_tokens / block_tokens);
    cfg.store.recent_blocks = recent_tokens / block_tokens;
    cfg.store.gen_budget = 0;
    cfg.store.gpu_memory_ratio = kvmem_default_ratio(g_kvmem_params.gpu_memory_ratio, 0.50);
    cfg.store.gpu_high_watermark = kvmem_default_ratio(g_kvmem_params.gpu_high_watermark, 0.95);
    cfg.store.gpu_low_watermark = kvmem_default_ratio(g_kvmem_params.gpu_low_watermark, 0.85);
    cfg.store.estimated_block_bytes = block_bytes;
    cfg.store.gpu_resident_block_bytes = block_bytes;
    cfg.cpu_bytes = g_kvmem_params.cpu_bytes;
    cfg.nvme_bytes = g_kvmem_params.nvme_bytes;
    if (g_kvmem_params.nvme_dir && g_kvmem_params.nvme_dir[0]) {
        cfg.nvme_dir = g_kvmem_params.nvme_dir;
    } else if (cfg.nvme_bytes > 0) {
        cfg.nvme_dir = "/tmp/kvmem_nvme";
    }
    return cfg;
}

llama_memory_i * llama_memory_kvmem_maybe_create(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) {
    if (!g_kvmem_params.enabled) {
        return nullptr;
    }
    if (llm_arch_is_hybrid(model.arch) || llm_arch_is_recurrent(model.arch)) {
        LLAMA_LOG_WARN("%s: KVMem P1 skips hybrid/recurrent arch %s\n",
                __func__, llm_arch_name(model.arch));
        return nullptr;
    }
    if (model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        LLAMA_LOG_WARN("%s: KVMem P1 skips SWA models\n", __func__);
        return nullptr;
    }
    if (cparams.n_seq_max > 1) {
        LLAMA_LOG_WARN("%s: KVMem P1 requires n_seq_max=1 (got %u)\n",
                __func__, cparams.n_seq_max);
        return nullptr;
    }
    return new llama_memory_kvmem(model, params, cparams);
}

llama_memory_kvmem::llama_memory_kvmem(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) :
    model_(model) {
    backend_.owner = this;
    trace_ = getenv("KVMEM_TRACE") != nullptr;

    block_tokens_ = g_kvmem_params.block_tokens ? g_kvmem_params.block_tokens : 32u;
    uint32_t budget = g_kvmem_params.budget;
    if (budget == 0) {
        budget = kvmem_align_tokens(cparams.n_ctx_seq, block_tokens_);
        if (budget == 0) {
            budget = block_tokens_;
        }
    } else {
        budget = kvmem_align_tokens(budget, block_tokens_);
    }

    uint32_t gen_reserve = g_kvmem_params.gen_reserve ? g_kvmem_params.gen_reserve : 256u;
    gen_reserve = kvmem_align_tokens(gen_reserve, block_tokens_);

    const uint32_t n_layer = model.hparams.n_layer();
    const uint32_t n_embd_k = model.hparams.n_embd_k_gqa(0);
    const uint32_t n_embd_v = model.hparams.n_embd_v_gqa(0);
    const uint64_t k_row = ggml_row_size(params.type_k, n_embd_k);
    const uint64_t v_row = ggml_row_size(params.type_v, n_embd_v);
    const uint64_t block_bytes = static_cast<uint64_t>(n_layer) * (k_row + v_row) * block_tokens_;

    const double ratio = kvmem_default_ratio(g_kvmem_params.gpu_memory_ratio, 0.50);
    const uint64_t gpu_total = kvmem_first_gpu_total_bytes();
    uint32_t cap_blocks = 0;
    if (gpu_total > 0 && block_bytes > 0 && ratio > 0.0) {
        cap_blocks = static_cast<uint32_t>(
                (gpu_total * ratio) / std::max(block_bytes, uint64_t{1}));
    }

    uint32_t pool = budget + gen_reserve;
    if (pool > cparams.n_ctx_seq && g_kvmem_params.budget == 0) {
        pool = cparams.n_ctx_seq;
    }
    if (cap_blocks > 0) {
        const uint32_t cap_tokens = cap_blocks * block_tokens_;
        if (pool > cap_tokens) {
            pool = cap_tokens;
            if (budget >= pool) {
                gen_reserve = std::min(gen_reserve, block_tokens_);
                budget = pool > gen_reserve ? pool - gen_reserve : pool;
            } else if (budget + gen_reserve > pool) {
                gen_reserve = pool - budget;
            }
            budget = kvmem_align_tokens(std::max(budget, block_tokens_), block_tokens_);
            gen_reserve = kvmem_align_tokens(std::max(gen_reserve, block_tokens_), block_tokens_);
            pool = budget + gen_reserve;
            if (pool > cap_tokens) {
                pool = cap_tokens;
            }
        }
    }
    kv_size_ = std::max(pool, 1u);
    n_slots_ = (kv_size_ + block_tokens_ - 1) / block_tokens_;
    // Last slot may be a partial block when kv_size is not a multiple of BT.
    if (n_slots_ * block_tokens_ < kv_size_) {
        n_slots_ += 1;
    }
    // llama_kv_cache size is the cell count, not rounded up past kv_size_.
    // If n_slots * BT > kv_size, extra cells in the last slot simply do not exist;
    // fill_slot_info rejects cell >= kv_size.

    auto rt_cfg = make_runtime_cfg(
            block_tokens_, budget, g_kvmem_params.sink_tokens, g_kvmem_params.recent_tokens,
            block_bytes);
    rt_cfg.store.estimated_gpu_block_capacity = cap_blocks;
    runtime_ = std::make_unique<kvmem::KvMemRuntime>(rt_cfg, &backend_);

    kv_ = std::make_unique<llama_kv_cache>(
            model,
            model.hparams,
            params.type_k,
            params.type_v,
            !cparams.flash_attn,
            cparams.offload_kqv,
            /* unified */ true,
            kv_size_,
            /* n_seq_max */ 1,
            /* n_pad */ 1,
            model.hparams.n_swa,
            model.hparams.swa_type,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            "kvmem");

    reset_slots();

    n_layer_ = model.hparams.n_layer();
    n_embd_head_ = model.hparams.n_embd_head_k(0);
    n_head_kv_ = model.hparams.n_head_kv(0);
    n_head_ = model.hparams.n_head(0);
    n_embd_k_ = model.hparams.n_embd_k_gqa(0);
    n_embd_v_ = model.hparams.n_embd_v_gqa(0);
    rope_.n_rot = model.hparams.n_rot(0);
    rope_.n_embd_head = n_embd_head_;
    rope_.n_head_kv = n_head_kv_;
    rope_.freq_base = cparams.rope_freq_base;
    rope_.freq_scale = cparams.rope_freq_scale;
    type_k_ = params.type_k;
    type_v_ = params.type_v;
    v_trans_ = !cparams.flash_attn;
    method_ = g_kvmem_params.method;
    query_begin_ = g_kvmem_params.query_begin;
    query_end_ = g_kvmem_params.query_end;
    force_pos_ = g_kvmem_params.force_pos;
    kvmem::RawKvStoreConfig rcfg;
    rcfg.n_layer = n_layer_;
    rcfg.n_embd_k = n_embd_k_;
    rcfg.n_embd_v = n_embd_v_;
    rcfg.block_tokens = block_tokens_;
    raw_ = std::make_unique<kvmem::RawKvStore>(rcfg);
    q_sum_.assign(n_layer_, std::vector<float>(n_head_ * n_embd_head_, 0.0f));
    q_count_.assign(n_layer_, 0);
    kvmem_capture_bind(this);

    size_t kv_bytes = 0;
    for (const auto & kv : kv_->memory_breakdown()) {
        kv_bytes += kv.second;
    }
    LLAMA_LOG_INFO(
            "%s: KVMem slot-pool cells=%u slots=%u block_tokens=%u budget=%u gen_reserve=%u sink_blocks=%u method=%s n_embd_k=%u\n",
            __func__, kv_size_, n_slots_, block_tokens_, budget, gen_reserve,
            rt_cfg.store.sink_blocks,
            method_ == 1 ? "retrieval" : "recency",
            n_embd_k_);
    fprintf(stderr,
            "KVMEM_KV_BYTES bytes=%zu cells=%u slots=%u budget=%u pool=%u "
            "ratio=%.2f high=%.2f low=%.2f cap_blocks=%u gpu_total=%llu block_bytes=%llu\n",
            kv_bytes, kv_size_, n_slots_, budget, kv_size_,
            rt_cfg.store.gpu_memory_ratio,
            rt_cfg.store.gpu_high_watermark,
            rt_cfg.store.gpu_low_watermark,
            cap_blocks,
            (unsigned long long) gpu_total,
            (unsigned long long) block_bytes);
}

llama_memory_kvmem::~llama_memory_kvmem() {
    harvest_flush();
    d2h_free();
    kvmem_capture_unbind(this);
}

void llama_memory_kvmem::reset_slots() {
    free_slots_.clear();
    free_slots_.reserve(n_slots_);
    for (int32_t i = static_cast<int32_t>(n_slots_) - 1; i >= 0; --i) {
        free_slots_.push_back(i);
    }
}

int32_t llama_memory_kvmem::alloc_slot() {
    if (free_slots_.empty()) {
        return -1;
    }
    const int32_t slot = free_slots_.back();
    free_slots_.pop_back();
    return slot;
}

void llama_memory_kvmem::free_slot(int32_t slot) {
    if (slot < 0) {
        return;
    }
    free_slots_.push_back(slot);
}

uint32_t llama_memory_kvmem::resident_tokens() const {
    uint32_t n = 0;
    for (const auto & b : runtime_->store().blocks()) {
        if (b.gpu_slot >= 0) {
            n += b.n_tokens;
        }
    }
    return n;
}

void llama_memory_kvmem::trace_plan(const char * tag, const kvmem::KvMemPlan & plan) const {
    if (!trace_) {
        return;
    }
    uint32_t skip = 0;
    for (const auto & r : plan.remaps) {
        if (r.skip) {
            skip++;
        }
    }
    fprintf(stderr,
            "KVMEM_TRACE %s stage_in=%zu stage_out=%zu skip=%u gpu_reused=%u window=%u free_slots=%zu\n",
            tag,
            plan.stage_in.size(),
            plan.stage_out.size(),
            skip,
            plan.gpu_reused_blocks,
            plan.total_window_tokens,
            free_slots_.size());
}

void llama_memory_kvmem::apply_plan_to_kv(const kvmem::KvMemPlan & plan) {
    auto & store = runtime_->store();
    for (uint32_t id : plan.stage_out) {
        harvest_gpu_v(id);
    }
    runtime_->spill_outgoing();
    for (uint32_t id : plan.stage_out) {
        if (id >= store.block_count()) {
            continue;
        }
        const kvmem::KvMemBlock & b = store.blocks()[id];
        if (b.n_tokens > 0) {
            kv_->seq_rm(0, static_cast<llama_pos>(b.orig_pos_start),
                        static_cast<llama_pos>(b.orig_pos_end()));
        }
    }
    runtime_->admit_incoming();
}

bool llama_memory_kvmem::gpu_kv_already_resident(uint32_t block_id) const {
    if (!kv_ || block_id >= runtime_->store().block_count()) {
        return false;
    }
    const kvmem::KvMemBlock & b = runtime_->store().blocks()[block_id];
    if (b.gpu_slot < 0 || b.n_tokens == 0) {
        return false;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx = static_cast<uint32_t>(b.gpu_slot) * block_tokens_;
    if (idx >= cells.size() || cells.is_empty(idx)) {
        return false;
    }
    return cells.pos_get(idx) == static_cast<llama_pos>(b.orig_pos_start);
}

void llama_memory_kvmem::occupy_block_cells(uint32_t block_id) {
    auto & store = runtime_->store();
    if (block_id >= store.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0 || blk.n_tokens == 0) {
        return;
    }
    const uint32_t nt = blk.n_tokens;
    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.resize(1);
    sinfo.strm[0] = 0;
    sinfo.idxs[0].resize(nt);
    std::vector<llama_pos> pos(nt);
    std::vector<int32_t> n_seq_id(nt, 1);
    std::vector<llama_seq_id> seq_data(nt, 0);
    std::vector<llama_seq_id *> seq_id(nt);
    std::vector<int8_t> output(nt, 0);
    std::vector<llama_token> tok(nt, 0);
    llama_seq_id seq0 = 0;
    for (uint32_t t = 0; t < nt; ++t) {
        sinfo.idxs[0][t] = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_ + t;
        pos[t] = static_cast<llama_pos>(blk.orig_pos_start + t);
        seq_id[t] = &seq_data[t];
    }
    llama_ubatch ub{};
    ub.n_tokens = nt;
    ub.n_seq_tokens = nt;
    ub.n_seqs = 1;
    ub.n_seqs_unq = 1;
    ub.n_pos = 1;
    ub.pos = pos.data();
    ub.n_seq_id = n_seq_id.data();
    ub.seq_id = seq_id.data();
    ub.seq_id_unq = &seq0;
    ub.output = output.data();
    ub.token = tok.data();
    {
        const llama_kv_cells & cells = kv_->get_cells(0);
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t idx = sinfo.idxs[0][t];
            if (idx < cells.size() && !cells.is_empty(idx)) {
                const llama_pos p = cells.pos_get(idx);
                kv_->seq_rm(0, p, p + 1);
            }
        }
    }
    kv_->apply_ubatch(sinfo, ub);
}

bool llama_memory_kvmem::layout_gpu_slots_by_orig_pos() {
    auto & store = runtime_->store();
    struct Item {
        uint32_t id;
        uint32_t orig;
        int32_t slot;
        uint32_t n;
        bool resident;
    };
    std::vector<Item> items;
    for (const auto & b : store.blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        items.push_back({b.block_id, b.orig_pos_start, b.gpu_slot, b.n_tokens,
                         gpu_kv_already_resident(b.block_id)});
    }
    if (items.empty()) {
        return false;
    }
    std::sort(items.begin(), items.end(),
              [](const Item & a, const Item & b) { return a.orig < b.orig; });
    bool sorted = true;
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].slot != static_cast<int32_t>(i)) {
            sorted = false;
            break;
        }
    }
    if (sorted) {
        return false;
    }
    const uint64_t payload_bytes =
            static_cast<uint64_t>(n_layer_) *
            (ggml_row_size(type_k_, n_embd_k_) + ggml_row_size(type_v_, n_embd_v_)) *
            block_tokens_;
    std::vector<std::vector<uint8_t>> payloads(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        if (!items[i].resident || payload_bytes == 0) {
            continue;
        }
        payloads[i].assign(static_cast<size_t>(payload_bytes), 0);
        copy_gpu_block_to_host(items[i].id, items[i].slot,
                               payloads[i].data(), payload_bytes);
    }
    for (const auto & it : items) {
        kv_->seq_rm(0, static_cast<llama_pos>(it.orig),
                    static_cast<llama_pos>(it.orig + it.n));
        store.set_block_gpu_slot(it.id, -1);
    }
    reset_slots();
    if (trace_) {
        fprintf(stderr, "KVMEM_TRACE layout_orig_pos");
        for (size_t i = 0; i < items.size(); ++i) {
            fprintf(stderr, " %u", items[i].id);
        }
        fprintf(stderr, "\n");
    }
    uint32_t n_move = 0;
    uint32_t n_raw = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const int32_t slot = alloc_slot();
        if (slot < 0) {
            LLAMA_LOG_ERROR("%s: no GPU slot while laying out block %u\n",
                            __func__, items[i].id);
            return false;
        }
        store.set_block_gpu_slot(items[i].id, slot);
        store.set_block_tier(items[i].id, kvmem::KvTier::GPU, -1,
                             store.blocks()[items[i].id].nvme_slot);
        if (items[i].resident && !payloads[i].empty()) {
            occupy_block_cells(items[i].id);
            copy_gpu_block_from_host(items[i].id, slot,
                                     payloads[i].data(), payload_bytes);
            n_move++;
        } else {
            write_block_to_gpu(items[i].id);
            n_raw++;
        }
    }
    if (trace_) {
        fprintf(stderr, "KVMEM_TRACE layout_writeback move=%u raw=%u\n", n_move, n_raw);
    }
    return true;
}

bool llama_memory_kvmem::prepare_working_set(uint32_t n_new_tokens) {
    if (n_new_tokens == 0) {
        return true;
    }
    if (replay_) {
        return true;
    }
    auto & store = runtime_->store();
    const uint32_t t0 = store.total_tokens();
    if (!replay_) {
        runtime_->register_append(n_new_tokens);
    }
    const uint32_t t1 = store.total_tokens();

    std::vector<uint32_t> incoming;
    for (const auto & b : store.blocks()) {
        if (b.orig_pos_end() > t0 && b.orig_pos_start < t1) {
            incoming.push_back(b.block_id);
        }
    }

    const uint32_t budget_blocks = store.prefill_budget_blocks();
    bool need_offload = false;
    // After retrieval the host store still holds every historical block, so
    // block_count() > budget is true and a recency pressure reselect would
    // drop the resurrected needle on the first generated token. Pin the
    // working set and place decode tokens into gen_reserve slots.
    if (!retrieval_pinned_) {
        try {
            need_offload = runtime_->maybe_offload_during_prefill(
                    n_new_tokens, resident_tokens(), kv_size_, incoming);
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: KVMem reselect failed: %s\n", __func__, e.what());
            runtime_->truncate_to(t0);
            return false;
        }
    }

    if (trace_) {
        fprintf(stderr,
                "KVMEM_TRACE append n=%u total=%u resident=%u incoming_blocks=%zu "
                "over_budget=%d need_offload=%d free_slots=%zu\n",
                n_new_tokens, t1, resident_tokens(), incoming.size(),
                (int) (store.block_count() > budget_blocks), (int) need_offload,
                free_slots_.size());
    }

    if (need_offload) {
        try {
            const kvmem::KvMemPlan & plan = runtime_->last_plan();
            trace_plan("prefill_pressure", plan);
            apply_plan_to_kv(plan);
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: KVMem reselect failed: %s\n", __func__, e.what());
            runtime_->truncate_to(t0);
            return false;
        }
    } else {
        for (uint32_t id : incoming) {
            if (store.blocks()[id].gpu_slot >= 0) {
                continue;
            }
            const int32_t slot = alloc_slot();
            if (slot < 0) {
                // Pinned retrieval will not evict the working set. gen_reserve
                // exhaustion is a known v1 limit (see docs/architecture.md).
                LLAMA_LOG_ERROR("%s: no free GPU slot for block %u\n", __func__, id);
                runtime_->truncate_to(t0);
                return false;
            }
            store.set_block_gpu_slot(id, slot);
            store.set_block_tier(id, kvmem::KvTier::GPU, -1, store.blocks()[id].nvme_slot);
        }
    }

    for (uint32_t id : incoming) {
        if (id < store.block_count() && store.blocks()[id].gpu_slot < 0) {
            LLAMA_LOG_ERROR("%s: incoming block %u was not placed on GPU\n", __func__, id);
            runtime_->truncate_to(t0);
            return false;
        }
    }
    return true;
}

llama_memory_context_ptr llama_memory_kvmem::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        const uint32_t n_new = balloc.get_n_tokens();
        if (!prepare_working_set(n_new)) {
            break;
        }

        llama_kv_cache::slot_info_vec_t sinfos;
        sinfos.reserve(ubatches.size());
        bool ok = true;
        for (const auto & ubatch : ubatches) {
            llama_kv_cache::slot_info sinfo;
            if (!kvmem_fill_slot_info(runtime_->store(), block_tokens_, kv_size_, ubatch, sinfo)) {
                ok = false;
                break;
            }
            sinfos.push_back(std::move(sinfo));
        }
        if (!ok) {
            break;
        }

        pos_queue_.clear();
        for (const auto & ubatch : ubatches) {
            std::vector<llama_pos> p(ubatch.n_tokens);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                p[i] = ubatch.pos ? ubatch.pos[i] : static_cast<llama_pos>(i);
            }
            pos_queue_.push_back(std::move(p));
        }

        return std::make_unique<llama_kv_cache_context>(
                kv_.get(), std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_kvmem::init_full() {
    return kv_->init_full();
}

llama_memory_context_ptr llama_memory_kvmem::init_update(llama_context * lctx, bool optimize) {
    return kv_->init_update(lctx, optimize);
}

void llama_memory_kvmem::clear(bool data) {
    kv_->clear(data);
    runtime_->truncate_to(0);
    reset_slots();
    retrieval_pinned_ = false;
}

bool llama_memory_kvmem::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const bool ok = kv_->seq_rm(seq_id, p0, p1);
    if (!ok) {
        return false;
    }
    // Query replay seq_rm's the query span, which can be the whole prompt on
    // short inputs. That must not wipe KvMemStore metadata / GPU slots.
    if (!replay_ &&
        seq_id <= 0 && p0 <= 0 &&
        (p1 < 0 || p1 >= static_cast<llama_pos>(runtime_->store().total_tokens()))) {
        runtime_->truncate_to(0);
        reset_slots();
        retrieval_pinned_ = false;
    }
    return true;
}

void llama_memory_kvmem::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_kvmem::seq_keep(llama_seq_id seq_id) {
    kv_->seq_keep(seq_id);
}

void llama_memory_kvmem::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_kvmem::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_kvmem::seq_pos_min(llama_seq_id seq_id) const {
    return kv_->seq_pos_min(seq_id);
}

llama_pos llama_memory_kvmem::seq_pos_max(llama_seq_id seq_id) const {
    return kv_->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_kvmem::memory_breakdown() const {
    return kv_->memory_breakdown();
}

void llama_memory_kvmem::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    kv_->state_write(io, seq_id, flags);
}

void llama_memory_kvmem::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    kv_->state_read(io, seq_id, flags);
}

void llama_memory_kvmem::note_ubatch_pos(const std::vector<llama_pos> & pos) {
    pos_queue_.push_back(pos);
}

void llama_memory_kvmem::register_capture(ggml_tensor * t, int il, char which) {
    if (!t) {
        return;
    }
    pending_capture_.push_back({t, il, which});
    if (which == 'q') {
        graph_has_q_ = true;
    }
}

void llama_memory_kvmem::capture_on_new_graph() {
    pending_capture_.clear();
    graph_has_q_ = false;
}

bool llama_memory_kvmem::capture_can_reuse(uint32_t n_tokens, uint32_t n_pos,
                                           const llama_pos * pos) const {
    return llama_kvmem_ubatch_needs_q_capture(n_tokens, n_pos, pos) == graph_has_q_;
}

bool llama_memory_kvmem::d2h_init() {
    if (d2h_ && d2h_->ok) {
        return true;
    }
    if (!d2h_) {
        d2h_ = std::make_unique<CaptureD2hPipe>();
    }
    if (d2h_->stream == nullptr) {
        if (!kvmem_cuda_ok(cudaStreamCreateWithFlags(&d2h_->stream, cudaStreamNonBlocking),
                           "stream")) {
            return false;
        }
    }
    if (d2h_->snap == nullptr) {
        if (!kvmem_cuda_ok(cudaEventCreateWithFlags(&d2h_->snap, cudaEventDisableTiming),
                           "snap")) {
            return false;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (d2h_->slots[i].done == nullptr) {
            if (!kvmem_cuda_ok(cudaEventCreateWithFlags(&d2h_->slots[i].done, cudaEventDisableTiming),
                               "event")) {
                return false;
            }
        }
    }
    d2h_->ok = true;
    return true;
}

void llama_memory_kvmem::d2h_free() {
    if (!d2h_) {
        return;
    }
    if (d2h_->stream) {
        cudaStreamSynchronize(d2h_->stream);
    }
    for (int i = 0; i < 2; ++i) {
        auto & s = d2h_->slots[i];
        if (s.gpu) {
            cudaFree(s.gpu);
            s.gpu = nullptr;
        }
        if (s.pin) {
            cudaFreeHost(s.pin);
            s.pin = nullptr;
        }
        if (s.done) {
            cudaEventDestroy(s.done);
            s.done = nullptr;
        }
        s.cap = 0;
        s.inflight = false;
    }
    if (d2h_->snap) {
        cudaEventDestroy(d2h_->snap);
        d2h_->snap = nullptr;
    }
    if (d2h_->stream) {
        cudaStreamDestroy(d2h_->stream);
        d2h_->stream = nullptr;
    }
    d2h_->ok = false;
}

void llama_memory_kvmem::d2h_commit(int slot) {
    if (!d2h_ || slot < 0 || slot > 1) {
        return;
    }
    auto & s = d2h_->slots[slot];
    if (!s.inflight) {
        return;
    }
    if (s.done) {
        cudaEventSynchronize(s.done);
    }
    cur_pos_ = s.pos;
    for (const auto & it : s.items) {
        harvest_from_host(it.il, it.which, s.pin + it.offset, it.type,
                          it.d, it.h, it.n, it.nb0, it.nb1, it.nb2);
    }
    s.inflight = false;
    s.items.clear();
    s.pos.clear();
}

void llama_memory_kvmem::harvest_flush() {
    if (!d2h_) {
        return;
    }
    d2h_commit(1 - d2h_->next);
    d2h_commit(d2h_->next);
}

bool llama_memory_kvmem::d2h_submit(ggml_backend_t be) {
    if (!d2h_init() || pending_capture_.empty() || pos_queue_.empty()) {
        return false;
    }
    size_t bytes = 0;
    for (const CaptureNode & n : pending_capture_) {
        if (n.t) {
            bytes += ggml_nbytes(n.t);
        }
    }
    if (bytes == 0) {
        return false;
    }
    auto & s = d2h_->slots[d2h_->next];
    if (s.inflight) {
        d2h_commit(d2h_->next);
    }
    if (s.cap < bytes) {
        if (s.gpu) {
            cudaFree(s.gpu);
            s.gpu = nullptr;
        }
        if (s.pin) {
            cudaFreeHost(s.pin);
            s.pin = nullptr;
        }
        if (!kvmem_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&s.gpu), bytes), "gpu staging") ||
            !kvmem_cuda_ok(cudaMallocHost(reinterpret_cast<void **>(&s.pin), bytes), "pinned host")) {
            s.cap = 0;
            return false;
        }
        s.cap = bytes;
    }
    if (be) {
        ggml_backend_synchronize(be);
    } else {
        cudaDeviceSynchronize();
    }
    s.items.clear();
    s.pos = std::move(pos_queue_.front());
    pos_queue_.erase(pos_queue_.begin());
    size_t off = 0;
    uint32_t n_q = 0, n_k = 0, n_v = 0, n_host = 0, n_dev = 0;
    for (const CaptureNode & n : pending_capture_) {
        if (!n.t) {
            continue;
        }
        CaptureD2hPipe::Item it;
        it.il = n.il;
        it.which = n.which;
        it.offset = off;
        it.nbytes = ggml_nbytes(n.t);
        it.d = n.t->ne[0];
        it.h = n.t->ne[1];
        it.n = n.t->ne[2];
        it.nb0 = n.t->nb[0];
        it.nb1 = n.t->nb[1];
        it.nb2 = n.t->nb[2];
        it.type = n.t->type;
        ggml_backend_buffer_t buf = n.t->view_src ? n.t->view_src->buffer : n.t->buffer;
        const uint8_t * src = static_cast<const uint8_t *>(n.t->data);
        if (buf && ggml_backend_buffer_is_host(buf)) {
            memcpy(s.pin + off, src, it.nbytes);
            n_host++;
        } else {
            if (!kvmem_cuda_ok(cudaMemcpyAsync(s.gpu + off, src, it.nbytes,
                                               cudaMemcpyDeviceToDevice, d2h_->stream),
                               "D2D")) {
                pos_queue_.insert(pos_queue_.begin(), std::move(s.pos));
                return false;
            }
            it.from_gpu = true;
            n_dev++;
        }
        if (n.which == 'q') {
            n_q++;
        } else if (n.which == 'k') {
            n_k++;
        } else if (n.which == 'v') {
            n_v++;
        }
        s.items.push_back(it);
        off += it.nbytes;
    }
    if (n_dev > 0) {
        if (!kvmem_cuda_ok(cudaEventRecord(d2h_->snap, d2h_->stream), "snap record") ||
            !kvmem_cuda_ok(cudaEventSynchronize(d2h_->snap), "snap wait")) {
            return false;
        }
        if (n_host == 0) {
            if (!kvmem_cuda_ok(cudaMemcpyAsync(s.pin, s.gpu, off, cudaMemcpyDeviceToHost, d2h_->stream),
                               "D2H")) {
                return false;
            }
        } else {
            for (const auto & it : s.items) {
                if (!it.from_gpu) {
                    continue;
                }
                if (!kvmem_cuda_ok(cudaMemcpyAsync(s.pin + it.offset, s.gpu + it.offset, it.nbytes,
                                                   cudaMemcpyDeviceToHost, d2h_->stream),
                                   "D2H item")) {
                    return false;
                }
            }
        }
    }
    if (!kvmem_cuda_ok(cudaEventRecord(s.done, d2h_->stream), "record")) {
        return false;
    }
    s.inflight = true;
    if (trace_) {
        fprintf(stderr,
                "KVMEM_TRACE harvest n=%zu q=%u k=%u v=%u host=%u gpu=%u bytes=%zu n_pos=%zu async=1 slot=%d\n",
                s.items.size(), n_q, n_k, n_v, n_host, n_dev, off, s.pos.size(), d2h_->next);
    }
    d2h_->next = 1 - d2h_->next;
    return true;
}

void llama_memory_kvmem::harvest_pending(ggml_backend_sched_t sched) {
    if (d2h_ && d2h_->ok) {
        d2h_commit(d2h_->next == 0 ? 1 : 0);
    }
    if (pending_capture_.empty() || pos_queue_.empty()) {
        harvest_flush();
        return;
    }
    ggml_backend_t be = nullptr;
    if (sched) {
        for (const CaptureNode & n : pending_capture_) {
            if (n.t) {
                be = ggml_backend_sched_get_tensor_backend(sched, n.t);
                if (be) {
                    break;
                }
            }
        }
    }
    if (d2h_submit(be)) {
        return;
    }
    cur_pos_ = std::move(pos_queue_.front());
    pos_queue_.erase(pos_queue_.begin());
    for (const CaptureNode & n : pending_capture_) {
        harvest_capture(n.t, n.il, n.which);
    }
}

void llama_memory_kvmem::reset_query_acc() {
    for (auto & s : q_sum_) {
        std::fill(s.begin(), s.end(), 0.0f);
    }
    std::fill(q_count_.begin(), q_count_.end(), 0);
}

void llama_memory_kvmem::bytes_to_f32_token_major(const uint8_t * data, ggml_type type,
                                                  int64_t d, int64_t h, int64_t n,
                                                  size_t nb0, size_t nb1, size_t nb2,
                                                  std::vector<float> & out) {
    out.assign(static_cast<size_t>(n * h * d), 0.0f);
    if (!data) {
        return;
    }
    for (int64_t tok = 0; tok < n; ++tok) {
        for (int64_t head = 0; head < h; ++head) {
            for (int64_t dim = 0; dim < d; ++dim) {
                const size_t off = static_cast<size_t>(tok * nb2 + head * nb1 + dim * nb0);
                float val = 0.0f;
                if (type == GGML_TYPE_F32) {
                    val = *reinterpret_cast<const float *>(data + off);
                } else if (type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(data + off));
                }
                out[static_cast<size_t>(tok * h * d + head * d + dim)] = val;
            }
        }
    }
}

void llama_memory_kvmem::tensor_to_f32_token_major(const ggml_tensor * t, std::vector<float> & out) {
    std::vector<uint8_t> tmp;
    const uint8_t * data = nullptr;
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        data = static_cast<const uint8_t *>(t->data);
    } else {
        tmp.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
        data = tmp.data();
    }
    bytes_to_f32_token_major(data, t->type, t->ne[0], t->ne[1], t->ne[2],
                             t->nb[0], t->nb[1], t->nb[2], out);
}

void llama_memory_kvmem::harvest_from_host(int il, char which, const uint8_t * host,
                                           ggml_type type, int64_t d, int64_t h, int64_t ntok,
                                           size_t nb0, size_t nb1, size_t nb2) {
    if (!host || il < 0 || static_cast<uint32_t>(il) >= n_layer_ || cur_pos_.empty()) {
        return;
    }
    std::vector<float> flat;
    bytes_to_f32_token_major(host, type, d, h, ntok, nb0, nb1, nb2, flat);
    const uint32_t n = static_cast<uint32_t>(cur_pos_.size());
    const uint32_t pos0 = static_cast<uint32_t>(cur_pos_[0]);
    if (which == 'k' || which == 'v') {
        const float * kptr = which == 'k' ? flat.data() : nullptr;
        const float * vptr = which == 'v' ? flat.data() : nullptr;
        raw_->write_layer_tokens(pos0, n, static_cast<uint32_t>(il), kptr, vptr);
    } else if (which == 'q') {
        const uint32_t qdim = n_head_ * n_embd_head_;
        if (flat.size() < static_cast<size_t>(n) * qdim) {
            return;
        }
        int32_t qb = query_begin_;
        int32_t qe = query_end_;
        if (qe <= 0) {
            qe = 1 << 30;
        }
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t pos = cur_pos_[i];
            if (qb >= 0 && (pos < qb || pos >= qe)) {
                continue;
            }
            if (qb < 0 && method_ != 1) {
                continue;
            }
            float * dst = q_sum_[static_cast<uint32_t>(il)].data();
            const float * src = flat.data() + i * qdim;
            for (uint32_t d0 = 0; d0 < qdim; ++d0) {
                dst[d0] += src[d0];
            }
            q_count_[static_cast<uint32_t>(il)]++;
        }
    }
}

void llama_memory_kvmem::harvest_capture(ggml_tensor * t, int il, char which) {
    if (!t || il < 0 || static_cast<uint32_t>(il) >= n_layer_) {
        return;
    }
    if (cur_pos_.empty()) {
        return;
    }
    std::vector<float> flat;
    tensor_to_f32_token_major(t, flat);
    const uint32_t n = static_cast<uint32_t>(cur_pos_.size());
    if (n == 0) {
        return;
    }
    const uint32_t pos0 = static_cast<uint32_t>(cur_pos_[0]);

    if (which == 'k' || which == 'v') {
        const float * kptr = which == 'k' ? flat.data() : nullptr;
        const float * vptr = which == 'v' ? flat.data() : nullptr;
        raw_->write_layer_tokens(pos0, n, static_cast<uint32_t>(il), kptr, vptr);
        if (which == 'k') {
            static bool dumped = false;
            if (!dumped && getenv("KVMEM_DUMP_CAPTURE") && il == 0) {
                dumped = true;
                double acc = 0;
                for (float x : flat) {
                    acc += static_cast<double>(x) * x;
                }
                fprintf(stderr, "KVMEM_DUMP k_prerope layer0 n=%u rms=%.6f shape_elems=%zu\n",
                        n, std::sqrt(acc / std::max<size_t>(flat.size(), 1)), flat.size());
            }
        }
    } else if (which == 'q') {
        const uint32_t qdim = n_head_ * n_embd_head_;
        if (flat.size() < static_cast<size_t>(n) * qdim) {
            return;
        }
        int32_t qb = query_begin_;
        int32_t qe = query_end_;
        if (qe <= 0) {
            qe = 1 << 30;
        }
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t pos = cur_pos_[i];
            if (qb >= 0 && (pos < qb || pos >= qe)) {
                continue;
            }
            if (qb < 0 && method_ != 1) {
                continue;
            }
            float * dst = q_sum_[static_cast<uint32_t>(il)].data();
            const float * src = flat.data() + i * qdim;
            for (uint32_t d = 0; d < qdim; ++d) {
                dst[d] += src[d];
            }
            q_count_[static_cast<uint32_t>(il)]++;
        }
    }
}

void llama_memory_kvmem::harvest_gpu_v(uint32_t block_id) {
    if (v_trans_ || !raw_ || !kv_) {
        return;
    }
    auto & store = runtime_->store();
    if (block_id >= store.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0 || blk.n_tokens == 0) {
        return;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_;
    if (idx >= cells.size() || cells.is_empty(idx)) {
        return;
    }
    std::vector<float> v;
    uint32_t n_ok = 0;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!read_gpu_block(block_id, il, false, v)) {
            continue;
        }
        raw_->write_layer_tokens(blk.orig_pos_start, blk.n_tokens, il, nullptr, v.data());
        n_ok++;
    }
    if (trace_) {
        fprintf(stderr, "KVMEM_TRACE harvest_v block=%u slot=%d n=%u layers=%u\n",
                block_id, blk.gpu_slot, blk.n_tokens, n_ok);
    }
}

void llama_memory_kvmem::write_block_to_gpu(uint32_t block_id) {
    auto & store = runtime_->store();
    if (block_id >= store.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0 || !raw_->has_block(block_id)) {
        return;
    }
    occupy_block_cells(block_id);
    const uint32_t nt = blk.n_tokens;
    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.resize(1);
    sinfo.strm[0] = 0;
    sinfo.idxs[0].resize(nt);
    for (uint32_t t = 0; t < nt; ++t) {
        sinfo.idxs[0][t] = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_ + t;
    }

    std::vector<float> roped(static_cast<size_t>(nt) * n_embd_k_);
    std::vector<ggml_fp16_t> k16(n_embd_k_);
    std::vector<ggml_fp16_t> v16(n_embd_v_);
    for (uint32_t il = 0; il < n_layer_; ++il) {
        const float * rk = raw_->k(block_id, il);
        const float * rv = raw_->v(block_id, il);
        if (!rk) {
            continue;
        }
        kvmem::rope_neox_apply(rope_, rk, nt, static_cast<int32_t>(blk.orig_pos_start), roped.data());
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        const size_t krow = ggml_row_size(type_k_, n_embd_k_);
        const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t cell = sinfo.idxs[0][t];
            const float * src = roped.data() + t * n_embd_k_;
            if (type_k_ == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(src, k16.data(), static_cast<int64_t>(n_embd_k_));
                ggml_backend_tensor_set(kt, k16.data(), cell * krow, krow);
            } else {
                ggml_backend_tensor_set(kt, src, cell * krow, krow);
            }
            if (rv && vt && !v_trans_) {
                const float * vs = rv + t * n_embd_v_;
                if (type_v_ == GGML_TYPE_F16) {
                    ggml_fp32_to_fp16_row(vs, v16.data(), static_cast<int64_t>(n_embd_v_));
                    ggml_backend_tensor_set(vt, v16.data(), cell * vrow, vrow);
                } else {
                    ggml_backend_tensor_set(vt, vs, cell * vrow, vrow);
                }
            }
        }
    }
    if (trace_) {
        fprintf(stderr, "KVMEM_TRACE stage_in_raw block=%u slot=%d n=%u orig=%u\n",
                block_id, blk.gpu_slot, nt, blk.orig_pos_start);
    }
}

void llama_memory_kvmem::copy_gpu_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                                void * host, uint64_t bytes) {
    if (!host || !kv_ || gpu_slot < 0 || bytes == 0) {
        return;
    }
    (void) block_id;
    auto * dst = static_cast<uint8_t *>(host);
    uint64_t off = 0;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    for (uint32_t il = 0; il < n_layer_; ++il) {
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        for (uint32_t t = 0; t < block_tokens_; ++t) {
            const uint32_t cell = static_cast<uint32_t>(gpu_slot) * block_tokens_ + t;
            if (off + krow > bytes) {
                return;
            }
            if (kt && cell < kv_size_) {
                ggml_backend_tensor_get(kt, dst + off, cell * krow, krow);
            }
            off += krow;
        }
        for (uint32_t t = 0; t < block_tokens_; ++t) {
            if (off + vrow > bytes) {
                return;
            }
            const uint32_t cell = static_cast<uint32_t>(gpu_slot) * block_tokens_ + t;
            if (vt && !v_trans_ && cell < kv_size_) {
                ggml_backend_tensor_get(vt, dst + off, cell * vrow, vrow);
            }
            off += vrow;
        }
    }
}

void llama_memory_kvmem::copy_gpu_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                                  const void * host, uint64_t bytes) {
    if (!host || !kv_ || gpu_slot < 0 || bytes == 0) {
        return;
    }
    (void) block_id;
    const auto * src = static_cast<const uint8_t *>(host);
    uint64_t off = 0;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    for (uint32_t il = 0; il < n_layer_; ++il) {
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        for (uint32_t t = 0; t < block_tokens_; ++t) {
            const uint32_t cell = static_cast<uint32_t>(gpu_slot) * block_tokens_ + t;
            if (off + krow > bytes) {
                return;
            }
            if (kt && cell < kv_size_) {
                ggml_backend_tensor_set(kt, src + off, cell * krow, krow);
            }
            off += krow;
        }
        for (uint32_t t = 0; t < block_tokens_; ++t) {
            if (off + vrow > bytes) {
                return;
            }
            const uint32_t cell = static_cast<uint32_t>(gpu_slot) * block_tokens_ + t;
            if (vt && !v_trans_ && cell < kv_size_) {
                ggml_backend_tensor_set(vt, src + off, cell * vrow, vrow);
            }
            off += vrow;
        }
    }
}

void llama_memory_kvmem::score_retrieval() {
    auto & store = runtime_->store();
    const uint32_t nblk = store.block_count();
    std::vector<double> scores(nblk, 0.0);
    std::vector<float> mk(n_embd_k_, 0.0f);
    const uint32_t g = n_head_kv_ == 0 ? 1 : n_head_ / n_head_kv_;
    for (uint32_t b = 0; b < nblk; ++b) {
        if (!raw_->has_block(b)) {
            continue;
        }
        double acc = 0;
        uint32_t nlay = 0;
        for (uint32_t il = 0; il < n_layer_; ++il) {
            if (q_count_[il] == 0) {
                continue;
            }
            raw_->mean_k(b, il, mk.data());
            double layer = 0;
            const float invq = 1.0f / static_cast<float>(q_count_[il] * std::max(1u, g));
            for (uint32_t h = 0; h < n_head_kv_; ++h) {
                std::vector<float> qh(n_embd_head_, 0.0f);
                for (uint32_t gi = 0; gi < g; ++gi) {
                    const uint32_t qh_i = h * g + gi;
                    if (qh_i >= n_head_) {
                        break;
                    }
                    const float * q = q_sum_[il].data() + qh_i * n_embd_head_;
                    for (uint32_t d = 0; d < n_embd_head_; ++d) {
                        qh[d] += q[d];
                    }
                }
                const float * kh = mk.data() + h * n_embd_head_;
                float dot = 0, nq = 0, nk = 0;
                for (uint32_t d = 0; d < n_embd_head_; ++d) {
                    const float qv = qh[d] * invq;
                    dot += qv * kh[d];
                    nq += qv * qv;
                    nk += kh[d] * kh[d];
                }
                layer += dot / (std::sqrt(nq) * std::sqrt(nk) + 1e-6f);
            }
            acc += layer / static_cast<double>(std::max(1u, n_head_kv_));
            nlay++;
        }
        scores[b] = nlay ? acc / nlay : 0;
    }
    store.set_retrieval_scores(scores);
}

void llama_memory_kvmem::apply_retrieval() {
    harvest_flush();
    if (method_ != 1) {
        return;
    }
    score_retrieval();
    std::vector<uint32_t> mandatory;
    if (query_begin_ >= 0) {
        const uint32_t qe = query_end_ > 0 ? static_cast<uint32_t>(query_end_)
                                           : runtime_->store().total_tokens();
        for (const auto & b : runtime_->store().blocks()) {
            if (b.orig_pos_end() > static_cast<uint32_t>(query_begin_) && b.orig_pos_start < qe) {
                mandatory.push_back(b.block_id);
            }
        }
    }
    if (force_pos_ >= 0) {
        const int32_t bid = runtime_->store().block_id_containing(static_cast<uint32_t>(force_pos_));
        if (bid >= 0) {
            mandatory.push_back(static_cast<uint32_t>(bid));
            if (trace_) {
                fprintf(stderr, "KVMEM_TRACE force_pos=%d needle_block=%d\n",
                        force_pos_, bid);
            }
        }
    }
    const kvmem::KvMemPlan plan = runtime_->prepare_reselect(mandatory);
    trace_plan("retrieval", plan);
    if (trace_) {
        fprintf(stderr, "KVMEM_TRACE selected");
        for (const auto & r : plan.remaps) {
            fprintf(stderr, " %u", r.block_id);
        }
        fprintf(stderr, "\n");
    }
    apply_plan_to_kv(plan);
    auto & store = runtime_->store();
    retrieval_pinned_ = true;
    // Native GPU KV of a reused block at orig_pos != 0 is the ground truth for
    // host RoPE + capture layout. Prefer a block that is NOT in stage_in so the
    // slot still holds prefill bytes. Must run before layout rewrite.
    if (trace_) {
        std::vector<bool> is_stage_in(store.block_count(), false);
        for (uint32_t id : plan.stage_in) {
            if (id < is_stage_in.size()) {
                is_stage_in[id] = true;
            }
        }
        for (const auto & r : plan.remaps) {
            if (r.block_id >= store.block_count() || is_stage_in[r.block_id]) {
                continue;
            }
            const kvmem::KvMemBlock & b = store.blocks()[r.block_id];
            if (b.gpu_slot < 0 || b.orig_pos_start == 0 || !raw_->has_block(r.block_id)) {
                continue;
            }
            fprintf(stderr,
                    "KVMEM_KV --- native reused block %u orig=%u skip=%d (GPU vs host rope) ---\n",
                    r.block_id, b.orig_pos_start, (int) r.skip);
            dump_kv_compare(static_cast<int32_t>(r.block_id), false);
            break;
        }
    }
    const bool laid_out = layout_gpu_slots_by_orig_pos();
    uint32_t n_raw = 0;
    uint32_t n_skip = 0;
    if (!laid_out) {
        for (uint32_t id : plan.stage_in) {
            if (id >= store.block_count() || store.blocks()[id].gpu_slot < 0) {
                continue;
            }
            if (gpu_kv_already_resident(id)) {
                n_skip++;
                continue;
            }
            write_block_to_gpu(id);
            n_raw++;
        }
    }
    if (trace_) {
        fprintf(stderr, "KVMEM_TRACE writeback laid_out=%d raw=%u skip_resident=%u stage_in=%zu\n",
                (int) laid_out, n_raw, n_skip, plan.stage_in.size());
    }
    if (trace_) {
        for (uint32_t id : plan.stage_in) {
            if (id < store.block_count() && store.blocks()[id].gpu_slot >= 0) {
                fprintf(stderr, "KVMEM_KV --- after stage_in_raw block %u ---\n", id);
                dump_kv_compare(static_cast<int32_t>(id), false);
                break;
            }
        }
        trace_working_set("after_retrieval");
    }
}

void llama_memory_kvmem::trace_working_set(const char * tag) const {
    if (!kv_ || !runtime_) {
        return;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    fprintf(stderr,
            "KVMEM_TRACE %s cells used=%u used_max_p1=%u size=%u seq_pos=[%d,%d]\n",
            tag,
            cells.get_used(), cells.used_max_p1(), cells.size(),
            kv_->seq_pos_min(0), kv_->seq_pos_max(0));
    const auto & store = runtime_->store();
    for (const auto & b : store.blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        const uint32_t idx = static_cast<uint32_t>(b.gpu_slot) * block_tokens_;
        const bool empty = idx >= cells.size() || cells.is_empty(idx);
        llama_pos p_last = -1;
        bool last_empty = true;
        if (!empty && b.n_tokens > 0) {
            const uint32_t last = idx + b.n_tokens - 1;
            last_empty = last >= cells.size() || cells.is_empty(last);
            p_last = last_empty ? -1 : cells.pos_get(last);
        }
        fprintf(stderr,
                "KVMEM_TRACE occupy tag=%s block=%u slot=%d cell=%u empty=%d pos=%d last_empty=%d last_pos=%d n=%u orig=%u\n",
                tag, b.block_id, b.gpu_slot, idx, (int) empty,
                empty ? -1 : (int) cells.pos_get(idx),
                (int) last_empty, (int) p_last, b.n_tokens, b.orig_pos_start);
    }
}

void llama_memory_kvmem::kv_stats(const char * tag, const float * a, const float * b, size_t n) {
    if (!a || !b || n == 0) {
        fprintf(stderr, "KVMEM_KV %s missing n=%zu\n", tag, n);
        return;
    }
    double dot = 0, na = 0, nb = 0, se = 0, maxabs = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i];
        const double db = b[i];
        dot += da * db;
        na += da * da;
        nb += db * db;
        const double e = da - db;
        se += e * e;
        maxabs = std::max(maxabs, std::abs(e));
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    const double rmse = std::sqrt(se / static_cast<double>(n));
    const double rms_a = std::sqrt(na / static_cast<double>(n));
    fprintf(stderr, "KVMEM_KV %s n=%zu cos=%.6f rmse=%.6f maxabs=%.6f rms_gpu=%.6f\n",
            tag, n, cos, rmse, maxabs, rms_a);
}

bool llama_memory_kvmem::read_gpu_block(uint32_t block_id, uint32_t il, bool is_k,
                                        std::vector<float> & out) const {
    const auto & store = runtime_->store();
    if (block_id >= store.block_count() || il >= n_layer_) {
        return false;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0) {
        return false;
    }
    ggml_tensor * t = is_k ? kv_->get_k_storage(static_cast<int32_t>(il))
                           : kv_->get_v_storage(static_cast<int32_t>(il));
    if (!t) {
        return false;
    }
    const uint32_t dim = is_k ? n_embd_k_ : n_embd_v_;
    const ggml_type ty = is_k ? type_k_ : type_v_;
    const size_t row = ggml_row_size(ty, dim);
    const uint32_t nt = blk.n_tokens;
    out.assign(static_cast<size_t>(nt) * dim, 0.0f);
    std::vector<uint8_t> raw(row);
    for (uint32_t t_i = 0; t_i < nt; ++t_i) {
        const uint32_t cell = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_ + t_i;
        ggml_backend_tensor_get(t, raw.data(), static_cast<size_t>(cell) * row, row);
        float * dst = out.data() + t_i * dim;
        if (ty == GGML_TYPE_F16) {
            const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(raw.data());
            for (uint32_t d = 0; d < dim; ++d) {
                dst[d] = ggml_fp16_to_fp32(src[d]);
            }
        } else if (ty == GGML_TYPE_F32) {
            std::memcpy(dst, raw.data(), row);
        }
    }
    return true;
}

void llama_memory_kvmem::dump_kv_compare(int32_t block_id, bool writeback_test) {
    auto & store = runtime_->store();
    if (block_id < 0 && force_pos_ >= 0) {
        block_id = store.block_id_containing(static_cast<uint32_t>(force_pos_));
    }
    if (block_id < 0) {
        block_id = 0;
    }
    const uint32_t bid = static_cast<uint32_t>(block_id);
    if (bid >= store.block_count()) {
        fprintf(stderr, "KVMEM_KV dump: no block %d\n", block_id);
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[bid];
    fprintf(stderr,
            "KVMEM_KV dump block=%u slot=%d n=%u orig=%u raw=%d v_trans=%d type_k=%s rope n_rot=%u freq_base=%.1f scale=%g pos0=%d\n",
            bid, blk.gpu_slot, blk.n_tokens, blk.orig_pos_start,
            (int) raw_->has_block(bid), (int) v_trans_, ggml_type_name(type_k_),
            rope_.n_rot, rope_.freq_base, rope_.freq_scale,
            static_cast<int>(blk.orig_pos_start));

    const uint32_t nt = blk.n_tokens;
    const size_t nk = static_cast<size_t>(nt) * n_embd_k_;
    const size_t nv = static_cast<size_t>(nt) * n_embd_v_;
    std::vector<float> gpu_k, gpu_v, rec_k(nk);

    const uint32_t layers_show[] = {0, n_layer_ / 2, n_layer_ > 0 ? n_layer_ - 1 : 0};
    for (uint32_t li = 0; li < 3; ++li) {
        const uint32_t il = layers_show[li];
        if (il >= n_layer_) {
            continue;
        }
        if (!read_gpu_block(bid, il, true, gpu_k)) {
            fprintf(stderr, "KVMEM_KV L%u K gpu read fail\n", il);
            continue;
        }
        const float * rk = raw_->k(bid, il);
        const float * rv = raw_->v(bid, il);
        if (!rk) {
            fprintf(stderr, "KVMEM_KV L%u no raw K\n", il);
            continue;
        }
        kvmem::rope_neox_apply(rope_, rk, nt, static_cast<int32_t>(blk.orig_pos_start), rec_k.data());

        char tag[64];
        snprintf(tag, sizeof(tag), "L%u K rebuild_vs_gpu", il);
        kv_stats(tag, gpu_k.data(), rec_k.data(), nk);

        snprintf(tag, sizeof(tag), "L%u K raw_vs_gpu (no rope; ~1 only at pos0)", il);
        kv_stats(tag, gpu_k.data(), rk, n_embd_k_);

        if (rv && read_gpu_block(bid, il, false, gpu_v) && !v_trans_) {
            snprintf(tag, sizeof(tag), "L%u V raw_vs_gpu (no rope)", il);
            kv_stats(tag, gpu_v.data(), rv, nv);
        }
    }

    if (writeback_test && read_gpu_block(bid, 0, true, gpu_k) && raw_->k(bid, 0)) {
        std::vector<float> before = gpu_k;
        kvmem::rope_neox_apply(rope_, raw_->k(bid, 0), nt,
                               static_cast<int32_t>(blk.orig_pos_start), rec_k.data());
        write_block_to_gpu(bid);
        std::vector<float> after;
        if (read_gpu_block(bid, 0, true, after)) {
            kv_stats("L0 K gpu_after_writeback_vs_gpu_before", before.data(), after.data(), before.size());
            kv_stats("L0 K gpu_after_writeback_vs_rebuild", rec_k.data(), after.data(), after.size());
        }
    }
}

void llama_kvmem_dump_kv_compare(struct llama_context * /*ctx*/, int32_t block_id) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->dump_kv_compare(block_id, false);
    }
}

void llama_kvmem_dump_kv_writeback(struct llama_context * /*ctx*/, int32_t block_id) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->dump_kv_compare(block_id, true);
    }
}

void llama_kvmem_apply_retrieval(struct llama_context * /*ctx*/) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->apply_retrieval();
    }
}

void llama_kvmem_set_replay(bool replay) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->set_replay(replay);
    }
}

void llama_kvmem_trace_cells(struct llama_context * /*ctx*/, const char * tag) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->trace_working_set(tag ? tag : "cells");
    }
}
