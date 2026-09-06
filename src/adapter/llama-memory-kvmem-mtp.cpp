#include "llama-memory-kvmem-mtp.h"

#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-kvmem-capture.h"
#include "llama-kvmem-hooks.h"
#include "llama-kvmem-quant.h"
#include "llama-kvmem-stagein.h"
#include "llama-model.h"

#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint8_t * kvmem_mtp_cuda_ptr(ggml_tensor * t) {
    if (!t || !t->data) {
        return nullptr;
    }
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buf || ggml_backend_buffer_is_host(buf)) {
        return nullptr;
    }
    return static_cast<uint8_t *>(t->data);
}

llama_memory_kvmem_mtp::llama_memory_kvmem_mtp(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams,
        llama_memory_kvmem * target) :
    model_(model),
    target_(target) {
    GGML_ASSERT(target_ && "MTP follower requires a KVMem target");
    trace_ = getenv("KVMEM_TRACE") != nullptr;
    kv_size_ = target_->kv_size();
    block_tokens_ = target_->block_tokens();
    if (block_tokens_ == 0) {
        block_tokens_ = 32;
    }
    n_layer_trunk_ = model.hparams.n_layer();
    il_graph_ = n_layer_trunk_;
    n_embd_k_ = target_->n_embd_k();
    n_embd_v_ = target_->n_embd_v();
    type_k_ = target_->type_k();
    type_v_ = target_->type_v();
    v_trans_ = target_->v_trans();
    rope_ = target_->rope();

    const uint32_t n_layer = n_layer_trunk_;
    llama_kv_cache::layer_filter_cb filter =
            [n_layer](uint32_t il) { return il >= n_layer; };

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
            filter,
            nullptr,
            nullptr,
            "kvmem-mtp");

    kvmem::RawKvStoreConfig rcfg;
    rcfg.n_layer = std::max(1u, model.hparams.n_layer_nextn);
    const uint32_t il_mtp = n_layer;
    rcfg.n_embd_k = model.hparams.n_embd_k_gqa(il_mtp);
    rcfg.n_embd_v = model.hparams.n_embd_v_gqa(il_mtp);
    rcfg.block_tokens = block_tokens_;
    if (ggml_is_quantized(type_k_)) {
        rcfg.k_row_bytes = ggml_row_size(type_k_, n_embd_k_);
    }
    if (!v_trans_) {
        rcfg.v_gpu_row_bytes = ggml_row_size(type_v_, n_embd_v_);
    }
    const llama_kvmem_params * kp = llama_kvmem_get_params();
    if (kp && kp->raw_k_nvme) {
        const uint64_t ntok = std::max<uint64_t>(cparams.n_ctx, kv_size_);
        const uint64_t need =
                static_cast<uint64_t>(rcfg.n_layer) * ntok *
                (static_cast<uint64_t>(rcfg.n_embd_k) + rcfg.n_embd_v) *
                sizeof(uint16_t);
        rcfg.nvme_bytes = need + 32ull * 1024ull * 1024ull;
        rcfg.nvme_dir = (kp->nvme_dir && kp->nvme_dir[0])
                ? kp->nvme_dir
                : "/tmp/kvmem_nvme";
        rcfg.nvme_file = "kvmem_raw_mtp_k.bin";
    }
    raw_ = std::make_unique<kvmem::RawKvStore>(rcfg);

    size_t bytes = 0;
    for (const auto & kv : kv_->memory_breakdown()) {
        bytes += kv.second;
    }
    const uint32_t n_layers = (uint32_t) kv_->get_layer_ids().size();
    fprintf(stderr,
            "KVMEM_TRACE mtp_pool cells=%u target_cells=%u n_ctx=%u bytes=%zu layers=%u block_tokens=%u\n",
            kv_size_, target_->kv_size(), cparams.n_ctx, bytes, n_layers, block_tokens_);
    LLAMA_LOG_INFO(
            "%s: KVMem MTP follower cells=%u target_cells=%u n_ctx=%u bytes=%.2f MiB layers=%u\n",
            __func__, kv_size_, target_->kv_size(), cparams.n_ctx,
            bytes / (1024.0 * 1024.0), n_layers);
    target_->set_mtp_follower(this);
    kvmem_mtp_bind(this);
}

llama_memory_kvmem_mtp::~llama_memory_kvmem_mtp() {
    kvmem_mtp_unbind(this);
    if (target_) {
        target_->set_mtp_follower(nullptr);
        target_ = nullptr;
    }
}

bool llama_memory_kvmem_mtp::fill_from_target(
        const llama_ubatch & ubatch,
        llama_kv_cache::slot_info & out) {
    if (!target_ || ubatch.n_tokens == 0 || !ubatch.pos) {
        return false;
    }
    out.s0 = 0;
    out.s1 = 0;
    out.resize(1);
    out.strm[0] = 0;
    out.idxs[0].clear();
    out.idxs[0].reserve(ubatch.n_tokens);

    int32_t first_slot = -1;
    uint32_t first_cell = 0;
    std::vector<llama_pos> pos_note;
    pos_note.reserve(ubatch.n_tokens);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id && ubatch.n_seq_id[i] > 1) {
            LLAMA_LOG_ERROR("%s: KVMem MTP is single-sequence only\n", __func__);
            return false;
        }
        const llama_pos pos = ubatch.pos[i];
        int32_t slot = -1;
        uint32_t off = 0;
        if (!target_->slot_for_orig_pos(pos, &slot, &off)) {
            LLAMA_LOG_ERROR("%s: no target slot for pos %d\n", __func__, (int) pos);
            return false;
        }
        const uint32_t cell = static_cast<uint32_t>(slot) * block_tokens_ + off;
        if (cell >= kv_size_) {
            LLAMA_LOG_ERROR("%s: cell %u >= kv_size %u (slot %d pos %d)\n",
                    __func__, cell, kv_size_, (int) slot, (int) pos);
            return false;
        }
        if (i == 0) {
            first_slot = slot;
            first_cell = cell;
        }
        out.idxs[0].push_back(cell);
        pos_note.push_back(pos);
    }
    pos_queue_.push_back(std::move(pos_note));
    if (trace_ && ubatch.n_tokens > 0) {
        fprintf(stderr,
                "KVMEM_TRACE mtp_occupy n=%u first_pos=%d first_slot=%d first_cell=%u "
                "tgt_slot=%d kv_size=%u\n",
                ubatch.n_tokens, (int) ubatch.pos[0], (int) first_slot, first_cell,
                (int) first_slot, kv_size_);
    }
    return out.idxs[0].size() == ubatch.n_tokens;
}

llama_memory_context_ptr llama_memory_kvmem_mtp::init_batch(
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
        llama_kv_cache::slot_info_vec_t sinfos;
        sinfos.reserve(ubatches.size());
        bool ok = true;
        for (const auto & ubatch : ubatches) {
            llama_kv_cache::slot_info sinfo;
            if (!fill_from_target(ubatch, sinfo)) {
                ok = false;
                break;
            }
            sinfos.push_back(std::move(sinfo));
        }
        if (!ok) {
            break;
        }
        return std::make_unique<llama_kv_cache_context>(
                kv_.get(), std::move(sinfos), std::move(ubatches));
    } while (false);
    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_kvmem_mtp::init_full() {
    return kv_->init_full();
}

llama_memory_context_ptr llama_memory_kvmem_mtp::init_update(llama_context * lctx, bool optimize) {
    return kv_->init_update(lctx, optimize);
}

void llama_memory_kvmem_mtp::clear(bool data) {
    kv_->clear(data);
}

bool llama_memory_kvmem_mtp::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return kv_->seq_rm(seq_id, p0, p1);
}

void llama_memory_kvmem_mtp::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_kvmem_mtp::seq_keep(llama_seq_id seq_id) {
    kv_->seq_keep(seq_id);
}

void llama_memory_kvmem_mtp::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_kvmem_mtp::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_kvmem_mtp::seq_pos_min(llama_seq_id seq_id) const {
    return kv_->seq_pos_min(seq_id);
}

llama_pos llama_memory_kvmem_mtp::seq_pos_max(llama_seq_id seq_id) const {
    return kv_->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_kvmem_mtp::memory_breakdown() const {
    return kv_->memory_breakdown();
}

void llama_memory_kvmem_mtp::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    kv_->state_write(io, seq_id, flags);
}

void llama_memory_kvmem_mtp::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    kv_->state_read(io, seq_id, flags);
}

void llama_memory_kvmem_mtp::register_capture(struct ggml_tensor * t, int il, char which) {
    GGML_UNUSED(il);
    if (!t) {
        return;
    }
    pending_capture_.push_back({t, which});
}

void llama_memory_kvmem_mtp::capture_on_new_graph() {
    pending_capture_.clear();
}

void llama_memory_kvmem_mtp::harvest_flush() {
    if (raw_) {
        raw_->wait_writes();
    }
}

void llama_memory_kvmem_mtp::harvest_capture(struct ggml_tensor * t, char which) {
    if (!t || cur_pos_.empty() || !raw_) {
        return;
    }
    if (!llama_kvmem_want_prefill_capture()) {
        return;
    }
    if (which != 'k' && which != 'v') {
        return;
    }
    std::vector<uint8_t> tmp(ggml_nbytes(t));
    const int64_t t0 = ggml_time_us();
    ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
    perf_sync_us_ += ggml_time_us() - t0;
    const uint32_t n = (uint32_t) cur_pos_.size();
    const uint32_t pos0 = (uint32_t) cur_pos_[0];
    const int64_t d = t->ne[0];
    const int64_t h = t->ne[1] > 0 ? t->ne[1] : 1;
    const int64_t ntok = t->ne[2] > 0 ? t->ne[2] : 1;
    const uint32_t dim = which == 'v' ? n_embd_v_ : n_embd_k_;
    std::vector<float> flat((size_t) n * dim, 0.0f);
    const uint32_t n_use = std::min(n, (uint32_t) ntok);
    for (uint32_t tok = 0; tok < n_use; ++tok) {
        for (int64_t head = 0; head < h; ++head) {
            for (int64_t dim0 = 0; dim0 < d; ++dim0) {
                const size_t off = (size_t) tok * t->nb[2] + (size_t) head * t->nb[1]
                        + (size_t) dim0 * t->nb[0];
                float val = 0.0f;
                if (t->type == GGML_TYPE_F32) {
                    val = *reinterpret_cast<const float *>(tmp.data() + off);
                } else if (t->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(tmp.data() + off));
                } else {
                    return;
                }
                const size_t dst = (size_t) tok * dim + (size_t) head * (size_t) d + (size_t) dim0;
                if (dst < flat.size()) {
                    flat[dst] = val;
                }
            }
        }
    }
    const uint64_t b0 = raw_->nvme_bytes_written();
    const uint64_t sc0 = raw_->nvme_syscalls();
    if (which == 'k') {
        if (ggml_is_quantized(type_k_)) {
            const size_t krow = ggml_row_size(type_k_, n_embd_k_);
            std::vector<uint8_t> packed((size_t) n * krow);
            if (kvmem_cache_pack_rows(type_k_, flat.data(), packed.data(),
                                      (int64_t) n, (int64_t) n_embd_k_)) {
                raw_->write_layer_k_rows(pos0, n, 0, packed.data(), flat.data());
            }
        } else {
            raw_->write_layer_tokens(pos0, n, 0, flat.data(), nullptr);
        }
    } else {
        raw_->write_layer_tokens(pos0, n, 0, nullptr, flat.data());
    }
    perf_nvme_bytes_ += raw_->nvme_bytes_written() - b0;
    perf_nvme_syscalls_ += raw_->nvme_syscalls() - sc0;
}

void llama_memory_kvmem_mtp::harvest_pending(struct ggml_backend_sched * sched) {
    // Match trunk: keep pending tensors across graph reuse. Clearing them
    // here drops MTP K for every ubatch after the first of a given size.
    if (!llama_kvmem_want_prefill_capture()) {
        pending_capture_.clear();
        pos_queue_.clear();
        return;
    }
    if (pending_capture_.empty()) {
        pos_queue_.clear();
        return;
    }
    if (pos_queue_.empty()) {
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
    if (be) {
        const char * sync_old = getenv("KVMEM_HARVEST_SYNC");
        const bool old = sync_old && sync_old[0] != '\0' && sync_old[0] != '0';
        const int64_t t0 = ggml_time_us();
        if (old) {
            ggml_backend_synchronize(be);
        } else {
            ggml_backend_event_t ev = ggml_backend_event_new(ggml_backend_get_device(be));
            if (ev) {
                ggml_backend_event_record(ev, be);
                ggml_backend_event_synchronize(ev);
                ggml_backend_event_free(ev);
            } else {
                ggml_backend_synchronize(be);
            }
        }
        perf_sync_us_ += ggml_time_us() - t0;
    }
    cur_pos_ = std::move(pos_queue_.front());
    pos_queue_.erase(pos_queue_.begin());
    for (const CaptureNode & n : pending_capture_) {
        harvest_capture(n.t, n.which);
    }
    perf_n_ubatch_ += 1;
    const char * perf = getenv("KVMEM_PERF");
    if (perf && perf[0] != '\0' && perf[0] != '0') {
        fprintf(stderr,
                "KVMEM_HARVEST ubatch=%u n=%zu is_mtp=1 "
                "harvest_entry_us=0 sync_us=%lld d2d_us=0 snap_wait_us=0 "
                "d2h_submit_us=0 commit_us=0 pack_us=0 "
                "nvme_us=0 nvme_bytes=%llu nvme_syscalls=%llu\n",
                perf_n_ubatch_, cur_pos_.size(),
                (long long) perf_sync_us_,
                (unsigned long long) perf_nvme_bytes_,
                (unsigned long long) perf_nvme_syscalls_);
    }
    if (trace_) {
        fprintf(stderr, "KVMEM_TRACE mtp_harvest n=%zu which=%zu queue=%zu\n",
                cur_pos_.size(), pending_capture_.size(), pos_queue_.size());
    }
}

void llama_memory_kvmem_mtp::occupy_block(uint32_t block_id) {
    if (!target_ || !kv_) {
        return;
    }
    const auto & st = target_->store();
    if (block_id >= st.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = st.blocks()[block_id];
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
        sinfo.idxs[0][t] = (uint32_t) blk.gpu_slot * block_tokens_ + t;
        pos[t] = (llama_pos) (blk.orig_pos_start + t);
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

void llama_memory_kvmem_mtp::harvest_v(uint32_t block_id) {
    if (!target_ || !kv_ || !raw_ || v_trans_) {
        return;
    }
    const auto & st = target_->store();
    if (block_id >= st.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = st.blocks()[block_id];
    if (blk.gpu_slot < 0 || blk.n_tokens == 0) {
        return;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx0 = (uint32_t) blk.gpu_slot * block_tokens_;
    if (idx0 >= cells.size() || cells.is_empty(idx0)
            || cells.pos_get(idx0) != (llama_pos) blk.orig_pos_start) {
        return;
    }
    if (raw_->has_v(block_id, 0)) {
        return;
    }
    const uint32_t nt = blk.n_tokens;
    const uint32_t cell0 = (uint32_t) blk.gpu_slot * block_tokens_;
    ggml_tensor * vt = kv_->get_v_storage((int32_t) il_graph_);
    if (!vt) {
        return;
    }
    const size_t row = ggml_row_size(type_v_, n_embd_v_);
    std::vector<uint8_t> packed((size_t) nt * row);
    ggml_backend_tensor_get(vt, packed.data(), (size_t) cell0 * row, (size_t) nt * row);
    raw_->write_layer_v_gpu(blk.orig_pos_start, nt, 0, packed.data());
}

void llama_memory_kvmem_mtp::write_block_to_gpu(uint32_t block_id) {
    if (!target_ || !kv_ || !raw_) {
        return;
    }
    const auto & st = target_->store();
    if (block_id >= st.block_count() || !raw_->has_block(block_id)) {
        return;
    }
    const kvmem::KvMemBlock & blk = st.blocks()[block_id];
    if (blk.gpu_slot < 0) {
        return;
    }
    const uint32_t nt = blk.n_tokens;
    occupy_block(block_id);
    ggml_tensor * kt = kv_->get_k_storage((int32_t) il_graph_);
    ggml_tensor * vt = kv_->get_v_storage((int32_t) il_graph_);
    if (!kt) {
        return;
    }
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t cell0 = (uint32_t) blk.gpu_slot * block_tokens_;
    kvmem_stagein_gpu_ready((size_t) block_tokens_ * std::max(n_embd_k_, n_embd_v_),
                            std::max(krow, vrow) * (size_t) block_tokens_);
    std::vector<uint8_t> kpack((size_t) nt * krow);
    std::vector<uint8_t> vpack((size_t) nt * vrow);
    std::vector<float> rk((size_t) nt * n_embd_k_);
    bool have_k = false;
    const bool k_quant = ggml_is_quantized(type_k_);
    if (k_quant) {
        have_k = raw_->copy_k_rows(block_id, 0, kpack.data(), nt);
    } else {
        have_k = raw_->copy_k(block_id, 0, rk.data());
    }
    if (!have_k) {
        return;
    }
    const int n_head = static_cast<int>(rope_.n_head_kv);
    const int n_eh = static_cast<int>(rope_.n_embd_head);
    const int nrot_k = kvmem_attn_rot_on(type_k_, n_eh) ? kvmem_hadamard_nrot_k(n_eh) : 0;
    const int nrot_v = kvmem_attn_rot_on(type_v_, n_eh) ? kvmem_hadamard_nrot_v(n_eh) : 0;
    const int n_rot_rope = (int) rope_.n_rot;
    const int n_theta = n_rot_rope / 2;
    std::vector<float> theta;
    if (n_theta > 0) {
        theta.resize((size_t) n_theta);
        for (int i = 0; i < n_theta; ++i) {
            theta[(size_t) i] = std::pow(rope_.freq_base,
                    -2.0f * (float) i / (float) n_rot_rope) * rope_.freq_scale;
        }
    }
    uint8_t * kbase = kvmem_mtp_cuda_ptr(kt);
    uint8_t * vbase = (vt && !v_trans_) ? kvmem_mtp_cuda_ptr(vt) : nullptr;
    bool k_gpu = false;
    if (k_quant && kbase && !theta.empty()) {
        const size_t nbytes = krow * (size_t) nt;
        k_gpu = kvmem_stagein_enqueue_k(
                type_k_, kpack.data(), nbytes, kbase + cell0 * krow,
                (int64_t) nt, (int64_t) n_embd_k_, nrot_k, n_head, n_eh, n_rot_rope,
                (int32_t) blk.orig_pos_start, theta.data(), n_theta,
                nullptr, nullptr, nullptr, nullptr);
        if (!k_gpu) {
            k_gpu = kvmem_stagein_quant_ok(type_k_) &&
                    (nrot_k == 0 || kvmem_stagein_fwht_ok(nrot_k)) &&
                    kvmem_stagein_h2d_packed(kpack.data(), nbytes) &&
                    kvmem_stagein_dequant(type_k_, (int64_t) nt, (int64_t) n_embd_k_) &&
                    kvmem_stagein_rope_neox((int64_t) nt, n_head, n_eh, n_rot_rope,
                                            (int32_t) blk.orig_pos_start, theta.data(), n_theta) &&
                    (nrot_k == 0 || kvmem_stagein_fwht((int64_t) nt, (int64_t) n_embd_k_, nrot_k)) &&
                    kvmem_stagein_quantize(type_k_, kbase + cell0 * krow, (int64_t) nt,
                                           (int64_t) n_embd_k_);
        }
    }
    if (!k_gpu) {
        std::vector<float> roped((size_t) nt * n_embd_k_);
        if (k_quant && !kvmem_cache_unpack_rows(type_k_, kpack.data(), rk.data(),
                                                (int64_t) nt, (int64_t) n_embd_k_)) {
            return;
        }
        kvmem::rope_neox_apply(rope_, rk.data(), nt, (int32_t) blk.orig_pos_start, roped.data());
        bool k_gpu_f32 = false;
        if (kbase) {
            k_gpu_f32 = kvmem_stagein_quant_ok(type_k_) &&
                    (nrot_k == 0 || kvmem_stagein_fwht_ok(nrot_k)) &&
                    kvmem_stagein_h2d_f32(roped.data(), (int64_t) nt * n_embd_k_) &&
                    (nrot_k == 0 || kvmem_stagein_fwht((int64_t) nt, (int64_t) n_embd_k_, nrot_k)) &&
                    kvmem_stagein_quantize(type_k_, kbase + cell0 * krow, (int64_t) nt,
                                           (int64_t) n_embd_k_);
        }
        if (!k_gpu_f32) {
            if (nrot_k > 0) {
                kvmem_hadamard_rows(roped.data(), (int64_t) nt, n_head, n_eh, nrot_k);
            }
            if (!kvmem_cache_pack_rows(type_k_, roped.data(), kpack.data(),
                                       (int64_t) nt, (int64_t) n_embd_k_)) {
                return;
            }
            ggml_backend_tensor_set(kt, kpack.data(), cell0 * krow, nt * krow);
        }
    }
    std::vector<float> rv;
    const bool have_gpu_v = raw_->copy_v_gpu(block_id, 0, vpack.data(), nt);
    bool have_v = have_gpu_v;
    if (!have_gpu_v) {
        rv.assign((size_t) nt * n_embd_v_, 0.0f);
        have_v = raw_->copy_v(block_id, 0, rv.data());
    }
    if (have_v && vt && !v_trans_) {
        if (have_gpu_v) {
            const size_t nbytes = vrow * (size_t) nt;
            if (!(vbase && kvmem_stagein_enqueue_v(vpack.data(), nbytes,
                                                   vbase + cell0 * vrow, nullptr))) {
                if (!(vbase && kvmem_stagein_h2d_bytes(vbase + cell0 * vrow, vpack.data(),
                                                       nbytes))) {
                    ggml_backend_tensor_set(vt, vpack.data(), cell0 * vrow, nt * vrow);
                }
            }
        } else {
            bool v_gpu = false;
            if (vbase && kvmem_stagein_quant_ok(type_v_) &&
                (nrot_v == 0 || kvmem_stagein_fwht_ok(nrot_v))) {
                v_gpu = kvmem_stagein_h2d_f32(rv.data(), (int64_t) nt * n_embd_v_) &&
                        (nrot_v == 0 || kvmem_stagein_fwht((int64_t) nt, (int64_t) n_embd_v_, nrot_v)) &&
                        kvmem_stagein_quantize(type_v_, vbase + cell0 * vrow, (int64_t) nt,
                                               (int64_t) n_embd_v_);
            }
            if (!v_gpu) {
                if (nrot_v > 0) {
                    kvmem_hadamard_rows(rv.data(), (int64_t) nt, n_head, n_eh, nrot_v);
                }
                if (kvmem_cache_pack_rows(type_v_, rv.data(), vpack.data(),
                                          (int64_t) nt, (int64_t) n_embd_v_)) {
                    ggml_backend_tensor_set(vt, vpack.data(), cell0 * vrow, nt * vrow);
                }
            }
        }
    }
}

void llama_memory_kvmem_mtp::on_stage_out(uint32_t block_id) {
    if (!target_) {
        return;
    }
    harvest_v(block_id);
    const auto & st = target_->store();
    if (block_id >= st.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & b = st.blocks()[block_id];
    if (b.n_tokens > 0) {
        kv_->seq_rm(0, (llama_pos) b.orig_pos_start, (llama_pos) b.orig_pos_end());
    }
}

void llama_memory_kvmem_mtp::harvest_resident_v() {
    if (!target_ || !kv_) {
        return;
    }
    // Only copy V from cells that still hold this block. Newly admitted
    // stage_in slots are empty here; overwriting raw would wipe the V that
    // on_stage_out saved when the block left the pool during prefill.
    const llama_kv_cells & cells = kv_->get_cells(0);
    for (const auto & b : target_->store().blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        const uint32_t idx = (uint32_t) b.gpu_slot * block_tokens_;
        if (idx >= cells.size() || cells.is_empty(idx)) {
            continue;
        }
        if (cells.pos_get(idx) != (llama_pos) b.orig_pos_start) {
            continue;
        }
        harvest_v(b.block_id);
    }
}

void llama_memory_kvmem_mtp::follow_retrieval() {
    if (!target_ || !kv_) {
        return;
    }
    kv_->seq_rm(0, 0, -1);
    uint32_t n_gpu = 0;
    uint32_t n_wb = 0;
    uint32_t n_miss = 0;
    if (trace_) {
        fprintf(stderr, "KVMEM_TRACE mtp_selected");
    }
    for (const auto & b : target_->store().blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        n_gpu++;
        if (trace_) {
            fprintf(stderr, " %u", b.block_id);
        }
        if (raw_ && raw_->has_block(b.block_id) && raw_->has_k(b.block_id, 0)) {
            write_block_to_gpu(b.block_id);
            n_wb++;
        } else {
            n_miss++;
        }
    }
    kvmem_stagein_flush(nullptr, nullptr, nullptr, nullptr);
    kvmem_stagein_sync();
    if (trace_) {
        fprintf(stderr, "\n");
    }
    fprintf(stderr,
            "KVMEM_TRACE mtp_follow n_gpu=%u n_writeback=%u n_no_raw=%u seq_pos=[%d,%d]\n",
            n_gpu, n_wb, n_miss,
            (int) kv_->seq_pos_min(0), (int) kv_->seq_pos_max(0));
}
