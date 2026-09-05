#include "llama-memory-kvmem-mtp.h"

#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-kvmem-capture.h"
#include "llama-kvmem-hooks.h"
#include "llama-model.h"

#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

void llama_memory_kvmem_mtp::harvest_capture(struct ggml_tensor * t, char which) {
    if (!t || cur_pos_.empty() || !raw_) {
        return;
    }
    if (!llama_kvmem_want_prefill_capture()) {
        return;
    }
    std::vector<uint8_t> tmp(ggml_nbytes(t));
    ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
    const uint32_t n = (uint32_t) cur_pos_.size();
    const uint32_t pos0 = (uint32_t) cur_pos_[0];
    const int64_t d = t->ne[0];
    const int64_t h = t->ne[1] > 0 ? t->ne[1] : 1;
    const int64_t ntok = t->ne[2] > 0 ? t->ne[2] : 1;
    const uint32_t dim = n_embd_k_;
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
    const float * kptr = which == 'k' ? flat.data() : nullptr;
    const float * vptr = which == 'v' ? flat.data() : nullptr;
    raw_->write_layer_tokens(pos0, n, 0, kptr, vptr);
}

void llama_memory_kvmem_mtp::harvest_pending(struct ggml_backend_sched * /*sched*/) {
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
    cur_pos_ = std::move(pos_queue_.front());
    pos_queue_.erase(pos_queue_.begin());
    for (const CaptureNode & n : pending_capture_) {
        harvest_capture(n.t, n.which);
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
    ggml_tensor * vt = kv_->get_v_storage((int32_t) il_graph_);
    if (!vt) {
        return;
    }
    const uint32_t nt = blk.n_tokens;
    const size_t row = ggml_row_size(type_v_, n_embd_v_);
    std::vector<float> v((size_t) nt * n_embd_v_, 0.0f);
    std::vector<uint8_t> raw(row);
    for (uint32_t t = 0; t < nt; ++t) {
        const uint32_t cell = (uint32_t) blk.gpu_slot * block_tokens_ + t;
        ggml_backend_tensor_get(vt, raw.data(), (size_t) cell * row, row);
        float * dst = v.data() + t * n_embd_v_;
        if (type_v_ == GGML_TYPE_F16) {
            const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(raw.data());
            for (uint32_t d = 0; d < n_embd_v_; ++d) {
                dst[d] = ggml_fp16_to_fp32(src[d]);
            }
        } else if (type_v_ == GGML_TYPE_F32) {
            std::memcpy(dst, raw.data(), row);
        }
    }
    raw_->write_layer_tokens(blk.orig_pos_start, nt, 0, nullptr, v.data());
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
    const float * rk = raw_->k(block_id, 0);
    if (!rk) {
        return;
    }
    occupy_block(block_id);
    const uint32_t nt = blk.n_tokens;
    std::vector<float> roped((size_t) nt * n_embd_k_);
    kvmem::rope_neox_apply(rope_, rk, nt, (int32_t) blk.orig_pos_start, roped.data());
    ggml_tensor * kt = kv_->get_k_storage((int32_t) il_graph_);
    ggml_tensor * vt = kv_->get_v_storage((int32_t) il_graph_);
    if (!kt) {
        return;
    }
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    std::vector<ggml_fp16_t> k16(n_embd_k_);
    std::vector<ggml_fp16_t> v16(n_embd_v_);
    const float * rv = raw_->v(block_id, 0);
    for (uint32_t t = 0; t < nt; ++t) {
        const uint32_t cell = (uint32_t) blk.gpu_slot * block_tokens_ + t;
        const float * src = roped.data() + t * n_embd_k_;
        if (type_k_ == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(src, k16.data(), (int64_t) n_embd_k_);
            ggml_backend_tensor_set(kt, k16.data(), cell * krow, krow);
        } else {
            ggml_backend_tensor_set(kt, src, cell * krow, krow);
        }
        if (rv && vt && !v_trans_) {
            const float * vs = rv + t * n_embd_v_;
            if (type_v_ == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(vs, v16.data(), (int64_t) n_embd_v_);
                ggml_backend_tensor_set(vt, v16.data(), cell * vrow, vrow);
            } else {
                ggml_backend_tensor_set(vt, vs, cell * vrow, vrow);
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
    fprintf(stderr, "KVMEM_TRACE mtp_selected");
    for (const auto & b : target_->store().blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        n_gpu++;
        fprintf(stderr, " %u", b.block_id);
        if (raw_ && raw_->has_block(b.block_id) && raw_->k(b.block_id, 0)) {
            write_block_to_gpu(b.block_id);
            n_wb++;
        } else {
            n_miss++;
        }
    }
    fprintf(stderr, "\n");
    fprintf(stderr,
            "KVMEM_TRACE mtp_follow n_gpu=%u n_writeback=%u n_no_raw=%u seq_pos=[%d,%d]\n",
            n_gpu, n_wb, n_miss,
            (int) kv_->seq_pos_min(0), (int) kv_->seq_pos_max(0));
}
