#pragma once

#include <set>
#include <numeric>

static void multimodal_validate_capacity(const ServerState & st, const kvmem_prompt & prompt, int query, int end) {
    if (!st.kparams.enabled || !st.kparams.budget || !prompt.has_media()) return;
    const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
    const uint32_t budget = st.kparams.budget / block;
    std::vector<std::pair<uint32_t, uint32_t>> groups;
    for (const auto & range : prompt.media_ranges()) {
        const uint32_t lo = (range.first ? range.first - 1 : 0) / block;
        const uint32_t hi = (std::min<uint32_t>(end, range.second + 1) + block - 1) / block;
        if (!groups.empty() && lo < groups.back().second) groups.back().second = hi;
        else groups.emplace_back(lo, hi);
    }
    const uint32_t sink = st.kparams.sink_tokens ? (st.kparams.sink_tokens + block - 1) / block : 1;
    for (const auto & group : groups) {
        if (group.second - group.first + std::min(group.first, sink) > budget)
            throw std::invalid_argument("image group exceeds KV budget; reduce --image-max-tokens or increase --kvmem-budget");
    }
    std::set<uint32_t> required;
    for (uint32_t i = 0; i < sink; ++i) required.insert(i);
    for (uint32_t i = groups.back().first; i < groups.back().second; ++i) required.insert(i);
    for (uint32_t i = std::max(0, query) / block; i < ((uint32_t) end + block - 1) / block; ++i) required.insert(i);
    if (required.size() > budget)
        throw std::invalid_argument("latest image and text query exceed KV budget; reduce the image size or query span");
}

// Included after the single-slot server state and stream helpers.
static MultimodalCheckpoint multimodal_checkpoint(ServerState & st, int row) {
    MultimodalCheckpoint result;
    result.row = row;
    llama_synchronize(st.ctx);
    llama_kvmem_get_tail_mean(row, result.tail_mean);
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t size = llama_state_seq_get_size_ext(st.ctx, 0, flags);
    result.recurrent.resize(size);
    if (llama_state_seq_get_data_ext(st.ctx, result.recurrent.data(), size, 0, flags) != size) {
        throw std::runtime_error("multimodal recurrent checkpoint failed");
    }
    if (st.spec.ok && !common_speculative_get_state(st.spec.spec, 0, result.draft_carry)) {
        throw std::runtime_error("MTP carry checkpoint failed");
    }
    return result;
}

static void multimodal_remember(ServerState & st, MultimodalCheckpoint checkpoint) {
    auto & entries = st.mm_checkpoints;
    entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const auto & entry) {
        return entry.row >= checkpoint.row;
    }), entries.end());
    entries.push_back(std::move(checkpoint));
    if (entries.size() > 4) {
        const auto media_count = std::count_if(entries.begin(), entries.end(), [](const auto & e) { return e.media_boundary; });
        auto victim = std::find_if(entries.begin(), entries.end(), [&](const auto & e) {
            return e.media_boundary == (media_count > 2);
        });
        entries.erase(victim == entries.end() ? entries.begin() : victim);
    }
}

static void multimodal_restore(ServerState & st, const MultimodalCheckpoint & checkpoint, bool truncate) {
    llama_synchronize(st.ctx);
    if (st.spec.ctx_dft) llama_synchronize(st.spec.ctx_dft);
    llama_kvmem_decode_mean_flush();
    llama_kvmem_decode_mean_discard();
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const auto & data = checkpoint.recurrent;
    if (llama_state_seq_set_data_ext(st.ctx, data.data(), data.size(), 0, flags) != data.size()) {
        throw std::runtime_error("multimodal recurrent restore failed");
    }
    if (!llama_kvmem_remove_logical(st.ctx, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted target rows");
    }
    if (st.spec.ctx_dft && !llama_kvmem_remove_logical(st.spec.ctx_dft, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted MTP rows");
    }
    if (st.spec.ok) common_speculative_set_state(st.spec.spec, 0, checkpoint.draft_carry);
    if (truncate) {
        llama_kvmem_truncate_cached(checkpoint.row);
        llama_kvmem_set_tail_mean(checkpoint.row, checkpoint.tail_mean);
    }
    st.mm_live_row = checkpoint.row;
}

static void multimodal_finish_request(ServerState & st) {
    if (!st.vision || st.mm_committed || !st.mm_rollback) return;
    try {
        llama_kvmem_set_replay(false);
        multimodal_restore(st, *st.mm_rollback, true);
        llama_kvmem_begin_cached_turn();
        st.cached_prompt = st.mm_rollback_prompt;
        st.cached_tokens = st.cached_prompt ? st.cached_prompt->tokens : std::vector<llama_token>{};
        st.cached_tokens.resize(std::min(st.cached_tokens.size(), (size_t) st.mm_live_row));
        multimodal_remember(st, *st.mm_rollback);
        fprintf(stderr, "KVMEM_TRACE multimodal_rollback context=%p row=%d\n", (void *) st.ctx, st.mm_live_row);
        st.mm_committed = true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        fprintf(stderr, "KVMEM_TRACE multimodal_rollback_failed error=%s\n", e.what());
    }
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
}

static int multimodal_decode_span(ServerState & st, int begin, int end, bool replay, StreamIo * io) {
    const auto & prompt = *st.active_prompt;
    auto dispatch = [&](llama_batch batch) -> int {
        if (!stream_heartbeat(io)) return KVMEM_DECODE_ABORT;
        fprintf(stderr, "KVMEM_TRACE multimodal_decode context=%p rows=[%d,%d) model_pos=%d image=%d replay=%d\n",
                (void *) st.ctx, batch.logical_pos[0], batch.logical_pos[batch.n_tokens - 1] + 1,
                batch.pos[0], batch.token == nullptr, replay);
        const auto start = std::chrono::steady_clock::now();
        int rc = llama_decode(st.ctx, batch);
        if (rc == 0 && st.spec.ok && !common_speculative_process(st.spec.spec, batch)) rc = -1;
        llama_synchronize(st.ctx);
        fprintf(stderr, "KVMEM_TRACE multimodal_compute rows=%d elapsed_ms=%.3f image=%d replay=%d\n",
                batch.n_tokens, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                batch.token == nullptr, replay);
        if (rc == 0) {
            st.mm_live_row = batch.logical_pos[batch.n_tokens - 1] + 1;
            if (replay) st.mm_replayed += batch.n_tokens;
            else {
                const int tail = std::clamp(st.mm_lcp - batch.logical_pos[0], 0, batch.n_tokens);
                st.mm_tail_replayed += tail;
                if (batch.token) st.mm_new_text += batch.n_tokens - tail;
                else st.mm_new_image += batch.n_tokens - tail;
            }
        }
        return rc;
    };
    int row = begin;
    while (row < end) {
        if (!stream_heartbeat(io)) return KVMEM_DECODE_ABORT;
        if (prompt.tokens[row] == LLAMA_TOKEN_NULL) {
            const int next = (int) prompt.media_end(row);
            if (next > end) throw std::runtime_error("prefill boundary splits an image");
            if (!replay) {
                auto checkpoint = multimodal_checkpoint(st, row);
                checkpoint.media_boundary = true;
                multimodal_remember(st, std::move(checkpoint));
            }
            const int rc = st.vision->decode(st.ctx, prompt, row, st.n_batch, dispatch);
            if (rc != 0) return rc;
            row = next;
            continue;
        }
        const int limit = std::min(end, row + st.n_batch);
        int next = row;
        while (next < limit && prompt.tokens[next] != LLAMA_TOKEN_NULL) ++next;
        std::vector<llama_pos> pos(next - row), logical(next - row);
        const auto pos0 = prompt.model_pos(row);
        for (int i = row; i < next; ++i) {
            pos[i - row] = pos0 + i - row;
            logical[i - row] = i;
        }
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(prompt.tokens.data()) + row, next - row);
        batch.pos = pos.data();
        batch.logical_pos = logical.data();
        std::vector<int32_t> n_seq(next - row, 1);
        llama_seq_id seq = 0;
        std::vector<llama_seq_id *> seq_ids(next - row, &seq);
        std::vector<int8_t> outputs(next - row, 0);
        outputs.back() = !st.spec.ok;
        batch.n_seq_id = n_seq.data();
        batch.seq_id = seq_ids.data();
        batch.logits = outputs.data();
        const int rc = dispatch(batch);
        if (rc != 0) return rc;
        row = next;
    }
    return 0;
}

static bool run_prefill_multimodal(ServerState & st, StreamIo * io, int * n_cache_hit) {
    try {
        st.mm_error.clear();
        st.mm_error_status = 500;
        if (st.mm_reset_requested) {
            fprintf(stderr, "KVMEM_TRACE multimodal_reset context=%p reason=explicit_cache_reset\n", (void *) st.ctx);
            memory_clear_all(st);
            st.mm_reset_requested = false;
        }
        st.mm_new_text = st.mm_new_image = st.mm_replayed = st.mm_tail_replayed = 0;
        st.vision->reset_stats();
        const auto & prompt = *st.active_prompt;
        const int eval_end = (int) prompt.tokens.size() - (st.spec.ok ? 1 : 0);
        const int lcp = st.cached_prompt ? (int) prompt.common_prefix(*st.cached_prompt) : 0;
        st.mm_lcp = lcp;
        // Sequence checkpoints do not restore logits. Ordinary decoding must evaluate
        // at least one token; MTP evaluates the pending prompt token in spec_generate.
        const int keep = std::min({lcp, st.mm_live_row, eval_end - (st.spec.ok ? 0 : 1)});
        MultimodalCheckpoint base;
        bool found = false;
        for (const auto & checkpoint : st.mm_checkpoints) {
            if (checkpoint.row <= keep && (!found || checkpoint.row > base.row)) {
                base = checkpoint;
                found = true;
            }
        }
        if (!found) {
            // Shared template tokens do not identify a conversation. Like llama-server,
            // treat a missing recurrent checkpoint as a cache miss and evaluate the supplied prompt.
            fprintf(stderr, "KVMEM_TRACE multimodal_reset context=%p reason=%s lcp=%d keep=%d cached_rows=%zu live_rows=%d oldest_checkpoint=%d checkpoint_count=%zu\n",
                    (void *) st.ctx, st.cached_prompt ? "no_recurrent_checkpoint" : "new_conversation",
                    lcp, keep, st.cached_tokens.size(), st.mm_live_row,
                    st.mm_checkpoints.empty() ? -1 : st.mm_checkpoints.front().row, st.mm_checkpoints.size());
            memory_clear_all(st);
            st.mm_lcp = 0; // No old rows survived the reset; count all evaluated rows as new.
            base = multimodal_checkpoint(st, 0);
            if (st.spec.ok) {
                std::fill(base.draft_carry.begin(), base.draft_carry.end(), 0);
                common_speculative_set_state(st.spec.spec, 0, base.draft_carry);
            }
        }
        st.mm_rollback = std::make_shared<MultimodalCheckpoint>(base);
        st.mm_rollback_prompt = st.cached_prompt ? st.cached_prompt->prefix(base.row) : nullptr;
        st.mm_committed = false;
        multimodal_restore(st, base, true);
        llama_kvmem_begin_cached_turn();
        std::vector<uint32_t> starts, ends;
        const auto ranges = prompt.media_ranges();
        for (const auto & range : ranges) {
            starts.push_back(range.first > 0 ? range.first - 1 : 0);
            ends.push_back(std::min<uint32_t>(prompt.tokens.size(), range.second + 1));
        }
        llama_kvmem_set_media_ranges(starts.data(), ends.data(), starts.size());
        int query = std::max(base.row, std::min(st.kparams.query_begin, eval_end));
        if (!ranges.empty()) query = std::max(query, (int) ranges.back().second);
        query = std::min(query, eval_end);
        const bool retrieve = st.kparams.enabled && st.kparams.method == 1 && query < eval_end;
        llama_kvmem_set_request_span(query, eval_end, st.kparams.force_pos);
        if (n_cache_hit) *n_cache_hit = base.row;
        if (multimodal_decode_span(st, base.row, query, false, io) != 0) throw std::runtime_error("multimodal prefill failed or cancelled");
        auto query_checkpoint = multimodal_checkpoint(st, query);
        multimodal_remember(st, query_checkpoint);
        if (multimodal_decode_span(st, query, eval_end, false, io) != 0) throw std::runtime_error("multimodal query prefill failed or cancelled");
        if (retrieve) {
            llama_kvmem_apply_retrieval(st.ctx);
            // Replay only the new text query against the selected history.
            multimodal_restore(st, query_checkpoint, false);
            llama_kvmem_set_replay(true);
            const int rc = multimodal_decode_span(st, query, eval_end, true, io);
            llama_kvmem_set_replay(false);
            if (rc != 0) throw std::runtime_error("multimodal query replay failed or cancelled");
        }
        llama_kvmem_pin_working_set();
        multimodal_remember(st, multimodal_checkpoint(st, eval_end));
        std::vector<uint8_t> carry;
        llama_pos synced = 0;
        if (st.spec.ok) {
            common_speculative_get_state(st.spec.spec, 0, carry);
            if (carry.size() < sizeof(synced)) throw std::runtime_error("MTP carry missing");
            std::memcpy(&synced, carry.data(), sizeof(synced));
            if (synced != eval_end) throw std::runtime_error("MTP has unsynchronized visual rows");
        }
        fprintf(stderr, "KVMEM_TRACE multimodal_prefill context=%p prefix_hit_rows=%d lcp=%d new_text_rows=%u new_image_rows=%u replayed_rows=%u vision_encode_calls=%u encoder_ms=%.2f logical_cursor=%d model_cursor=%d mtp_synced_rows=%d cached_tail_rows=%u replay_reason=query embedding_cache_bytes=%zu checkpoint_bytes=%zu\n",
                (void *) st.ctx, base.row, lcp, st.mm_new_text, st.mm_new_image, st.mm_replayed,
                st.vision->encode_calls, st.vision->encode_ms, eval_end, prompt.model_pos(eval_end), synced, st.mm_tail_replayed, st.vision->cache_bytes(),
                std::accumulate(st.mm_checkpoints.begin(), st.mm_checkpoints.end(), size_t(0),
                    [](size_t n, const auto & c) { return n + c.recurrent.size() + c.draft_carry.size() + c.tail_mean.size()*sizeof(float); }));
        return true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        fprintf(stderr, "KVMEM_TRACE multimodal_error error=%s\n", e.what());
        multimodal_finish_request(st);
        return false;
    }
}

static void multimodal_commit(ServerState & st, const std::vector<llama_token> & gen) {
    st.cached_prompt = st.active_prompt->with_generated(gen);
    st.mm_live_row = st.kparams.enabled ? (int) llama_kvmem_store_n_tokens()
        : st.mm_live_row;
    multimodal_remember(st, multimodal_checkpoint(st, st.mm_live_row));
    st.mm_committed = true;
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
}

static int multimodal_decode_generated(ServerState & st, llama_token id, int row) {
    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_pos logical = row;
    llama_pos pos = st.active_prompt->model_pos(row);
    batch.logical_pos = &logical;
    batch.pos = &pos;
    const int rc = llama_decode(st.ctx, batch);
    if (rc == 0) st.mm_live_row = row + 1;
    return rc;
}
