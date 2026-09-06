#include "kvmem-spec.h"
#include "llama-kvmem-hooks.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

ggml_type kvmem_parse_cache_type(const char * s, bool * ok) {
    if (ok) {
        *ok = true;
    }
    if (!s) {
        if (ok) {
            *ok = false;
        }
        return GGML_TYPE_F16;
    }
    if (std::strcmp(s, "f16") == 0 || std::strcmp(s, "fp16") == 0) {
        return GGML_TYPE_F16;
    }
    if (std::strcmp(s, "f32") == 0 || std::strcmp(s, "fp32") == 0) {
        return GGML_TYPE_F32;
    }
    if (std::strcmp(s, "q8_0") == 0 || std::strcmp(s, "q8") == 0) {
        return GGML_TYPE_Q8_0;
    }
    if (std::strcmp(s, "q4_0") == 0 || std::strcmp(s, "q4") == 0) {
        return GGML_TYPE_Q4_0;
    }
    if (ok) {
        *ok = false;
    }
    return GGML_TYPE_F16;
}

bool kvmem_cache_types_ok(ggml_type type_k, ggml_type type_v) {
    if (ggml_is_quantized(type_k) || ggml_is_quantized(type_v)) {
        return type_k == type_v;
    }
    return true;
}

bool kvmem_spec_start(kvmem_spec_session & sess,
                      llama_model * model_tgt,
                      llama_context * ctx_tgt,
                      const kvmem_spec_opts & opts) {
    sess = kvmem_spec_session{};
    if (!model_tgt || !ctx_tgt) {
        fprintf(stderr, "kvmem_spec_start: missing target model/context\n");
        return false;
    }

    common_init();

    common_params p;
    p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    p.speculative.draft.n_max = opts.n_max;
    p.speculative.draft.n_min = opts.n_min;
    p.speculative.draft.p_min = opts.p_min;
    p.speculative.draft.n_gpu_layers = opts.n_gpu_layers;
    if (!opts.draft_model.empty()) {
        p.speculative.draft.mparams.path = opts.draft_model;
    }
    p.n_gpu_layers = opts.n_gpu_layers;
    p.n_ctx = opts.n_ctx > 0 ? opts.n_ctx : (int) llama_n_ctx(ctx_tgt);
    p.n_batch = opts.n_batch;
    p.n_ubatch = opts.n_ubatch > 0 ? opts.n_ubatch : opts.n_batch;
    p.n_parallel = 1;
    p.n_outputs_max = 1 + std::max(0, opts.n_max);
    p.n_outputs_max_per_seq = p.n_outputs_max;
    p.cache_type_k = opts.type_k;
    p.cache_type_v = opts.type_v;

    common_params p_dft = common_base_params_to_speculative(p);
    sess.init = common_speculative_init_from_params(p_dft, model_tgt, ctx_tgt);
    sess.ctx_dft = sess.init ? sess.init->context() : nullptr;
    if (!sess.ctx_dft) {
        fprintf(stderr,
                "kvmem_spec_start: MTP draft context is null "
                "(GGUF missing nextn / qwen35.nextn_predict_layers?)\n");
        return false;
    }

    sess.spec_params = std::move(p);
    sess.spec_params.speculative.draft.ctx_tgt = ctx_tgt;
    sess.spec_params.speculative.draft.ctx_dft = sess.ctx_dft;
    sess.spec = common_speculative_init(sess.spec_params.speculative, 1);
    if (!sess.spec) {
        fprintf(stderr, "kvmem_spec_start: common_speculative_init failed\n");
        return false;
    }

    // Prefer llama.cpp n_rs_seq GPU planes (qw3-style per-token GDN rollback).
    // Host PARTIAL_ONLY is only the fallback when planes are missing or the
    // draft is longer than n_rs_seq. Do not probe can_seq_rm on live KVMem:
    // hybrid seq_rm skips GDN on query-replay holes, which would look like PART.
    sess.n_rs_tgt = llama_n_rs_seq(ctx_tgt);
    sess.use_ckpt_dft = false;
    if (sess.n_rs_tgt > 0) {
        sess.use_ckpt_tgt = false;
        fprintf(stderr,
                "KVMEM_TRACE spec_ckpt tgt=RS n_rs_seq=%u (GPU GDN planes; host ckpt if draft > n_rs)\n",
                sess.n_rs_tgt);
    } else if (llama_kvmem_has_recurrent()) {
        sess.use_ckpt_tgt = true;
        fprintf(stderr,
                "KVMEM_TRACE spec_ckpt tgt=PARTIAL_ONLY (hybrid GDN; n_rs_seq=0 host fallback)\n");
    } else if (!opts.kvmem_enabled) {
        sess.use_ckpt_tgt =
                common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        sess.use_ckpt_dft =
                common_context_can_seq_rm(sess.ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        fprintf(stderr, "KVMEM_TRACE spec_ckpt tgt=%d dft=%d (vanilla probe)\n",
                (int) sess.use_ckpt_tgt, (int) sess.use_ckpt_dft);
    } else {
        sess.use_ckpt_tgt = false;
        fprintf(stderr, "KVMEM_TRACE spec_ckpt tgt=0 (dense KVMem seq_rm)\n");
    }

    sess.ok = true;
    fprintf(stderr, "KVMEM_TRACE spec_start type=draft-mtp n_max=%d p_min=%.3f dft=%p\n",
            opts.n_max, opts.p_min, (void *) sess.ctx_dft);
    return true;
}

void kvmem_spec_stop(kvmem_spec_session & sess) {
    if (sess.spec) {
        common_speculative_free(sess.spec);
        sess.spec = nullptr;
    }
    sess.init.reset();
    sess.ctx_dft = nullptr;
    sess.ok = false;
}

int kvmem_spec_decode_span(llama_context * ctx,
                           common_speculative * spec,
                           const llama_token * toks,
                           int pos0, int pos1, int n_batch,
                           const char * what) {
    if (pos0 >= pos1) {
        return 0;
    }
    if (n_batch <= 0) {
        n_batch = 512;
    }
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int n_pos = pos0;
    while (n_pos < pos1) {
        const int n = std::min(n_batch, pos1 - n_pos);
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) {
            common_batch_add(batch, toks[n_pos + i], n_pos + i, { 0 }, false);
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "llama_decode(%s) failed rc=%d at pos=%d n=%d\n",
                    what, rc, n_pos, n);
            llama_batch_free(batch);
            return rc;
        }
        if (spec && !common_speculative_process(spec, batch)) {
            fprintf(stderr, "common_speculative_process(%s) failed at pos=%d n=%d\n",
                    what, n_pos, n);
            llama_batch_free(batch);
            return 1;
        }
        n_pos += n;
    }
    llama_batch_free(batch);
    return 0;
}

kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        float temp,
        kvmem_spec_on_token on_token) {
    kvmem_spec_gen_stats st;
    if (!sess.ok || !sess.spec || prompt.empty() || n_predict <= 0) {
        st.failed = true;
        return st;
    }

    llama_kvmem_end_prefill_capture();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    llama_context * ctx_dft = sess.ctx_dft;
    common_speculative * spec = sess.spec;
    const llama_seq_id seq_id = 0;

    common_params_sampling sparams;
    if (temp <= 0.0f) {
        sparams.temp = 0.0f;
        sparams.min_p = 0.0f;
        sparams.top_p = 1.0f;
        sparams.top_k = 0;
        sparams.penalty_repeat = 1.0f;
        sparams.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    } else {
        sparams.temp = temp;
    }
    common_sampler_ptr smpl(common_sampler_init(model_tgt, sparams));

    llama_tokens prompt_tgt(prompt.begin(), prompt.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));
    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_token id_last = prompt.back();
    int n_past = (int) prompt.size() - 1;
    llama_batch batch_tgt = llama_batch_init((int) llama_n_batch(ctx_tgt), 0, 1);
    llama_tokens draft;
    common_prompt_checkpoint ckpt;
    bool has_eos = false;

    while (st.n_gen < n_predict && !has_eos) {
        if (draft.empty()) {
            llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
            ckpt.update_pos(
                    (int64_t) prompt_tgt.size(),
                    mem_tgt ? llama_memory_seq_pos_min(mem_tgt, seq_id) : 0,
                    mem_tgt ? llama_memory_seq_pos_max(mem_tgt, seq_id) : 0);

            if (sess.use_ckpt_dft && ctx_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            n_draft_max = std::min(n_draft_max, n_predict - st.n_gen - 1);
            n_draft_max = std::max(n_draft_max, 0);

            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .n_past     = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft,
            };
            common_speculative_draft(spec);

            const bool host_ckpt = sess.use_ckpt_tgt
                    || (sess.n_rs_tgt > 0 && draft.size() > sess.n_rs_tgt);
            if (!draft.empty() && host_ckpt) {
                ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            if (ctx_dft) {
                if (sess.use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                if (mem_dft) {
                    llama_memory_seq_rm(mem_dft, seq_id, ckpt.pos_max + 1, -1);
                }
            }
        }

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + (llama_pos) i, { seq_id }, true);
        }

        const int rc = llama_decode(ctx_tgt, batch_tgt);
        if (rc != 0) {
            fprintf(stderr, "llama_decode(spec verify) failed rc=%d n_draft=%zu\n",
                    rc, draft.size());
            st.failed = true;
            break;
        }
        if (!common_speculative_process(spec, batch_tgt)) {
            fprintf(stderr, "common_speculative_process(verify) failed\n");
            st.failed = true;
            break;
        }

        const size_t n_draft = draft.size();
        const bool host_ckpt = sess.use_ckpt_tgt
                || (sess.n_rs_tgt > 0 && n_draft > sess.n_rs_tgt);
        common_sampler_ptr smpl_save;
        if (host_ckpt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        const bool restore = host_ckpt && ids.size() - 1 < n_draft;
        fprintf(stderr,
                "KVMEM_TRACE spec_verify n_draft=%zu n_accept=%zu restore=%d pos=%d ckpt_bytes=%zu n_rs=%u\n",
                n_draft, ids.size() > 0 ? ids.size() - 1 : 0, (int) restore, n_past - 1,
                ckpt.data_tgt.size(), sess.n_rs_tgt);

        if (restore) {
            ++st.n_restore;
            draft = std::move(ids);
            ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
            if (mem_tgt) {
                llama_memory_seq_rm(mem_tgt, seq_id, ckpt.pos_max + 1, -1);
            }
            if (ctx_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                if (mem_dft) {
                    llama_memory_seq_rm(mem_dft, seq_id, ckpt.pos_max + 1, -1);
                }
            }
            prompt_tgt.resize((size_t) ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        common_speculative_accept(spec, seq_id, (uint16_t) (ids.size() - 1));
        n_past += (int) ids.size() - 1;
        st.n_drafted += (int) n_draft;
        st.n_accept += (int) ids.size() - 1;

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }
            const std::string piece = common_token_to_piece(ctx_tgt, id_last);
            if (on_token) {
                on_token(id_last, piece, i + 1 < ids.size());
            }
            ++st.n_gen;
            if (st.n_gen >= n_predict) {
                break;
            }
        }

        draft.clear();
        {
            llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
            if (mem_tgt) {
                llama_memory_seq_rm(mem_tgt, seq_id, n_past, -1);
            }
            if (ctx_dft) {
                llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                if (mem_dft) {
                    llama_memory_seq_rm(mem_dft, seq_id, n_past, -1);
                }
            }
        }
    }

    fprintf(stderr,
            "KVMEM_TRACE spec_stats n_gen=%d n_drafted=%d n_accept=%d n_restore=%d accept_pct=%.1f\n",
            st.n_gen, st.n_drafted, st.n_accept, st.n_restore,
            st.n_drafted > 0 ? 100.0 * st.n_accept / st.n_drafted : 0.0);
    common_speculative_print_stats(spec);

    llama_batch_free(batch_tgt);
    return st;
}
