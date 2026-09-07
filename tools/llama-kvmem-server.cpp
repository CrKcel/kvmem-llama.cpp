#include "llama.h"
#include "llama-kvmem-hooks.h"
#include "kvmem-spec.h"

#include "chat.h"
#include "common.h"
#include "json.h"
#include "sampling.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

static bool eq(const char * a, const char * b) {
    return std::strcmp(a, b) == 0;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m model.gguf [options]\n"
            "\n"
            "  Independent single-slot OpenAI-compatible server. Does not patch llama-server.\n"
            "\n"
            "  -m, --model PATH           GGUF path\n"
            "  --host HOST                bind address (default 127.0.0.1)\n"
            "  --port N                   port (default 8080)\n"
            "  -c, --ctx-size N           context size (default 2048)\n"
            "  -n, --n-predict N          default max_tokens (default 128)\n"
            "  -b, --batch-size N         logical batch (default 512)\n"
            "  -ngl, --n-gpu-layers N     GPU layers (default 99)\n"
            "  --kvmem / --no-kvmem       enable KVMem (default on)\n"
            "  --kvmem-budget N           GPU working-set tokens; 0 = n_ctx\n"
            "  --kvmem-block-tokens N     block size (default 32)\n"
            "  --kvmem-gen-reserve N      decode slack (default 256)\n"
            "  --kvmem-method NAME        recency | retrieval (default retrieval)\n"
            "  --kvmem-query-last N       fallback query-last if last-user span missing (default 64)\n"
            "  --kvmem-gpu-ratio R        cap slot pool at this fraction of GPU VRAM (default 0.50)\n"
            "  --kvmem-cpu-gb GB          CPU spill arena in GiB (0 = off)\n"
            "  --kvmem-nvme-gb GB         NVMe file in GiB (0 = off)\n"
            "  --kvmem-nvme-dir PATH      NVMe directory (default /tmp/kvmem_nvme)\n"
            "  --kvmem-harvest-v          prefill D2H V with raw-K (default off; RAM until NVMe flush)\n"
            "  --kvmem-raw-k-nvme         store raw-K and V on NVMe (needs --kvmem-nvme-gb)\n"
            "  --kv-dtype NAME            GPU KV cache type for K and V: f16 | q8_0 | q4_0 (default q8_0)\n"
            "  -ctk, --cache-type-k TYPE  GPU K cache type (llama.cpp name; default q8_0)\n"
            "  -ctv, --cache-type-v TYPE  GPU V cache type (must match K when quantized)\n"
            "  --spec-type TYPE           none | draft-mtp (default none)\n"
            "  --spec-draft-n-max N       MTP draft tokens (default 2)\n"
            "  --spec-draft-p-min P       min draft probability (default 0)\n",
            argv0);
}

static std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text, bool add_special) {
    const int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> out;
    if (n <= 0) {
        return out;
    }
    out.resize((size_t) n);
    llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), n, add_special, true);
    return out;
}

static std::string token_piece(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n <= 0) {
        return {};
    }
    return std::string(buf, (size_t) n);
}

static int force_pos_from_substr(const llama_vocab * vocab, const std::vector<llama_token> & toks,
                                 const std::string & needle) {
    if (needle.empty()) {
        return -1;
    }
    std::string acc;
    for (int i = 0; i < (int) toks.size(); ++i) {
        acc += token_piece(vocab, toks[(size_t) i]);
        if (acc.find(needle) != std::string::npos) {
            return i;
        }
    }
    return -1;
}

struct ServerState {
    std::mutex mu;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr tmpls;
    llama_kvmem_params kparams {};
    int n_batch = 512;
    int n_predict_default = 128;
    int query_last_fallback = 64;
    std::string model_name = "kvmem";
    kvmem_spec_session spec;
    ggml_type cache_type_k = GGML_TYPE_Q8_0;
    ggml_type cache_type_v = GGML_TYPE_Q8_0;
    bool spec_mtp = false;
    int spec_n_max = 2;
    float spec_p_min = 0.0f;
    std::vector<llama_token> cached_tokens;
    int perf_p_eval = 0;
    std::vector<uint8_t> gdn_ckpt;
    int gdn_ckpt_pos = -1; // last pos included in gdn_ckpt
};

static int decode_span(llama_context * ctx, const llama_token * toks, int pos0, int pos1, int n_batch, const char * what);
static int decode_span_maybe_spec(ServerState & st, const llama_token * toks, int pos0, int pos1, const char * what);

static bool gdn_sync_to(ServerState & st, const std::vector<llama_token> & prompt, int n_past) {
    if (!llama_kvmem_has_recurrent()) {
        return true;
    }
    const llama_pos want = n_past - 1;
    llama_pos rmax = llama_kvmem_recr_pos_max();
    if (rmax == want) {
        return true;
    }
    if (st.gdn_ckpt.empty() || st.gdn_ckpt_pos < 0 || st.gdn_ckpt_pos > want) {
        fprintf(stderr, "KVMEM_TRACE gdn_sync fail rmax=%d want=%d ckpt_pos=%d\n",
                (int) rmax, (int) want, st.gdn_ckpt_pos);
        return false;
    }
    const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    if (llama_state_seq_set_data_ext(st.ctx, st.gdn_ckpt.data(), st.gdn_ckpt.size(), 0, fl)
            != st.gdn_ckpt.size()) {
        fprintf(stderr, "GDN catch-up restore failed\n");
        return false;
    }
    const int from = st.gdn_ckpt_pos + 1;
    if (from < n_past) {
        // Trunk only. MTP draft still holds [from, n_past) after seq_rm(n_past,-1);
        // spec_decode_span would llama_decode(ctx_dft) at Y=from while X=n_past-1
        // and M-RoPE rejects X < Y.
        llama_kvmem_set_replay(true);
        const int rc = decode_span(st.ctx, prompt.data(), from, n_past, st.n_batch, "gdn-catchup");
        llama_kvmem_set_replay(false);
        if (rc != 0) {
            return false;
        }
    }
    rmax = llama_kvmem_recr_pos_max();
    fprintf(stderr, "KVMEM_TRACE gdn_sync ckpt_pos=%d from=%d n_past=%d rmax=%d\n",
            st.gdn_ckpt_pos, from, n_past, (int) rmax);
    return rmax == want;
}

static int common_token_prefix(const std::vector<llama_token> & a,
                               const std::vector<llama_token> & b) {
    const int n = (int) std::min(a.size(), b.size());
    int i = 0;
    while (i < n && a[(size_t) i] == b[(size_t) i]) {
        i++;
    }
    return i;
}

static void memory_clear_all(ServerState & st) {
    llama_memory_t mem = llama_get_memory(st.ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
    if (st.spec.ctx_dft) {
        llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
        if (md) {
            llama_memory_clear(md, true);
        }
    }
    st.cached_tokens.clear();
    st.gdn_ckpt.clear();
    st.gdn_ckpt_pos = -1;
}

static void commit_cached(ServerState & st, const std::vector<llama_token> & prompt,
                          const std::vector<llama_token> & gen) {
    st.cached_tokens = prompt;
    st.cached_tokens.insert(st.cached_tokens.end(), gen.begin(), gen.end());
    fprintf(stderr, "KVMEM_TRACE cache_commit n_prompt=%d n_gen=%d n_cached=%d stored=%u\n",
            (int) prompt.size(), (int) gen.size(), (int) st.cached_tokens.size(),
            llama_kvmem_store_n_tokens());
}

static int decode_span(llama_context * ctx, const llama_token * toks, int pos0, int pos1, int n_batch, const char * what) {
    if (pos0 >= pos1) {
        return 0;
    }
    if (n_batch <= 0) {
        n_batch = 512;
    }
    // Explicit pos: T5 query sits in the middle of the prompt (last user, then
    // assistant tool XML + role=tool). llama_batch_get_one would append at
    // seq_pos_max+1 and miss the hole after seq_rm(q0,q1).
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int n_pos = pos0;
    while (n_pos < pos1) {
        const int n = std::min(n_batch, pos1 - n_pos);
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) {
            common_batch_add(batch, toks[n_pos + i], n_pos + i, { 0 }, i == n - 1);
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "llama_decode(%s) failed rc=%d at pos=%d n=%d\n", what, rc, n_pos, n);
            llama_batch_free(batch);
            return rc;
        }
        n_pos += n;
    }
    llama_batch_free(batch);
    return 0;
}

static int decode_span_maybe_spec(ServerState & st, const llama_token * toks, int pos0, int pos1, const char * what) {
    if (st.spec.ok) {
        return kvmem_spec_decode_span(st.ctx, st.spec.spec, toks, pos0, pos1, st.n_batch, what);
    }
    return decode_span(st.ctx, toks, pos0, pos1, st.n_batch, what);
}

static bool run_prefill_retrieval(ServerState & st, const std::vector<llama_token> & prompt) {
    const int n_prompt = (int) prompt.size();
    llama_context * ctx = st.ctx;
    const int eval_end = st.spec.ok ? n_prompt - 1 : n_prompt;

    int n_past = 0;
    bool reused = false;
    if (!st.cached_tokens.empty() && n_prompt > 1) {
        const int lcp = common_token_prefix(st.cached_tokens, prompt);
        const uint32_t stored = st.kparams.enabled ? llama_kvmem_store_n_tokens()
                                                   : (uint32_t) st.cached_tokens.size();
        fprintf(stderr,
                "KVMEM_TRACE prefix_try lcp=%d n_cached=%d stored=%u n_prompt=%d\n",
                lcp, (int) st.cached_tokens.size(), stored, n_prompt);
        reused = lcp > 0 && lcp < n_prompt && stored >= (uint32_t) lcp;
        if (reused) {
            n_past = lcp;
            if (st.kparams.enabled) {
                llama_kvmem_begin_cached_turn();
            }
            llama_memory_t mem = llama_get_memory(ctx);
            if (mem) {
                llama_memory_seq_rm(mem, 0, n_past, -1);
            }
            if (st.spec.ctx_dft) {
                llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
                if (md) {
                    llama_memory_seq_rm(md, 0, n_past, -1);
                }
            }
            if (st.kparams.enabled) {
                llama_kvmem_truncate_cached((uint32_t) n_past);
            }
            if (!gdn_sync_to(st, prompt, n_past)) {
                reused = false;
            }
        }
    }
    if (!reused) {
        memory_clear_all(st);
        n_past = 0;
    }

    const bool do_retr = st.kparams.enabled && st.kparams.method == 1 && st.kparams.query_begin > 0;
    const bool recr_ckpt = do_retr && llama_kvmem_has_recurrent();
    int q0 = st.kparams.query_begin;
    int q1 = st.kparams.query_end;
    if (q0 < 0) {
        q0 = 0;
    }
    if (q1 <= q0 || q1 > eval_end) {
        q1 = eval_end;
    }
    fprintf(stderr,
            "KVMEM_TRACE prefix_reuse reused=%d n_past=%d n_prompt=%d n_cached=%d "
            "n_new=%d stored=%u query=[%d,%d)\n",
            (int) reused, n_past, n_prompt, (int) st.cached_tokens.size(),
            eval_end - n_past, llama_kvmem_store_n_tokens(),
            q0, q1);

    auto dec = [&](int a, int b, const char * what) -> bool {
        if (a < 0) {
            a = 0;
        }
        if (b > eval_end) {
            b = eval_end;
        }
        if (a >= b) {
            return true;
        }
        return decode_span_maybe_spec(st, prompt.data(), a, b, what) == 0;
    };
    // Hole-fill / recapture of positions that may already sit in KV. Trunk
    // only: MTP draft is M-RoPE and cannot decode Y while X (seq_pos_max)
    // is still ahead of Y.
    auto replay = [&](int a, int b, const char * what) -> bool {
        if (a < 0) {
            a = 0;
        }
        if (b > eval_end) {
            b = eval_end;
        }
        if (a >= b) {
            return true;
        }
        llama_kvmem_set_replay(true);
        const int rc = decode_span(st.ctx, prompt.data(), a, b, st.n_batch, what);
        llama_kvmem_set_replay(false);
        return rc == 0;
    };

    if (!do_retr) {
        if (!dec(n_past, eval_end, reused ? "prefill-suffix" : "prefill")) {
            return false;
        }
        llama_synchronize(ctx);
        const llama_perf_context_data p = llama_perf_context(ctx);
        const int d = p.n_p_eval - st.perf_p_eval;
        st.perf_p_eval = p.n_p_eval;
        fprintf(stderr, "KVMEM_TRACE prefix_prefill n_p_eval=%d reused=%d n_past=%d n_new=%d\n",
                d, (int) reused, n_past, eval_end - n_past);
        return true;
    }

    llama_kvmem_reset_query();
    if (!dec(n_past, q0, reused ? "prefill-suffix" : "prefill")) {
        return false;
    }
    std::vector<uint8_t> gdn_ckpt;
    if (recr_ckpt) {
        if (n_past > q0 && !gdn_sync_to(st, prompt, q0)) {
            fprintf(stderr, "KVMEM_TRACE gdn_sync to query_begin failed\n");
            return false;
        }
        llama_synchronize(ctx);
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, fl);
        if (sz == 0) {
            fprintf(stderr, "GDN checkpoint size 0\n");
            return false;
        }
        gdn_ckpt.resize(sz);
        if (llama_state_seq_get_data_ext(ctx, gdn_ckpt.data(), sz, 0, fl) != sz) {
            fprintf(stderr, "GDN checkpoint copy failed\n");
            return false;
        }
        st.gdn_ckpt = gdn_ckpt;
        st.gdn_ckpt_pos = q0 > 0 ? q0 - 1 : -1;
        fprintf(stderr, "KVMEM_TRACE gdn_ckpt pos_end=%d bytes=%zu query_begin=%d ckpt_pos=%d\n",
                q0, sz, q0, st.gdn_ckpt_pos);
    }
    // Query may already sit inside the reused prefix (T5: last user, then
    // assistant tool XML + role=tool). Recapture Q over the cached part
    // (replay skips K/mean); prefill only the missing tail of the span.
    if (n_past > q0 && n_past < q1) {
        if (!replay(q0, n_past, "query-q-capture")) {
            return false;
        }
    }
    if (n_past < q1) {
        if (!dec(std::max(n_past, q0), q1, "prefill-query")) {
            return false;
        }
    } else if (!replay(q0, q1, "query-q-capture")) {
        return false;
    }
    fprintf(stderr, "KVMEM_TRACE query_q_capture n_past=%d query=[%d,%d) recapture=%d\n",
            n_past, q0, q1, (int) (n_past > q0));
    llama_synchronize(ctx);
    {
        const llama_perf_context_data p = llama_perf_context(ctx);
        const int d = p.n_p_eval - st.perf_p_eval;
        st.perf_p_eval = p.n_p_eval;
        fprintf(stderr, "KVMEM_TRACE prefix_prefill n_p_eval=%d reused=%d n_past=%d n_new=%d\n",
                d, (int) reused, n_past, eval_end - n_past);
    }

    llama_kvmem_apply_retrieval(ctx);
    if (recr_ckpt && !gdn_ckpt.empty()) {
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        if (llama_state_seq_set_data_ext(ctx, gdn_ckpt.data(), gdn_ckpt.size(), 0, fl) != gdn_ckpt.size()) {
            fprintf(stderr, "GDN restore failed\n");
            return false;
        }
        fprintf(stderr, "KVMEM_TRACE gdn_restore bytes=%zu\n", gdn_ckpt.size());
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        fprintf(stderr, "KVMEM_TRACE before_seq_rm seq_pos=[%d,%d] query=[%d,%d)\n",
                llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                q0, q1);
        llama_memory_seq_rm(mem, 0, q0, q1);
        fprintf(stderr, "KVMEM_TRACE after_seq_rm seq_pos=[%d,%d] auto_pos0=%d\n",
                llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                llama_memory_seq_pos_max(mem, 0) + 1);
    }
    if (st.spec.ctx_dft) {
        llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
        if (md) {
            llama_memory_seq_rm(md, 0, q0, q1);
            fprintf(stderr, "KVMEM_TRACE mtp_after_seq_rm seq_pos=[%d,%d]\n",
                    llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
        }
    }
    if (!replay(q0, q1, "query replay")) {
        return false;
    }
    llama_synchronize(ctx);
    fprintf(stderr, "KVMEM_TRACE query_replay begin=%d n=%d recr_ckpt=%d\n",
            q0, q1 - q0, (int) recr_ckpt);
    if (n_past > q1 && !replay(q1, n_past, "gdn-after-query")) {
        return false;
    }
    if (st.spec.ctx_dft) {
        // Query seq_rm left a hole in the draft cache. M-RoPE cannot fill it
        // while the suffix remains, so drop [q0, inf) and append in order.
        llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
        if (md) {
            llama_memory_seq_rm(md, 0, q0, -1);
        }
        const int d_end = std::max(n_past, q1);
        if (d_end > q0) {
            const int rc = decode_span(st.spec.ctx_dft, prompt.data(), q0, d_end, st.n_batch,
                                       "mtp-resync");
            if (rc != 0) {
                return false;
            }
            fprintf(stderr, "KVMEM_TRACE mtp_resync query=[%d,%d) to=%d\n", q0, q1, d_end);
        }
    }
    if (!dec(std::max(n_past, q1), eval_end, "prefill-tail")) {
        return false;
    }
    if (st.spec.ctx_dft) {
        llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
        if (md) {
            fprintf(stderr, "KVMEM_TRACE mtp_after_query_replay seq_pos=[%d,%d]\n",
                    llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
        }
    }
    return true;
}

static void derive_query_span(ServerState & st, const std::string & prompt, const std::string & last_user,
                              const std::vector<llama_token> & toks, int & qbegin, int & qend) {
    qbegin = -1;
    qend = (int) toks.size();
    if (!last_user.empty()) {
        const auto idx = prompt.rfind(last_user);
        if (idx != std::string::npos) {
            const std::string prefix = prompt.substr(0, idx);
            const std::string through = prompt.substr(0, idx + last_user.size());
            qbegin = (int) tokenize_text(st.vocab, prefix, true).size();
            qend = (int) tokenize_text(st.vocab, through, true).size();
        }
    }
    if (qend > (int) toks.size()) {
        qend = (int) toks.size();
    }
    if (qbegin < 0 || qend <= qbegin) {
        qend = (int) toks.size();
        const int last = std::min(st.query_last_fallback, qend);
        qbegin = qend > last ? qend - last : 0;
    }
    if (qbegin >= qend) {
        qbegin = 0;
    }
}

struct ChatRequest {
    std::vector<common_chat_msg> msgs;
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool parallel_tool_calls = false;
    bool parallel_tool_calls_set = false;
    std::string json_schema;
    std::string grammar;
    std::vector<std::string> stop;
    std::string last_user;
    int max_tokens = 128;
    float temperature = 0.0f;
    bool stream = false;
    int query_begin = -1;
    int query_end = -1;
    std::string force_substr;
    bool enable_thinking = false;
};

static common_json nlohmann_to_common(const json & j) {
    return common_json::parse(j.dump());
}

static const char * tool_choice_cstr(common_chat_tool_choice c) {
    switch (c) {
        case COMMON_CHAT_TOOL_CHOICE_NONE:     return "none";
        case COMMON_CHAT_TOOL_CHOICE_REQUIRED: return "required";
        default:                               return "auto";
    }
}

static const char * grammar_type_cstr(common_grammar_type t) {
    switch (t) {
        case COMMON_GRAMMAR_TYPE_TOOL_CALLS:    return "tool_calls";
        case COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT: return "output_format";
        case COMMON_GRAMMAR_TYPE_USER:          return "user";
        default:                                return "none";
    }
}

static common_params_sampling make_chat_sampling(
        const llama_vocab * vocab,
        float temp,
        const common_chat_params & chat,
        const ChatRequest & cr) {
    common_params_sampling sp;
    if (temp <= 0.0f) {
        sp.temp = 0.0f;
        sp.min_p = 0.0f;
        sp.top_p = 1.0f;
        sp.top_k = 0;
        sp.penalty_repeat = 1.0f;
        sp.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    } else {
        sp.temp = temp;
    }
    std::string g = chat.grammar.empty() ? cr.grammar : chat.grammar;
    if (!g.empty()) {
        common_grammar_type ty = COMMON_GRAMMAR_TYPE_USER;
        if (!cr.tools.empty() && cr.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            ty = COMMON_GRAMMAR_TYPE_TOOL_CALLS;
        } else if (!cr.json_schema.empty()) {
            ty = COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT;
        }
        sp.grammar = {ty, std::move(g)};
    }
    sp.grammar_lazy = chat.grammar_lazy;
    sp.generation_prompt = chat.generation_prompt;
    if (vocab) {
        for (const auto & t : chat.preserved_tokens) {
            const auto ids = common_tokenize(vocab, t, false, true);
            if (ids.size() == 1) {
                sp.preserved_tokens.insert(ids[0]);
            }
        }
        for (const auto & trigger : chat.grammar_triggers) {
            if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                const auto ids = common_tokenize(vocab, trigger.value, false, true);
                if (ids.size() == 1) {
                    common_grammar_trigger tr;
                    tr.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                    tr.value = trigger.value;
                    tr.token = ids[0];
                    sp.grammar_triggers.push_back(std::move(tr));
                    continue;
                }
            }
            sp.grammar_triggers.push_back(trigger);
        }
    } else {
        sp.grammar_triggers = chat.grammar_triggers;
    }
    return sp;
}

static common_chat_msg parse_assistant_output(
        const std::string & content,
        const common_chat_params & chat,
        bool parse_tools) {
    try {
        common_chat_parser_params pp(chat);
        pp.parse_tool_calls = parse_tools;
        if (!chat.parser.empty()) {
            pp.parser.load(chat.parser);
        }
        common_chat_msg msg = common_chat_parse(content, false, pp);
        if (msg.role.empty()) {
            msg.role = "assistant";
        }
        return msg;
    } catch (const std::exception & e) {
        fprintf(stderr, "KVMEM_TRACE chat_out_parse_fail %s\n", e.what());
        common_chat_msg msg;
        msg.role = "assistant";
        msg.content = content;
        return msg;
    }
}

static json message_to_nlohmann(const common_chat_msg & msg) {
    return json::parse(msg.to_json_oaicompat().dump());
}

static json chat_diff_to_delta(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"] = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({std::move(tool_call)});
    }
    return delta;
}

struct StreamChatOut {
    common_chat_parser_params pp;
    common_chat_msg prev;
    std::string acc;
    std::vector<std::string> tc_ids;
    int n_id = 0;
    int n_tc_delta = 0;

    StreamChatOut(const common_chat_params & chat, bool parse_tools) {
        pp = common_chat_parser_params(chat);
        pp.parse_tool_calls = parse_tools;
        if (!chat.parser.empty()) {
            pp.parser.load(chat.parser);
        }
    }

    std::vector<json> set_text(const std::string & text, bool partial) {
        acc = text;
        std::vector<json> chunks;
        try {
            common_chat_msg msg = common_chat_parse(acc, partial, pp);
            if (msg.empty() && partial) {
                return chunks;
            }
            if (msg.role.empty()) {
                msg.role = "assistant";
            }
            msg.set_tool_call_ids(tc_ids, [this]() {
                return std::string("call_") + std::to_string(++n_id);
            });
            const auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
            prev = std::move(msg);
            for (const auto & d : diffs) {
                json delta = chat_diff_to_delta(d);
                if (delta.empty()) {
                    continue;
                }
                if (d.tool_call_index != std::string::npos) {
                    n_tc_delta++;
                }
                chunks.push_back(std::move(delta));
            }
        } catch (const std::exception & e) {
            if (!partial) {
                fprintf(stderr, "KVMEM_TRACE chat_stream_parse_fail %s\n", e.what());
            }
        }
        return chunks;
    }

    const char * finish_reason(bool hit_limit) const {
        if (!prev.tool_calls.empty()) {
            return "tool_calls";
        }
        if (hit_limit) {
            return "length";
        }
        return "stop";
    }
};

static json stream_choice_chunk(const std::string & cid, const json & delta, const char * finish) {
    json choice = {
        {"index", 0},
        {"delta", delta},
        {"finish_reason", finish ? json(finish) : json(nullptr)},
    };
    return json{
        {"id", cid},
        {"object", "chat.completion.chunk"},
        {"choices", json::array({std::move(choice)})},
    };
}

static bool strip_stop(std::string & content, const std::vector<std::string> & stops) {
    for (const auto & s : stops) {
        if (s.empty() || content.size() < s.size()) {
            continue;
        }
        if (content.compare(content.size() - s.size(), s.size(), s) == 0) {
            content.resize(content.size() - s.size());
            return true;
        }
    }
    return false;
}

static bool parse_chat_request(const json & body, ChatRequest & out, std::string & err) {
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        err = "messages array required";
        return false;
    }
    try {
        out.msgs = common_chat_msgs_parse_oaicompat(nlohmann_to_common(body.at("messages")));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (out.msgs.empty()) {
        err = "messages array required";
        return false;
    }
    for (auto it = out.msgs.rbegin(); it != out.msgs.rend(); ++it) {
        if (it->role == "user") {
            out.last_user = it->content.empty() ? it->render_content() : it->content;
            break;
        }
    }
    if (body.contains("tools") && !body["tools"].is_null()) {
        try {
            out.tools = common_chat_tools_parse_oaicompat(nlohmann_to_common(body.at("tools")));
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    }
    if (body.contains("tool_choice") && !body["tool_choice"].is_null()) {
        const auto & tc = body.at("tool_choice");
        try {
            if (tc.is_string()) {
                out.tool_choice = common_chat_tool_choice_parse_oaicompat(tc.get<std::string>());
            } else if (tc.is_object()) {
                // OpenAI named-function form → required (template/grammar in T2).
                out.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            } else {
                err = "tool_choice must be a string or object";
                return false;
            }
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    }
    if (body.contains("parallel_tool_calls") && body["parallel_tool_calls"].is_boolean()) {
        out.parallel_tool_calls = body["parallel_tool_calls"].get<bool>();
        out.parallel_tool_calls_set = true;
    }
    if (body.contains("grammar") && body["grammar"].is_string()) {
        out.grammar = body["grammar"].get<std::string>();
    }
    if (body.contains("json_schema") && !body["json_schema"].is_null()) {
        out.json_schema = body["json_schema"].dump();
    }
    if (body.contains("response_format") && body["response_format"].is_object()) {
        const auto & rf = body["response_format"];
        const std::string rtype = rf.value("type", "");
        if (rtype == "json_object") {
            if (out.json_schema.empty()) {
                out.json_schema = rf.contains("schema") ? rf["schema"].dump() : "{}";
            }
        } else if (rtype == "json_schema") {
            const auto schema_wrapper = rf.value("json_schema", json::object());
            if (schema_wrapper.contains("schema")) {
                out.json_schema = schema_wrapper["schema"].dump();
            }
        } else if (!rtype.empty() && rtype != "text") {
            err = "response_format type must be text, json_object, or json_schema";
            return false;
        }
    }
    if (!out.tools.empty() && !out.grammar.empty()) {
        err = "Cannot use custom grammar constraints with tools.";
        return false;
    }
    if (body.contains("stop")) {
        const auto & stop = body.at("stop");
        if (stop.is_string()) {
            out.stop.push_back(stop.get<std::string>());
        } else if (stop.is_array()) {
            for (const auto & s : stop) {
                if (s.is_string()) {
                    out.stop.push_back(s.get<std::string>());
                }
            }
        }
    }
    if (body.contains("max_tokens") && body["max_tokens"].is_number()) {
        out.max_tokens = body["max_tokens"].get<int>();
    } else if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number()) {
        out.max_tokens = body["max_completion_tokens"].get<int>();
    }
    out.temperature = body.value("temperature", 0.0f);
    out.stream = body.value("stream", false);
    if (body.contains("kvmem") && body["kvmem"].is_object()) {
        const auto & k = body["kvmem"];
        if (k.contains("query_begin")) {
            out.query_begin = k["query_begin"].get<int>();
        }
        if (k.contains("query_end")) {
            out.query_end = k["query_end"].get<int>();
        }
        if (k.contains("force_substr") && k["force_substr"].is_string()) {
            out.force_substr = k["force_substr"].get<std::string>();
        }
        if (k.contains("pin")) {
            if (k["pin"].is_string()) {
                out.force_substr = k["pin"].get<std::string>();
            } else if (k["pin"].is_array() && !k["pin"].empty() && k["pin"][0].is_string()) {
                out.force_substr = k["pin"][0].get<std::string>();
            }
        }
        out.enable_thinking = k.value("enable_thinking", false);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string host = "127.0.0.1";
    std::string nvme_dir;
    int port = 8080;
    int n_ctx = 2048;
    int ngl = 99;
    ServerState st;
    st.kparams.block_tokens = 32;
    st.kparams.gen_reserve = 256;
    st.kparams.method = 1;
    st.kparams.enabled = true;
    st.kparams.query_begin = -1;
    st.kparams.query_end = -1;
    st.kparams.force_pos = -1;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                exit(1);
            }
            return argv[++i];
        };
        if (eq(arg, "-h") || eq(arg, "--help")) {
            print_usage(argv[0]);
            return 0;
        } else if (eq(arg, "-m") || eq(arg, "--model")) {
            model_path = need(arg);
        } else if (eq(arg, "--host")) {
            host = need(arg);
        } else if (eq(arg, "--port")) {
            port = std::atoi(need(arg));
        } else if (eq(arg, "-c") || eq(arg, "--ctx-size")) {
            n_ctx = std::atoi(need(arg));
        } else if (eq(arg, "-n") || eq(arg, "--n-predict")) {
            st.n_predict_default = std::atoi(need(arg));
        } else if (eq(arg, "-b") || eq(arg, "--batch-size")) {
            st.n_batch = std::atoi(need(arg));
        } else if (eq(arg, "-ngl") || eq(arg, "--n-gpu-layers")) {
            ngl = std::atoi(need(arg));
        } else if (eq(arg, "--kvmem")) {
            st.kparams.enabled = true;
        } else if (eq(arg, "--no-kvmem")) {
            st.kparams.enabled = false;
        } else if (eq(arg, "--kvmem-budget")) {
            st.kparams.budget = (uint32_t) std::atoi(need(arg));
        } else if (eq(arg, "--kvmem-block-tokens")) {
            st.kparams.block_tokens = (uint32_t) std::atoi(need(arg));
        } else if (eq(arg, "--kvmem-gen-reserve")) {
            st.kparams.gen_reserve = (uint32_t) std::atoi(need(arg));
        } else if (eq(arg, "--kvmem-method")) {
            const char * m = need(arg);
            st.kparams.method = (eq(m, "retrieval") || eq(m, "retrieve")) ? 1 : 0;
        } else if (eq(arg, "--kvmem-query-last")) {
            st.query_last_fallback = std::atoi(need(arg));
        } else if (eq(arg, "--kvmem-gpu-ratio")) {
            st.kparams.gpu_memory_ratio = std::strtof(need(arg), nullptr);
        } else if (eq(arg, "--kvmem-cpu-gb")) {
            const double gb = std::atof(need(arg));
            st.kparams.cpu_bytes = gb <= 0.0 ? 0
                : static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (eq(arg, "--kvmem-nvme-gb")) {
            const double gb = std::atof(need(arg));
            st.kparams.nvme_bytes = gb <= 0.0 ? 0
                : static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (eq(arg, "--kvmem-nvme-dir")) {
            nvme_dir = need(arg);
        } else if (eq(arg, "--kvmem-harvest-v")) {
            st.kparams.harvest_v = true;
        } else if (eq(arg, "--kvmem-raw-k-nvme")) {
            st.kparams.raw_k_nvme = true;
        } else if (eq(arg, "--kv-dtype") || eq(arg, "-ctk") || eq(arg, "--cache-type-k")
                   || eq(arg, "-ctv") || eq(arg, "--cache-type-v")) {
            bool ok = false;
            const ggml_type t = kvmem_parse_cache_type(need(arg), &ok);
            if (!ok) {
                fprintf(stderr, "unsupported cache type (want f16|q8_0|q4_0|f32)\n");
                return 1;
            }
            if (eq(arg, "-ctv") || eq(arg, "--cache-type-v")) {
                st.cache_type_v = t;
            } else if (eq(arg, "-ctk") || eq(arg, "--cache-type-k")) {
                st.cache_type_k = t;
            } else {
                st.cache_type_k = t;
                st.cache_type_v = t;
            }
        } else if (eq(arg, "--spec-type")) {
            const char * t = need(arg);
            if (eq(t, "draft-mtp")) {
                st.spec_mtp = true;
            } else if (eq(t, "none")) {
                st.spec_mtp = false;
            } else {
                fprintf(stderr, "unsupported --spec-type %s (P7-0: draft-mtp|none)\n", t);
                return 1;
            }
        } else if (eq(arg, "--spec-draft-n-max")) {
            st.spec_n_max = std::atoi(need(arg));
        } else if (eq(arg, "--spec-draft-p-min")) {
            st.spec_p_min = std::strtof(need(arg), nullptr);
        } else {
            fprintf(stderr, "unknown flag: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    {
        const auto slash = model_path.find_last_of("/\\");
        st.model_name = slash == std::string::npos ? model_path : model_path.substr(slash + 1);
    }

    setvbuf(stderr, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);

    common_init();
    llama_log_set([](enum ggml_log_level, const char * text, void *) {
        fputs(text, stderr);
        fflush(stderr);
    }, nullptr);
    ggml_backend_load_all();

    if (st.kparams.enabled) {
        if (!nvme_dir.empty()) {
            st.kparams.nvme_dir = nvme_dir.c_str();
        }
        llama_kvmem_set_params(&st.kparams);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    mparams.load_mtp = st.spec_mtp;
    st.model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!st.model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    st.vocab = llama_model_get_vocab(st.model);
    st.tmpls = common_chat_templates_init(st.model, "");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = (uint32_t) st.n_batch;
    cparams.n_ubatch = (uint32_t) st.n_batch;
    cparams.n_seq_max = 1;
    if (!kvmem_cache_types_ok(st.cache_type_k, st.cache_type_v)) {
        fprintf(stderr, "quantized K/V cache types must match (CUDA FA: q8_0/q8_0 or q4_0/q4_0)\n");
        return 1;
    }
    cparams.type_k = st.cache_type_k;
    cparams.type_v = st.cache_type_v;
    if (st.spec_mtp) {
        const uint32_t n_out = (uint32_t) (1 + std::max(0, st.spec_n_max));
        cparams.n_outputs_max = n_out;
        cparams.n_outputs_max_per_seq = n_out;
        cparams.n_rs_seq = (uint32_t) std::max(0, st.spec_n_max);
    }
    st.ctx = llama_init_from_model(st.model, cparams);
    if (!st.ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    if (st.spec_mtp) {
        kvmem_spec_opts sopts;
        sopts.n_max = st.spec_n_max;
        sopts.p_min = st.spec_p_min;
        sopts.n_gpu_layers = ngl;
        sopts.n_ctx = n_ctx;
        sopts.n_batch = st.n_batch;
        sopts.n_ubatch = st.n_batch;
        sopts.kvmem_enabled = st.kparams.enabled;
        sopts.type_k = st.cache_type_k;
        sopts.type_v = st.cache_type_v;
        if (!kvmem_spec_start(st.spec, st.model, st.ctx, sopts)) {
            return 1;
        }
    }

    httplib::Server svr;
    svr.set_read_timeout(1800, 0);
    svr.set_write_timeout(1800, 0);
    svr.set_idle_interval(0, 100000);
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
    });
    svr.Options(".*", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {
            {"object", "list"},
            {"data", json::array({json{{"id", st.model_name}, {"object", "model"}}})},
        };
        res.set_content(j.dump(), "application/json");
    });

    auto handle_chat = [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        ChatRequest cr;
        cr.max_tokens = st.n_predict_default;
        std::string err;
        if (!parse_chat_request(body, cr, err)) {
            res.status = 400;
            res.set_content(json{{"error", err}}.dump(), "application/json");
            return;
        }

        auto slot = std::make_shared<std::unique_lock<std::mutex>>(st.mu);

        common_chat_templates_inputs inputs;
        inputs.messages = cr.msgs;
        inputs.tools = cr.tools;
        inputs.tool_choice = cr.tool_choice;
        inputs.grammar = cr.grammar;
        inputs.json_schema = cr.json_schema;
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        inputs.enable_thinking = cr.enable_thinking;
        if (cr.parallel_tool_calls_set) {
            inputs.parallel_tool_calls = cr.parallel_tool_calls;
        } else {
            const auto caps = common_chat_templates_get_caps(st.tmpls.get());
            const auto it = caps.find("supports_parallel_tool_calls");
            inputs.parallel_tool_calls = it != caps.end() && it->second;
        }
        const common_chat_params formatted = common_chat_templates_apply(st.tmpls.get(), inputs);
        const std::string & prompt = formatted.prompt;
        auto toks = tokenize_text(st.vocab, prompt, true);
        if (toks.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"empty prompt\"}", "application/json");
            return;
        }
        if ((int) toks.size() + cr.max_tokens > (int) llama_n_ctx(st.ctx)) {
            res.status = 400;
            res.set_content("{\"error\":\"prompt + max_tokens exceeds n_ctx\"}", "application/json");
            return;
        }

        int qbegin = cr.query_begin;
        int qend = cr.query_end;
        if (qbegin < 0 || qend < 0) {
            derive_query_span(st, prompt, cr.last_user, toks, qbegin, qend);
        }
        const int force = force_pos_from_substr(st.vocab, toks, cr.force_substr);
        st.kparams.query_begin = qbegin;
        st.kparams.query_end = qend;
        st.kparams.force_pos = force;
        if (st.kparams.enabled) {
            llama_kvmem_set_request_span(qbegin, qend, force);
        }
        int n_tool_hist = 0;
        for (const auto & m : cr.msgs) {
            if (m.role == "tool" || !m.tool_calls.empty()) {
                n_tool_hist++;
            }
        }
        bool prompt_has_tool = false;
        for (const auto & t : cr.tools) {
            if (!t.name.empty() && prompt.find(t.name) != std::string::npos) {
                prompt_has_tool = true;
                break;
            }
        }
        fprintf(stderr, "KVMEM_TRACE n_prompt=%d query=[%d,%d) force_pos=%d last_user_chars=%zu\n",
                (int) toks.size(), qbegin, qend, force, cr.last_user.size());
        fprintf(stderr,
                "KVMEM_TRACE chat_parse n_msg=%zu n_tools=%zu tool_choice=%s tool_hist=%d "
                "prompt_has_tool=%d grammar_bytes=%zu\n",
                cr.msgs.size(), cr.tools.size(), tool_choice_cstr(cr.tool_choice),
                n_tool_hist, (int) prompt_has_tool, formatted.grammar.size());

        const auto t_turn0 = std::chrono::steady_clock::now();
        if (!run_prefill_retrieval(st, toks)) {
            res.status = 500;
            res.set_content("{\"error\":\"prefill/retrieval failed\"}", "application/json");
            return;
        }
        const auto t_pf1 = std::chrono::steady_clock::now();
        const double prefill_ms =
                std::chrono::duration<double, std::milli>(t_pf1 - t_turn0).count();
        fprintf(stderr, "KVMEM_CHAT_PREFILL ms=%.2f n_prompt=%d\n",
                prefill_ms, (int) toks.size());

        llama_kvmem_end_prefill_capture();

        auto emit_gen_wall = [t_turn0, t_pf1, prefill_ms, n_prompt = (int) toks.size()](int n_gen) {
            const auto now = std::chrono::steady_clock::now();
            const double gen_ms = std::chrono::duration<double, std::milli>(now - t_pf1).count();
            const double wall_ms = std::chrono::duration<double, std::milli>(now - t_turn0).count();
            const double tps = gen_ms > 0.0 ? 1000.0 * (double) n_gen / gen_ms : 0.0;
            fprintf(stderr, "KVMEM_GEN_WALL n=%d ms=%.2f toks=%.2f\n", n_gen, gen_ms, tps);
            fprintf(stderr,
                    "KVMEM_CHAT_TURN n_prompt=%d n_gen=%d prefill_ms=%.2f gen_ms=%.2f "
                    "wall_ms=%.2f gen_toks=%.2f\n",
                    n_prompt, n_gen, prefill_ms, gen_ms, wall_ms, tps);
        };

        const std::string cid = "chatcmpl-kvmem";
        llama_context * ctx = st.ctx;
        const llama_vocab * vocab = st.vocab;

        common_params_sampling sparams = make_chat_sampling(vocab, cr.temperature, formatted, cr);
        std::vector<std::string> stops = cr.stop;
        stops.insert(stops.end(), formatted.additional_stops.begin(), formatted.additional_stops.end());
        fprintf(stderr,
                "KVMEM_TRACE chat_sample grammar_type=%s lazy=%d n_trig=%zu gen_prompt_bytes=%zu\n",
                grammar_type_cstr(sparams.grammar.type), (int) sparams.grammar_lazy,
                sparams.grammar_triggers.size(), sparams.generation_prompt.size());
        const bool parse_tools = !cr.tools.empty() &&
                cr.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

        bool use_spec = st.spec.ok;
        if (use_spec && !sparams.grammar.empty()) {
            try {
                common_params_sampling probe = sparams;
                common_sampler_ptr test(common_sampler_init(st.model, probe));
                if (!test) {
                    use_spec = false;
                }
            } catch (const std::exception & e) {
                fprintf(stderr, "KVMEM_TRACE spec sampler init failed (%s); greedy fallback\n", e.what());
                use_spec = false;
            }
        }

        auto emit_json = [&](const std::string & content, int n_gen, bool hit_limit) {
            common_chat_msg msg = parse_assistant_output(content, formatted, parse_tools);
            std::vector<std::string> tc_ids;
            int n_id = 0;
            msg.set_tool_call_ids(tc_ids, [&n_id]() {
                return std::string("call_") + std::to_string(++n_id);
            });
            std::string finish = "stop";
            if (!msg.tool_calls.empty()) {
                finish = "tool_calls";
            } else if (hit_limit) {
                finish = "length";
            }
            json message;
            try {
                message = message_to_nlohmann(msg);
            } catch (const std::exception &) {
                message = json{{"role", "assistant"}, {"content", content}};
            }
            fprintf(stderr, "KVMEM_TRACE chat_out n_tool_calls=%zu finish=%s content_chars=%zu\n",
                    msg.tool_calls.size(), finish.c_str(),
                    msg.content.size());
            json out = {
                {"id", cid},
                {"object", "chat.completion"},
                {"model", st.model_name},
                {"choices", json::array({json{
                    {"index", 0},
                    {"message", message},
                    {"finish_reason", finish},
                }})},
                {"usage", json{
                    {"prompt_tokens", (int) toks.size()},
                    {"completion_tokens", n_gen},
                    {"total_tokens", (int) toks.size() + n_gen},
                }},
            };
            res.set_content(out.dump(), "application/json");
        };

        if (use_spec) {
            if (cr.stream) {
                const int max_tokens = cr.max_tokens;
                res.set_header("Cache-Control", "no-cache");
                res.set_chunked_content_provider("text/event-stream",
                    [slot, &st, toks, cid, max_tokens, sparams, parse_tools, formatted, emit_gen_wall](size_t, httplib::DataSink & sink) mutable {
                        auto send = [&](const std::string & payload) {
                            const std::string line = "data: " + payload + "\n\n";
                            sink.write(line.data(), line.size());
                        };
                        send(stream_choice_chunk(cid, json{{"role", "assistant"}}, nullptr).dump());
                        std::vector<llama_token> gen;
                        std::string content;
                        StreamChatOut sco(formatted, parse_tools);
                        kvmem_spec_generate(st.ctx, st.model, st.spec, toks, max_tokens, sparams,
                            [&](llama_token id, const std::string & piece, bool) {
                                gen.push_back(id);
                                content += piece;
                                if (parse_tools) {
                                    for (const auto & delta : sco.set_text(content, true)) {
                                        send(stream_choice_chunk(cid, delta, nullptr).dump());
                                    }
                                } else {
                                    send(stream_choice_chunk(cid, json{{"content", piece}}, nullptr).dump());
                                }
                            });
                        const bool hit_limit = (int) gen.size() >= max_tokens;
                        if (parse_tools) {
                            for (const auto & delta : sco.set_text(content, false)) {
                                send(stream_choice_chunk(cid, delta, nullptr).dump());
                            }
                        }
                        const char * finish = parse_tools ? sco.finish_reason(hit_limit) : "stop";
                        fprintf(stderr, "KVMEM_TRACE chat_stream n_tc_delta=%d finish=%s\n",
                                sco.n_tc_delta, finish);
                        emit_gen_wall((int) gen.size());
                        commit_cached(st, toks, gen);
                        send(stream_choice_chunk(cid, json::object(), finish).dump());
                        sink.write("data: [DONE]\n\n", 15);
                        slot->unlock();
                        sink.done();
                        return true;
                    });
                return;
            }
            std::string content;
            std::vector<llama_token> gen;
            const kvmem_spec_gen_stats gst = kvmem_spec_generate(
                    ctx, st.model, st.spec, toks, cr.max_tokens, sparams,
                    [&](llama_token id, const std::string & piece, bool) {
                        gen.push_back(id);
                        content += piece;
                    });
            if (gst.failed) {
                res.status = 500;
                res.set_content("{\"error\":\"speculative decode failed\"}", "application/json");
                return;
            }
            emit_gen_wall((int) gen.size());
            commit_cached(st, toks, gen);
            emit_json(content, (int) gen.size(), (int) gen.size() >= cr.max_tokens);
            return;
        }

        common_sampler * smpl = nullptr;
        try {
            common_params_sampling sp = sparams;
            smpl = common_sampler_init(st.model, sp);
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(json{{"error", std::string("sampler init failed: ") + e.what()}}.dump(),
                            "application/json");
            return;
        }
        if (!smpl) {
            res.status = 500;
            res.set_content("{\"error\":\"sampler init failed\"}", "application/json");
            return;
        }

        auto gen_one = [ctx, smpl, vocab](std::string & piece, bool & stopped, llama_token & id_out) -> bool {
            llama_token id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, id, true);
            if (llama_vocab_is_eog(vocab, id)) {
                stopped = true;
                return true;
            }
            id_out = id;
            piece = token_piece(vocab, id);
            llama_batch batch = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "llama_decode(gen) failed\n");
                return false;
            }
            return true;
        };

        if (cr.stream) {
            const int max_tokens = cr.max_tokens;
            res.set_header("Cache-Control", "no-cache");
            res.set_chunked_content_provider("text/event-stream",
                [slot, smpl, cid, gen_one, max_tokens, &st, toks, stops, parse_tools, formatted, emit_gen_wall](size_t, httplib::DataSink & sink) mutable {
                    auto send = [&](const std::string & payload) {
                        const std::string line = "data: " + payload + "\n\n";
                        sink.write(line.data(), line.size());
                    };
                    send(stream_choice_chunk(cid, json{{"role", "assistant"}}, nullptr).dump());
                    std::vector<llama_token> gen;
                    std::string content;
                    bool stopped = false;
                    bool hit_stop = false;
                    StreamChatOut sco(formatted, parse_tools);
                    while ((int) gen.size() < max_tokens && !stopped) {
                        std::string piece;
                        llama_token id = 0;
                        if (!gen_one(piece, stopped, id)) {
                            break;
                        }
                        if (stopped) {
                            break;
                        }
                        content += piece;
                        gen.push_back(id);
                        hit_stop = strip_stop(content, stops);
                        if (parse_tools) {
                            for (const auto & delta : sco.set_text(content, !hit_stop)) {
                                send(stream_choice_chunk(cid, delta, nullptr).dump());
                            }
                        } else {
                            send(stream_choice_chunk(cid, json{{"content", piece}}, nullptr).dump());
                        }
                        if (hit_stop) {
                            break;
                        }
                    }
                    const bool hit_limit = !stopped && !hit_stop && (int) gen.size() >= max_tokens;
                    if (parse_tools) {
                        for (const auto & delta : sco.set_text(content, false)) {
                            send(stream_choice_chunk(cid, delta, nullptr).dump());
                        }
                    }
                    const char * finish = parse_tools ? sco.finish_reason(hit_limit) : "stop";
                    fprintf(stderr, "KVMEM_TRACE chat_stream n_tc_delta=%d finish=%s\n",
                            sco.n_tc_delta, finish);
                    send(stream_choice_chunk(cid, json::object(), finish).dump());
                    sink.write("data: [DONE]\n\n", 15);
                    llama_kvmem_decode_mean_flush();
                    emit_gen_wall((int) gen.size());
                    commit_cached(st, toks, gen);
                    common_sampler_free(smpl);
                    slot->unlock();
                    sink.done();
                    return true;
                });
            return;
        }

        std::string content;
        std::vector<llama_token> gen;
        bool stopped = false;
        while ((int) gen.size() < cr.max_tokens && !stopped) {
            std::string piece;
            llama_token id = 0;
            if (!gen_one(piece, stopped, id)) {
                common_sampler_free(smpl);
                res.status = 500;
                res.set_content("{\"error\":\"decode failed\"}", "application/json");
                return;
            }
            if (stopped) {
                break;
            }
            content += piece;
            gen.push_back(id);
            if (strip_stop(content, stops)) {
                break;
            }
        }
        llama_kvmem_decode_mean_flush();
        emit_gen_wall((int) gen.size());
        commit_cached(st, toks, gen);
        common_sampler_free(smpl);
        emit_json(content, (int) gen.size(), !stopped && (int) gen.size() >= cr.max_tokens);
    };

    svr.Post("/v1/chat/completions", handle_chat);
    svr.Post("/chat/completions", handle_chat);

    fprintf(stderr, "llama-kvmem-server listening on http://%s:%d  model=%s kvmem=%d method=%s n_ctx=%d spec=%s n_max=%d\n",
            host.c_str(), port, st.model_name.c_str(), (int) st.kparams.enabled,
            st.kparams.method == 1 ? "retrieval" : "recency", n_ctx,
            st.spec.ok ? "draft-mtp" : "off", st.spec_n_max);
    if (!svr.listen(host, port)) {
        fprintf(stderr, "listen failed\n");
        return 1;
    }
    kvmem_spec_stop(st.spec);
    llama_free(st.ctx);
    llama_model_free(st.model);
    return 0;
}
