#include "llama.h"
#include "llama-kvmem-hooks.h"

#include "chat.h"
#include "common.h"

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
            "  --kvmem-query-last N       fallback query-last if last-user span missing (default 64)\n",
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

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += (char) c;
                }
        }
    }
    return o;
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
};

static int decode_span(llama_context * ctx, const llama_token * toks, int pos0, int pos1, int n_batch, const char * what) {
    int n_pos = pos0;
    while (n_pos < pos1) {
        const int n = std::min(n_batch, pos1 - n_pos);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(toks + n_pos), n);
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "llama_decode(%s) failed rc=%d at pos=%d n=%d\n", what, rc, n_pos, n);
            return rc;
        }
        n_pos += n;
    }
    return 0;
}

static bool run_prefill_retrieval(ServerState & st, const std::vector<llama_token> & prompt) {
    const int n_prompt = (int) prompt.size();
    const int n_batch = st.n_batch;
    llama_context * ctx = st.ctx;

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    const bool do_retr = st.kparams.enabled && st.kparams.method == 1 && st.kparams.query_begin > 0;
    const bool recr_ckpt = do_retr && llama_kvmem_has_recurrent();
    const int prefix_end = recr_ckpt ? st.kparams.query_begin : n_prompt;

    if (decode_span(ctx, prompt.data(), 0, prefix_end, n_batch, "prefill") != 0) {
        return false;
    }

    std::vector<uint8_t> gdn_ckpt;
    if (recr_ckpt) {
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
        fprintf(stderr, "KVMEM_TRACE gdn_ckpt pos_end=%d bytes=%zu query_begin=%d\n",
                prefix_end, sz, st.kparams.query_begin);
        if (decode_span(ctx, prompt.data(), prefix_end, n_prompt, n_batch, "prefill-query") != 0) {
            return false;
        }
    }
    llama_synchronize(ctx);

    if (!do_retr) {
        return true;
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
    mem = llama_get_memory(ctx);
    llama_kvmem_set_replay(true);
    if (mem) {
        fprintf(stderr, "KVMEM_TRACE before_seq_rm seq_pos=[%d,%d] query=[%d,%d)\n",
                llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                st.kparams.query_begin, st.kparams.query_end);
        llama_memory_seq_rm(mem, 0, st.kparams.query_begin, st.kparams.query_end);
        fprintf(stderr, "KVMEM_TRACE after_seq_rm seq_pos=[%d,%d] auto_pos0=%d\n",
                llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                llama_memory_seq_pos_max(mem, 0) + 1);
    }
    const int q0 = st.kparams.query_begin;
    const int qn = n_prompt - q0;
    if (qn > 0) {
        llama_batch qbatch = llama_batch_get_one(const_cast<llama_token *>(prompt.data() + q0), qn);
        if (llama_decode(ctx, qbatch) != 0) {
            fprintf(stderr, "llama_decode(query replay) failed\n");
            llama_kvmem_set_replay(false);
            return false;
        }
        llama_synchronize(ctx);
    }
    llama_kvmem_set_replay(false);
    fprintf(stderr, "KVMEM_TRACE query_replay begin=%d n=%d recr_ckpt=%d\n",
            q0, qn, (int) recr_ckpt);
    return true;
}

static void derive_query_span(ServerState & st, const std::string & prompt, const std::string & last_user,
                              const std::vector<llama_token> & toks, int & qbegin, int & qend) {
    qend = (int) toks.size();
    qbegin = -1;
    if (!last_user.empty()) {
        const auto idx = prompt.rfind(last_user);
        if (idx != std::string::npos) {
            const std::string prefix = prompt.substr(0, idx);
            qbegin = (int) tokenize_text(st.vocab, prefix, true).size();
        }
    }
    if (qbegin < 0) {
        const int last = std::min(st.query_last_fallback, qend);
        qbegin = qend > last ? qend - last : 0;
    }
    if (qbegin >= qend) {
        qbegin = 0;
    }
}

struct ChatRequest {
    std::vector<common_chat_msg> msgs;
    std::string last_user;
    int max_tokens = 128;
    float temperature = 0.0f;
    bool stream = false;
    int query_begin = -1;
    int query_end = -1;
    std::string force_substr;
    bool enable_thinking = false;
};

static bool parse_chat_request(const json & body, ChatRequest & out, std::string & err) {
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        err = "messages array required";
        return false;
    }
    for (const auto & m : body["messages"]) {
        common_chat_msg msg;
        msg.role = m.value("role", "user");
        if (m.contains("content") && m["content"].is_string()) {
            msg.content = m["content"].get<std::string>();
        } else if (m.contains("content") && m["content"].is_array()) {
            for (const auto & part : m["content"]) {
                if (part.is_object() && part.value("type", "") == "text") {
                    msg.content += part.value("text", "");
                } else if (part.is_string()) {
                    msg.content += part.get<std::string>();
                }
            }
        }
        if (msg.role == "user") {
            out.last_user = msg.content;
        }
        out.msgs.push_back(std::move(msg));
    }
    out.max_tokens = body.value("max_tokens", 128);
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
        llama_kvmem_set_params(&st.kparams);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
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
    st.ctx = llama_init_from_model(st.model, cparams);
    if (!st.ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    httplib::Server svr;
    svr.set_read_timeout(600, 0);
    svr.set_write_timeout(600, 0);
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
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        inputs.enable_thinking = cr.enable_thinking;
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
        fprintf(stderr, "KVMEM_TRACE n_prompt=%d query=[%d,%d) force_pos=%d last_user_chars=%zu\n",
                (int) toks.size(), qbegin, qend, force, cr.last_user.size());

        if (!run_prefill_retrieval(st, toks)) {
            res.status = 500;
            res.set_content("{\"error\":\"prefill/retrieval failed\"}", "application/json");
            return;
        }

        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sp);
        if (cr.temperature <= 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(cr.temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
        }

        const std::string cid = "chatcmpl-kvmem";
        llama_context * ctx = st.ctx;
        const llama_vocab * vocab = st.vocab;
        auto gen_one = [ctx, smpl, vocab](std::string & piece, bool & stopped) -> bool {
            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) {
                stopped = true;
                return true;
            }
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
                [slot, smpl, cid, gen_one, max_tokens](size_t, httplib::DataSink & sink) mutable {
                    auto send = [&](const std::string & payload) {
                        const std::string line = "data: " + payload + "\n\n";
                        sink.write(line.data(), line.size());
                    };
                    send(std::string("{\"id\":\"") + cid +
                         "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}");
                    int n_gen = 0;
                    bool stopped = false;
                    while (n_gen < max_tokens && !stopped) {
                        std::string piece;
                        if (!gen_one(piece, stopped)) {
                            break;
                        }
                        if (stopped) {
                            break;
                        }
                        ++n_gen;
                        send(std::string("{\"id\":\"") + cid +
                             "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
                             json_escape(piece) + "\"},\"finish_reason\":null}]}");
                    }
                    send(std::string("{\"id\":\"") + cid +
                         "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}");
                    sink.write("data: [DONE]\n\n", 15);
                    llama_sampler_free(smpl);
                    slot->unlock();
                    sink.done();
                    return true;
                });
            return;
        }

        std::string content;
        int n_gen = 0;
        bool stopped = false;
        while (n_gen < cr.max_tokens && !stopped) {
            std::string piece;
            if (!gen_one(piece, stopped)) {
                llama_sampler_free(smpl);
                res.status = 500;
                res.set_content("{\"error\":\"decode failed\"}", "application/json");
                return;
            }
            if (stopped) {
                break;
            }
            content += piece;
            ++n_gen;
        }
        llama_sampler_free(smpl);
        json out = {
            {"id", cid},
            {"object", "chat.completion"},
            {"model", st.model_name},
            {"choices", json::array({json{
                {"index", 0},
                {"message", json{{"role", "assistant"}, {"content", content}}},
                {"finish_reason", "stop"},
            }})},
            {"usage", json{
                {"prompt_tokens", (int) toks.size()},
                {"completion_tokens", n_gen},
                {"total_tokens", (int) toks.size() + n_gen},
            }},
        };
        res.set_content(out.dump(), "application/json");
    };

    svr.Post("/v1/chat/completions", handle_chat);
    svr.Post("/chat/completions", handle_chat);

    fprintf(stderr, "llama-kvmem-server listening on http://%s:%d  model=%s kvmem=%d method=%s n_ctx=%d\n",
            host.c_str(), port, st.model_name.c_str(), (int) st.kparams.enabled,
            st.kparams.method == 1 ? "retrieval" : "recency", n_ctx);
    if (!svr.listen(host, port)) {
        fprintf(stderr, "listen failed\n");
        return 1;
    }
    llama_free(st.ctx);
    llama_model_free(st.model);
    return 0;
}
