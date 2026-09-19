#pragma once

// Shared draft-mtp wiring for llama-kvmem-cli / llama-kvmem-server.
// Does not reimplement llama.cpp's speculative state machine.

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <functional>
#include <string>
#include <vector>

struct kvmem_spec_opts {
    int32_t n_max = 2;
    int32_t n_min = 0;
    float   p_min = 0.0f;
    int32_t n_gpu_layers = 99;
    int32_t n_ctx = 0;
    int32_t n_batch = 512;
    int32_t n_ubatch = 512;
    int32_t n_threads = -1;
    int32_t n_threads_batch = -1;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool    kvmem_enabled = false;
    std::string draft_model; // optional sidecar GGUF; empty = welded nextn
    ggml_type type_k = GGML_TYPE_Q8_0;
    ggml_type type_v = GGML_TYPE_Q8_0;
    ggml_type draft_type = GGML_TYPE_COUNT; // inherit target K/V types unless overridden
};

// Supported cache types: f16, f32, q8_0, q5_0, q4_0.
// q5_0 requires GGML_CUDA_FA_ALL_QUANTS (enabled by this project's build).
ggml_type kvmem_parse_cache_type(const char * s, bool * ok);
// Quantized K/V must match in the supported configurations.
bool kvmem_cache_types_ok(ggml_type type_k, ggml_type type_v);

struct kvmem_spec_session {
    common_params spec_params;
    common_speculative_init_result_ptr init;
    common_speculative * spec = nullptr;
    llama_context * ctx_dft = nullptr;
    bool use_ckpt_tgt = false;
    bool use_ckpt_dft = false;
    bool use_gdn_replay = false;
    uint32_t n_rs_tgt = 0;
    bool ok = false;
};

bool kvmem_spec_start(kvmem_spec_session & sess,
                      llama_model * model_tgt,
                      llama_context * ctx_tgt,
                      const kvmem_spec_opts & opts);

void kvmem_spec_stop(kvmem_spec_session & sess);

// decode_span / spec_decode_span: client disconnected mid-prefill.
constexpr int KVMEM_DECODE_ABORT = -100;

// Decode [pos0, pos1) with explicit pos/seq_id so process() can catch up draft KV.
// `abort` if set: return KVMEM_DECODE_ABORT when it returns true (checked between batches).
int kvmem_spec_decode_span(llama_context * ctx,
                           common_speculative * spec,
                           const llama_token * toks,
                           int pos0, int pos1, int n_batch,
                           const char * what,
                           const std::function<bool()> & abort = {});

using kvmem_spec_on_token =
        std::function<void(llama_token id, const std::string & piece, bool from_draft)>;

struct kvmem_spec_gen_stats {
    int n_past = 0;
    int n_gen = 0;
    int n_drafted = 0;
    int n_accept = 0;
    int n_restore = 0;
    bool failed = false;
};

// `prompt` is the full prompt; the last token is id_last and must not already
// be in the target KV (same contract as llama.cpp speculative-simple).
kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        float temp,
        kvmem_spec_on_token on_token);

// Same as above, but uses a fully-specified sampler (grammar / lazy triggers).
// `abort` if set: stop the generate loop when it returns true (not a failure).
kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        common_params_sampling sparams,
        kvmem_spec_on_token on_token,
        const std::function<bool()> & abort = {},
        llama_pos position_offset = 0);
