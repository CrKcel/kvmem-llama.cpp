#include "llama-kvmem-diag.h"
#include "kvmem-spec.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

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
    if (std::strcmp(s, "q5_0") == 0 || std::strcmp(s, "q5") == 0) {
        return GGML_TYPE_Q5_0;
    }
    if (ok) {
        *ok = false;
    }
    return GGML_TYPE_F16;
}

bool kvmem_cache_types_ok(ggml_type type_k, ggml_type type_v) {
    if (ggml_is_quantized(type_k) || ggml_is_quantized(type_v)) {
        const auto supported_quant = [](ggml_type type) {
            return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q4_0;
        };
        return supported_quant(type_k) && supported_quant(type_v);
    }
    return true;
}

bool kvmem_spec_start(kvmem_spec_session &, llama_model *, llama_context *, const kvmem_spec_opts &) {
    fprintf(stderr, "MTP is disabled in the Bonsai experimental build\n");
    return false;
}

void kvmem_spec_stop(kvmem_spec_session & sess) {
    sess = kvmem_spec_session{};
}

int kvmem_spec_decode_span(llama_context * ctx,
                           common_speculative * spec,
                           const llama_token * toks,
                           int pos0, int pos1, int n_batch,
                           const char * what,
                           const std::function<bool()> & abort) {
    if (pos0 >= pos1) {
        return 0;
    }
    if (n_batch <= 0) {
        n_batch = 512;
    }
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int n_pos = pos0;
    while (n_pos < pos1) {
        if (abort && abort()) {
            kvmem_diag("KVMEM_TRACE stream_abort phase=prefill pos=%d what=%s\n",
                    n_pos, what ? what : "");
            llama_batch_free(batch);
            return KVMEM_DECODE_ABORT;
        }
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
        llama_context *, llama_model *, kvmem_spec_session &, const std::vector<llama_token> &,
        int, float, kvmem_spec_on_token) {
    throw std::runtime_error("MTP is disabled in the Bonsai experimental build");
}

kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context *, llama_model *, kvmem_spec_session &, const std::vector<llama_token> &,
        int, common_params_sampling, kvmem_spec_on_token, const std::function<bool()> &, llama_pos) {
    throw std::runtime_error("MTP is disabled in the Bonsai experimental build");
}
