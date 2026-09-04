#include "llama-kvmem-capture.h"

#include "llama-kvmem-hooks.h"
#include "llama-memory-kvmem.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static llama_memory_kvmem * g_mem = nullptr;

void kvmem_capture_bind(llama_memory_kvmem * mem) {
    g_mem = mem;
}

llama_memory_kvmem * kvmem_capture_active() {
    return g_mem;
}

void kvmem_capture_unbind(llama_memory_kvmem * mem) {
    if (g_mem == mem) {
        g_mem = nullptr;
    }
}

void kvmem_capture_note_ubatch(const std::vector<llama_pos> & pos) {
    if (g_mem) {
        g_mem->note_ubatch_pos(pos);
    }
}

void kvmem_capture_reset_q() {
    if (g_mem) {
        g_mem->reset_query_acc();
    }
}

void kvmem_capture_register(struct ggml_tensor * t, int il, char which) {
    if (g_mem) {
        g_mem->register_capture(t, il, which);
    }
}

void kvmem_capture_on_new_graph(void) {
    if (g_mem) {
        g_mem->capture_on_new_graph();
    }
}

void kvmem_capture_harvest_ubatch(struct ggml_backend_sched * sched) {
    if (g_mem) {
        g_mem->harvest_pending(sched);
    }
}

static bool ubatch_overlaps_query(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    const llama_kvmem_params * kp = llama_kvmem_get_params();
    if (!kp || !kp->enabled || kp->method != 1 || kp->query_begin < 0 || n_tokens == 0) {
        return false;
    }
    if (g_mem && g_mem->replay()) {
        return false;
    }
    const int32_t qb = kp->query_begin;
    const int32_t qe = kp->query_end > 0 ? kp->query_end : (1 << 30);
    const uint32_t stride = n_pos > 0 ? n_pos : 1u;
    if (!pos) {
        return true;
    }
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_pos p = pos[i * stride];
        if (p >= qb && p < qe) {
            return true;
        }
    }
    return false;
}

bool kvmem_ubatch_needs_q_capture(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    return ubatch_overlaps_query(n_tokens, n_pos, pos);
}

bool kvmem_capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    if (!g_mem) {
        return true;
    }
    return g_mem->capture_can_reuse(n_tokens, n_pos, pos);
}

bool llama_kvmem_eval_callback(struct ggml_tensor * /*t*/, bool /*ask*/, void * /*user_data*/) {
    // Do not observe nodes: the scheduler splits and synchronizes on every
    // true ask. Harvest is llama_kvmem_harvest_ubatch after the full graph.
    return false;
}

void llama_kvmem_register_capture(struct ggml_tensor * t, int il, char which) {
    kvmem_capture_register(t, il, which);
}

void llama_kvmem_capture_on_new_graph(void) {
    kvmem_capture_on_new_graph();
}

void llama_kvmem_harvest_ubatch(struct ggml_backend_sched * sched) {
    kvmem_capture_harvest_ubatch(sched);
}

bool llama_kvmem_ubatch_needs_q_capture(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    return kvmem_ubatch_needs_q_capture(n_tokens, n_pos, pos);
}

bool llama_kvmem_capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    return kvmem_capture_can_reuse(n_tokens, n_pos, pos);
}
