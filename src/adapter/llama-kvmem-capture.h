#pragma once

#include "llama.h"

#include <vector>

struct llama_memory_kvmem;
struct ggml_tensor;

void kvmem_capture_bind(llama_memory_kvmem * mem);
void kvmem_capture_unbind(llama_memory_kvmem * mem);
llama_memory_kvmem * kvmem_capture_active();
void kvmem_capture_note_ubatch(const std::vector<llama_pos> & pos);
void kvmem_capture_reset_q();
void kvmem_capture_register(struct ggml_tensor * t, int il, char which);
void kvmem_capture_on_new_graph(void);
void kvmem_capture_harvest_ubatch(struct ggml_backend_sched * sched);
bool kvmem_ubatch_needs_q_capture(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos);
bool kvmem_capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos);
