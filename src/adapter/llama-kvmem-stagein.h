#pragma once

// GPU stage-in for quantized raw-K:
//   H2D packed q8/q4 → dequant F32 → NeoX RoPE → Walsh-Hadamard → quant into cache.
// F32-H2D path remains for fallback / unrotated V. Host Hadamard/RoPE is fallback only.

#include "ggml.h"

#include <cstddef>
#include <cstdint>

bool kvmem_stagein_gpu_ready(size_t n_f32, size_t n_packed = 0);
void kvmem_stagein_gpu_free();

bool kvmem_stagein_fwht_ok(int nrot);
bool kvmem_stagein_quant_ok(ggml_type ty);

// Host buffer is free to reuse after return (H2D is fenced). Later kernels stay async.
bool kvmem_stagein_h2d_f32(const float * host, int64_t n);
bool kvmem_stagein_h2d_packed(const void * host, size_t n);
bool kvmem_stagein_dequant(ggml_type ty, int64_t n_rows, int64_t n_embd);
bool kvmem_stagein_rope_neox(int64_t n_tokens, int n_head, int n_embd_head, int n_rot,
                             int32_t pos0, const float * theta, int n_theta);
bool kvmem_stagein_fwht(int64_t n_rows, int64_t n_embd, int nrot);
bool kvmem_stagein_quantize(ggml_type ty, void * gpu_dst, int64_t n_rows, int64_t n_embd);
bool kvmem_stagein_h2d_bytes(void * gpu_dst, const void * host, size_t n);
void kvmem_stagein_sync();
