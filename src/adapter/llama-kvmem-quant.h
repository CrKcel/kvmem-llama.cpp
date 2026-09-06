#pragma once

// Pack/unpack one token-major span of GPU KV cache rows (n_embd_k/v per row).
// Unrotated raw-K uses the same pack (no Hadamard). Stage-in unpacks, RoPEs,
// applies attn_rot, then packs into the working cache. Packed GPU V is copied
// as-is.
//
// Quantized llama.cpp caches apply a Walsh-Hadamard rotation per head before
// quantize (attn_rot_k/v when n_embd_head % 64 == 0). Stage-in/out must match.
// H is orthonormal and H^2 = I, so the same multiply inverts.

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

inline bool kvmem_attn_rot_on(ggml_type ty, int n_embd_head) {
    if (!ggml_is_quantized(ty) || n_embd_head < 64 || (n_embd_head % 64) != 0) {
        return false;
    }
    const char * e = std::getenv("LLAMA_ATTN_ROT_DISABLE");
    return !(e && std::atoi(e) != 0);
}

inline int kvmem_hadamard_nrot_k(int n_embd_head) {
    int nrot = 64;
    do {
        nrot *= 2;
    } while (n_embd_head % nrot == 0);
    return nrot / 2;
}

inline int kvmem_hadamard_nrot_v(int /*n_embd_head*/) {
    return 64;
}

inline const std::vector<float> & kvmem_hadamard_matrix(int n) {
    static std::mutex mu;
    static std::unordered_map<int, std::vector<float>> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(n);
    if (it != cache.end()) {
        return it->second;
    }
    std::vector<float> data(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);
    data[0] = 1.0f / std::sqrt(static_cast<float>(n));
    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                const float val = data[static_cast<size_t>(i) * n + j];
                data[static_cast<size_t>(i + s) * n + j] = val;
                data[static_cast<size_t>(i) * n + (j + s)] = val;
                data[static_cast<size_t>(i + s) * n + (j + s)] = -val;
            }
        }
    }
    return cache.emplace(n, std::move(data)).first->second;
}

// Token-major [n_rows][n_head * n_embd_head]. Applies H (nrot x nrot) to each
// nrot chunk of every head. H^2 = I.
inline void kvmem_hadamard_rows(float * rows, int64_t n_rows, int n_head,
                                int n_embd_head, int nrot) {
    if (!rows || n_rows <= 0 || n_head <= 0 || nrot <= 0 ||
        n_embd_head < nrot || (n_embd_head % nrot) != 0) {
        return;
    }
    const std::vector<float> & H = kvmem_hadamard_matrix(nrot);
    std::vector<float> tmp(static_cast<size_t>(nrot));
    const int64_t row = static_cast<int64_t>(n_head) * n_embd_head;
    for (int64_t t = 0; t < n_rows; ++t) {
        float * base = rows + t * row;
        for (int h = 0; h < n_head; ++h) {
            float * head = base + static_cast<int64_t>(h) * n_embd_head;
            for (int off = 0; off < n_embd_head; off += nrot) {
                float * x = head + off;
                for (int i = 0; i < nrot; ++i) {
                    float acc = 0.0f;
                    const float * Hi = H.data() + static_cast<size_t>(i) * nrot;
                    for (int j = 0; j < nrot; ++j) {
                        acc += Hi[j] * x[j];
                    }
                    tmp[static_cast<size_t>(i)] = acc;
                }
                std::memcpy(x, tmp.data(), static_cast<size_t>(nrot) * sizeof(float));
            }
        }
    }
}

inline bool kvmem_cache_pack_rows(ggml_type ty, const float * src, void * dst,
                                  int64_t n_rows, int64_t n_per_row) {
    if (!src || !dst || n_rows <= 0 || n_per_row <= 0) {
        return false;
    }
    const int64_t n = n_rows * n_per_row;
    if (ty == GGML_TYPE_F32) {
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
        return true;
    }
    if (ty == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, static_cast<ggml_fp16_t *>(dst), n);
        return true;
    }
    if (!ggml_is_quantized(ty)) {
        return false;
    }
    const int64_t bs = ggml_blck_size(ty);
    if (bs <= 0 || n_per_row % bs != 0) {
        return false;
    }
    ggml_quantize_chunk(ty, src, dst, 0, n_rows, n_per_row, nullptr);
    return true;
}

inline bool kvmem_cache_unpack_rows(ggml_type ty, const void * src, float * dst,
                                    int64_t n_rows, int64_t n_per_row) {
    if (!src || !dst || n_rows <= 0 || n_per_row <= 0) {
        return false;
    }
    const int64_t n = n_rows * n_per_row;
    if (ty == GGML_TYPE_F32) {
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
        return true;
    }
    if (ty == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(src), dst, n);
        return true;
    }
    const struct ggml_type_traits * tt = ggml_get_type_traits(ty);
    if (!tt || !tt->to_float) {
        return false;
    }
    const int64_t bs = ggml_blck_size(ty);
    if (bs <= 0 || n_per_row % bs != 0) {
        return false;
    }
    tt->to_float(src, dst, n);
    return true;
}
