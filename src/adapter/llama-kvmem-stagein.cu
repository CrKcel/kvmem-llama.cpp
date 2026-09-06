#include "llama-kvmem-stagein.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int QK8_0 = 32;
constexpr int QK4_0 = 32;

struct __align__(2) block_q8_0 {
    half    d;
    int8_t  qs[QK8_0];
};

struct __align__(2) block_q4_0 {
    half    d;
    uint8_t qs[QK4_0 / 2];
};

static_assert(sizeof(block_q8_0) == 34, "q8_0 block");
static_assert(sizeof(block_q4_0) == 18, "q4_0 block");

struct Stage {
    float *     dev_f32   = nullptr;
    size_t      n_f32     = 0;
    uint8_t *   dev_q     = nullptr;
    size_t      n_q       = 0;
    float *     dev_theta = nullptr;
    size_t      n_theta   = 0;
    cudaEvent_t h2d_ev    = nullptr;
};

Stage g_st;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "KVMEM stagein %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

cudaStream_t stream() {
    return cudaStreamPerThread;
}

template <int N>
__global__ void fwht_kernel(float * x, int64_t n_rows, float scale) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) {
        return;
    }
    float * row = x + r * (int64_t) N;
    float tmp[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        tmp[i] = row[i] * scale;
    }
    for (int h = 1; h < N; h *= 2) {
        for (int i = 0; i < N; i += 2 * h) {
            for (int j = 0; j < h; ++j) {
                const float a = tmp[i + j];
                const float b = tmp[i + j + h];
                tmp[i + j]       = a + b;
                tmp[i + j + h]   = a - b;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
        row[i] = tmp[i];
    }
}

__global__ void dequant_q8_0(const block_q8_0 * x, float * y, int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const float d = __half2float(x[i].d);
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        y[i * QK8_0 + j] = (float) x[i].qs[j] * d;
    }
}

__global__ void dequant_q4_0(const block_q4_0 * x, float * y, int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const float d = __half2float(x[i].d);
#pragma unroll
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const int x0 = (x[i].qs[j] & 0x0F) - 8;
        const int x1 = (x[i].qs[j] >> 4) - 8;
        y[i * QK4_0 + j]              = (float) x0 * d;
        y[i * QK4_0 + j + QK4_0 / 2]  = (float) x1 * d;
    }
}

__global__ void rope_neox_kernel(float * x, int64_t n_tokens, int n_head, int n_embd_head,
                                 int n_rot, int32_t pos0, const float * theta) {
    const int t = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int h = (int) blockIdx.y;
    if (t >= n_tokens || h >= n_head) {
        return;
    }
    float * row = x + ((int64_t) t * n_head + h) * n_embd_head;
    const int half = n_rot / 2;
    const float pos = (float) (pos0 + t);
    for (int i = 0; i < half; ++i) {
        float s;
        float c;
        sincosf(pos * theta[i], &s, &c);
        const float x0 = row[i];
        const float x1 = row[i + half];
        row[i]        = x0 * c - x1 * s;
        row[i + half] = x0 * s + x1 * c;
    }
}

__global__ void quant_q8_0(const float * x, block_q8_0 * y, int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const float * src = x + i * QK8_0;
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        amax = fmaxf(amax, fabsf(src[j]));
    }
    const float d  = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = __float2half(d);
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        y[i].qs[j] = (int8_t) roundf(src[j] * id);
    }
}

__global__ void quant_q4_0(const float * x, block_q4_0 * y, int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const float * src = x + i * QK4_0;
    float amax = 0.0f;
    float vmax = 0.0f;
#pragma unroll
    for (int j = 0; j < QK4_0; ++j) {
        const float v = src[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }
    const float d  = vmax / -8.0f;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = __float2half(d);
#pragma unroll
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const float x0 = src[j] * id;
        const float x1 = src[j + QK4_0 / 2] * id;
        int xi0 = (int) (x0 + 8.5f);
        int xi1 = (int) (x1 + 8.5f);
        if (xi0 > 15) {
            xi0 = 15;
        }
        if (xi1 > 15) {
            xi1 = 15;
        }
        y[i].qs[j] = (uint8_t) ((xi0 & 15) | ((xi1 & 15) << 4));
    }
}

}  // namespace

bool kvmem_stagein_gpu_ready(size_t n_f32, size_t n_packed) {
    if (n_f32 == 0) {
        return false;
    }
    if (!g_st.h2d_ev) {
        if (!cuda_ok(cudaEventCreateWithFlags(&g_st.h2d_ev, cudaEventDisableTiming), "event")) {
            return false;
        }
    }
    if (!g_st.dev_f32 || g_st.n_f32 < n_f32) {
        if (g_st.dev_f32) {
            cudaFree(g_st.dev_f32);
            g_st.dev_f32 = nullptr;
            g_st.n_f32 = 0;
        }
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&g_st.dev_f32), n_f32 * sizeof(float)),
                     "f32 scratch")) {
            g_st.dev_f32 = nullptr;
            return false;
        }
        g_st.n_f32 = n_f32;
    }
    if (n_packed > 0 && (!g_st.dev_q || g_st.n_q < n_packed)) {
        if (g_st.dev_q) {
            cudaFree(g_st.dev_q);
            g_st.dev_q = nullptr;
            g_st.n_q = 0;
        }
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&g_st.dev_q), n_packed), "q scratch")) {
            g_st.dev_q = nullptr;
            return false;
        }
        g_st.n_q = n_packed;
    }
    return g_st.dev_f32 != nullptr;
}

void kvmem_stagein_gpu_free() {
    if (g_st.dev_f32) {
        cudaFree(g_st.dev_f32);
        g_st.dev_f32 = nullptr;
    }
    g_st.n_f32 = 0;
    if (g_st.dev_q) {
        cudaFree(g_st.dev_q);
        g_st.dev_q = nullptr;
    }
    g_st.n_q = 0;
    if (g_st.dev_theta) {
        cudaFree(g_st.dev_theta);
        g_st.dev_theta = nullptr;
    }
    g_st.n_theta = 0;
    if (g_st.h2d_ev) {
        cudaEventDestroy(g_st.h2d_ev);
        g_st.h2d_ev = nullptr;
    }
}

bool kvmem_stagein_fwht_ok(int nrot) {
    return nrot == 64 || nrot == 128 || nrot == 256 || nrot == 512;
}

bool kvmem_stagein_quant_ok(ggml_type ty) {
    return ty == GGML_TYPE_Q8_0 || ty == GGML_TYPE_Q4_0;
}

bool kvmem_stagein_h2d_packed(const void * host, size_t n) {
    if (!host || n == 0 || !g_st.dev_q || n > g_st.n_q) {
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(g_st.dev_q, host, n, cudaMemcpyHostToDevice, stream()),
                 "H2D packed")) {
        return false;
    }
    if (!cuda_ok(cudaEventRecord(g_st.h2d_ev, stream()), "H2D packed record")) {
        return false;
    }
    return cuda_ok(cudaEventSynchronize(g_st.h2d_ev), "H2D packed wait");
}

bool kvmem_stagein_dequant(ggml_type ty, int64_t n_rows, int64_t n_embd) {
    if (!g_st.dev_q || !g_st.dev_f32 || n_rows <= 0 || n_embd <= 0) {
        return false;
    }
    if ((size_t) n_rows * (size_t) n_embd > g_st.n_f32) {
        return false;
    }
    const int threads = 256;
    if (ty == GGML_TYPE_Q8_0) {
        if (n_embd % QK8_0 != 0) {
            return false;
        }
        const int64_t n_blocks = n_rows * (n_embd / QK8_0);
        const int blocks = (int) ((n_blocks + threads - 1) / threads);
        dequant_q8_0<<<blocks, threads, 0, stream()>>>(
                reinterpret_cast<const block_q8_0 *>(g_st.dev_q), g_st.dev_f32, n_blocks);
        return cuda_ok(cudaGetLastError(), "dequant q8_0");
    }
    if (ty == GGML_TYPE_Q4_0) {
        if (n_embd % QK4_0 != 0) {
            return false;
        }
        const int64_t n_blocks = n_rows * (n_embd / QK4_0);
        const int blocks = (int) ((n_blocks + threads - 1) / threads);
        dequant_q4_0<<<blocks, threads, 0, stream()>>>(
                reinterpret_cast<const block_q4_0 *>(g_st.dev_q), g_st.dev_f32, n_blocks);
        return cuda_ok(cudaGetLastError(), "dequant q4_0");
    }
    return false;
}

bool kvmem_stagein_rope_neox(int64_t n_tokens, int n_head, int n_embd_head, int n_rot,
                             int32_t pos0, const float * theta, int n_theta) {
    if (!g_st.dev_f32 || !theta || n_tokens <= 0 || n_head <= 0 || n_embd_head <= 0) {
        return false;
    }
    if (n_rot < 2 || (n_rot % 2) != 0 || n_rot > n_embd_head || n_theta != n_rot / 2) {
        return false;
    }
    if ((size_t) n_tokens * (size_t) n_head * (size_t) n_embd_head > g_st.n_f32) {
        return false;
    }
    if (!g_st.dev_theta || g_st.n_theta < (size_t) n_theta) {
        if (g_st.dev_theta) {
            cudaFree(g_st.dev_theta);
            g_st.dev_theta = nullptr;
            g_st.n_theta = 0;
        }
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&g_st.dev_theta),
                                (size_t) n_theta * sizeof(float)),
                     "theta")) {
            return false;
        }
        g_st.n_theta = (size_t) n_theta;
    }
    if (!cuda_ok(cudaMemcpyAsync(g_st.dev_theta, theta, (size_t) n_theta * sizeof(float),
                                 cudaMemcpyHostToDevice, stream()),
                 "H2D theta")) {
        return false;
    }
    const int threads = 64;
    const int blocks_t = (int) ((n_tokens + threads - 1) / threads);
    dim3 grid(blocks_t, n_head, 1);
    rope_neox_kernel<<<grid, threads, 0, stream()>>>(
            g_st.dev_f32, n_tokens, n_head, n_embd_head, n_rot, pos0, g_st.dev_theta);
    return cuda_ok(cudaGetLastError(), "rope launch");
}

bool kvmem_stagein_h2d_f32(const float * host, int64_t n) {
    if (!host || n <= 0 || !g_st.dev_f32 || (size_t) n > g_st.n_f32) {
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(g_st.dev_f32, host, (size_t) n * sizeof(float),
                                 cudaMemcpyHostToDevice, stream()),
                 "H2D f32")) {
        return false;
    }
    if (!cuda_ok(cudaEventRecord(g_st.h2d_ev, stream()), "H2D record")) {
        return false;
    }
    return cuda_ok(cudaEventSynchronize(g_st.h2d_ev), "H2D wait");
}

bool kvmem_stagein_fwht(int64_t n_rows, int64_t n_embd, int nrot) {
    if (!g_st.dev_f32 || n_rows <= 0 || n_embd <= 0 || nrot <= 0) {
        return false;
    }
    if (n_embd % nrot != 0 || !kvmem_stagein_fwht_ok(nrot)) {
        return false;
    }
    const int64_t nrows = n_rows * (n_embd / nrot);
    const float scale = 1.0f / sqrtf((float) nrot);
    const int threads = 128;
    const int blocks = (int) ((nrows + threads - 1) / threads);
    switch (nrot) {
        case 64:
            fwht_kernel<64><<<blocks, threads, 0, stream()>>>(g_st.dev_f32, nrows, scale);
            break;
        case 128:
            fwht_kernel<128><<<blocks, threads, 0, stream()>>>(g_st.dev_f32, nrows, scale);
            break;
        case 256:
            fwht_kernel<256><<<blocks, threads, 0, stream()>>>(g_st.dev_f32, nrows, scale);
            break;
        case 512:
            fwht_kernel<512><<<blocks, threads, 0, stream()>>>(g_st.dev_f32, nrows, scale);
            break;
        default:
            return false;
    }
    return cuda_ok(cudaGetLastError(), "fwht launch");
}

bool kvmem_stagein_quantize(ggml_type ty, void * gpu_dst, int64_t n_rows, int64_t n_embd) {
    if (!g_st.dev_f32 || !gpu_dst || n_rows <= 0 || n_embd <= 0) {
        return false;
    }
    const int threads = 256;
    if (ty == GGML_TYPE_Q8_0) {
        if (n_embd % QK8_0 != 0) {
            return false;
        }
        const int64_t n_blocks = n_rows * (n_embd / QK8_0);
        const int blocks = (int) ((n_blocks + threads - 1) / threads);
        quant_q8_0<<<blocks, threads, 0, stream()>>>(
                g_st.dev_f32, static_cast<block_q8_0 *>(gpu_dst), n_blocks);
        return cuda_ok(cudaGetLastError(), "q8_0 launch");
    }
    if (ty == GGML_TYPE_Q4_0) {
        if (n_embd % QK4_0 != 0) {
            return false;
        }
        const int64_t n_blocks = n_rows * (n_embd / QK4_0);
        const int blocks = (int) ((n_blocks + threads - 1) / threads);
        quant_q4_0<<<blocks, threads, 0, stream()>>>(
                g_st.dev_f32, static_cast<block_q4_0 *>(gpu_dst), n_blocks);
        return cuda_ok(cudaGetLastError(), "q4_0 launch");
    }
    return false;
}

bool kvmem_stagein_h2d_bytes(void * gpu_dst, const void * host, size_t n) {
    if (!gpu_dst || !host || n == 0) {
        return n == 0;
    }
    return cuda_ok(cudaMemcpyAsync(gpu_dst, host, n, cudaMemcpyHostToDevice, stream()),
                   "H2D bytes");
}

void kvmem_stagein_sync() {
    cuda_ok(cudaStreamSynchronize(stream()), "sync");
}
