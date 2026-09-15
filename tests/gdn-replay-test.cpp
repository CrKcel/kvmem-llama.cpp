#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<float> values(size_t n, uint32_t seed, float scale) {
    std::vector<float> result(n);
    for (float & x : result) {
        seed = seed * 1664525u + 1013904223u;
        x = (float(int32_t(seed >> 8)) / 8388608.0f - 1.0f) * scale;
    }
    return result;
}

static void set(ggml_tensor * t, uint32_t seed, float scale) {
    const auto data = values(ggml_nelements(t), seed, scale);
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> result(ggml_nelements(t));
    ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
    return result;
}

static std::vector<float> snapshot(ggml_backend_t backend, int tokens) {
    ggml_init_params params{2 * 1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(params);
    require(ctx != nullptr, "context allocation failed");
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
    auto * k = ggml_dup_tensor(ctx, q);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, tokens, 1);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
    auto * b = ggml_dup_tensor(ctx, g);
    auto * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 48, 1);
    auto * out = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 3);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "buffer allocation failed");
    ggml_backend_buffer_clear(buffer, 0);
    set(q, 123, 0.12f);
    set(k, 234, 0.12f);
    set(v, 345, 0.25f);
    set(g, 456, 0.02f);
    set(b, 567, 0.5f);
    set(s, 678, 0.1f);
    const auto initial = get(s);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "snapshot compute failed");
    const auto result = get(out);
    require(initial == get(s), "snapshot overwrote input state");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static void check_replay(ggml_backend_t backend, int tokens, int rounds, bool fold = true) {
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "replay context allocation failed");
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
    auto * k = ggml_dup_tensor(ctx, q);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, tokens, 1);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
    auto * b = ggml_dup_tensor(ctx, g);
    auto * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 48, 1);
    auto * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 10240);
    auto * conv_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10240, tokens);
    auto * descriptor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(ggml_cuda_gdn_replay_layer));
    auto * reference = ggml_gated_delta_net(ctx, q, k, v, g, b, s, tokens);
    auto * recorded = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 0);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, recorded);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "replay buffer allocation failed");
    set(q, 12, 0.1f);
    set(k, 23, 0.1f);
    set(v, 34, 0.2f);
    set(s, 45, 0.1f);
    set(conv, 56, 0.1f);
    set(conv_input, 67, 0.1f);
    auto gates = values(ggml_nelements(g), 78, .05f);
    auto betas = values(ggml_nelements(b), 89, .9f);
    for (auto & x : gates) x = -std::fabs(x);
    for (auto & x : betas) x = std::fabs(x);
    ggml_backend_tensor_set(g, gates.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(b, betas.data(), 0, ggml_nbytes(b));
    const ggml_cuda_gdn_replay_layer layer{static_cast<float *>(s->data), static_cast<float *>(conv->data),
        static_cast<float *>(k->data), static_cast<float *>(v->data), static_cast<float *>(g->data),
        static_cast<float *>(b->data), static_cast<float *>(conv_input->data)};
    ggml_backend_tensor_set(descriptor, &layer, 0, sizeof(layer));
    const auto original_key = get(k);
    const auto columns = get(conv_input);
    require(ggml_backend_cuda_gdn_fold(nullptr, 0, 0, 0, nullptr), "zero Fold must not access descriptors");
    for (int round = 0; round < rounds; ++round) {
        ggml_backend_tensor_set(k, original_key.data(), 0, ggml_nbytes(k));
        const auto initial = get(s);
        const auto history = get(conv);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Record compute failed");
        require(initial == get(s), "Record changed committed state");
        const auto snapshots = get(reference);
        const auto output = get(recorded);
        require(std::memcmp(snapshots.data(), output.data(), ggml_nbytes(recorded)) == 0, "Record attention differs");
        if (!fold) continue;
        const int first = rounds == 1 ? 0 : round % (tokens + 1);
        const int last = rounds == 1 ? tokens : first;
        for (int keep = first; keep <= last; ++keep) {
            ggml_backend_tensor_set(s, initial.data(), 0, ggml_nbytes(s));
            ggml_backend_tensor_set(conv, history.data(), 0, ggml_nbytes(conv));
            auto key = original_key;
            std::fill(key.begin() + keep * 2048, key.end(), std::numeric_limits<float>::quiet_NaN());
            ggml_backend_tensor_set(k, key.data(), 0, ggml_nbytes(k));
            require(ggml_backend_cuda_gdn_fold(static_cast<const ggml_cuda_gdn_replay_layer *>(descriptor->data),
                        1, keep, tokens, nullptr), "Fold launch failed");
            const auto actual = get(s);
            const float * expected = keep == 0 ? initial.data()
                : snapshots.data() + output.size() + (tokens - keep) * initial.size();
            require(std::memcmp(expected, actual.data(), ggml_nbytes(s)) == 0, "Fold state differs from accepted snapshot");
            const auto actual_conv = get(conv);
            for (int c = 0; c < 10240; ++c) {
                for (int i = 0; i < 3; ++i) {
                    const int source = keep + i;
                    const float expected_conv = source < 3 ? history[c * 3 + source] : columns[(source - 3) * 10240 + c];
                    require(actual_conv[c * 3 + i] == expected_conv, "Fold conv history differs");
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("PASS %s: width=%d rounds=%d\n", fold ? "Record/Fold (all prefixes, rejected NaNs, nonzero state)" : "CPU Record", tokens, rounds);
}

int main(int argc, char ** argv) {
    try {
        ggml_backend_load_all();
        const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
        auto * device = ggml_backend_dev_by_type(cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) return 77;
        auto backend = ggml_backend_dev_init(device, nullptr);
        require(backend != nullptr, "GPU backend initialization failed");
        if (argc == 1 || cpu) {
            for (int width : {1, 2, 3, 4, 5, 6}) check_replay(backend, width, 1, !cpu);
            if (!cpu) check_replay(backend, 6, 1000);
            ggml_backend_free(backend);
            return 0;
        }
        const bool write = argc == 3 && std::string(argv[1]) == "--write";
        const bool check = argc == 3 && std::string(argv[1]) == "--check";
        require(write || check, "usage: gdn-replay-test --write|--check snapshot.bin");
        std::fstream file(argv[2], std::ios::binary | (write ? std::ios::out | std::ios::trunc : std::ios::in));
        require(bool(file), "cannot open snapshot fixture");
        for (int tokens : {1, 2, 3, 17, 512}) {
            const auto result = snapshot(backend, tokens);
            const size_t bytes = result.size() * sizeof(float);
            if (write) {
                file.write(reinterpret_cast<const char *>(result.data()), bytes);
            } else {
                std::vector<float> expected(result.size());
                file.read(reinterpret_cast<char *>(expected.data()), bytes);
                require(bool(file), "truncated snapshot fixture");
                require(std::memcmp(expected.data(), result.data(), bytes) == 0, "FP32 snapshot differs from original kernel");
            }
            std::printf("PASS original snapshots: tokens=%d bytes=%zu\n", tokens, bytes);
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
