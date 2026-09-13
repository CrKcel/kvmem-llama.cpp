// Usage: CUDA_VISIBLE_DEVICES=... build/bin/kvmem-mtp-kv-test model-mtp.gguf
#include "kvmem-spec.h"
#include "llama-kvmem-hooks.h"
#include "llama-memory-kvmem-mtp.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

struct test_spec_session : kvmem_spec_session {
    ~test_spec_session() { kvmem_spec_stop(*this); }
};

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static std::vector<uint8_t> pattern(size_t size, unsigned seed) {
    std::vector<uint8_t> bytes(size);
    for (auto & byte : bytes) {
        seed = seed * 1664525u + 1013904223u;
        byte = uint8_t(seed >> 24);
    }
    return bytes;
}

static void compare(ggml_tensor * tensor, const std::vector<uint8_t> & expected) {
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    require(actual == expected, "MTP KV bytes differ (including untouched rows)");
}

static void check_transfers(llama_memory_kvmem_mtp & mtp, ggml_type type) {
    auto * kv = mtp.get_kv();
    require(kv->get_layer_ids().size() == 1, "test requires one MTP layer");
    const int il = kv->get_layer_ids().front();
    ggml_tensor * tensors[] = {kv->get_k_storage(il), kv->get_v_storage(il)};
    const uint32_t block = mtp.target()->block_tokens();
    auto & store = mtp.target()->runtime().store();
    store.register_append(2 * block + 13);
    for (uint32_t id = 0; id < 3; ++id) {
        store.set_block_gpu_slot(id, id);
        mtp.occupy_block(id);
    }

    std::vector<uint8_t> original[2];
    for (int i = 0; i < 2; ++i) {
        auto * t = tensors[i];
        require(t && t->type == type, "draft did not inherit requested KV type");
        require(!ggml_backend_buffer_is_host(t->buffer), "test requires GPU KV");
        original[i] = pattern(ggml_nbytes(t), 42 + i);
        ggml_backend_tensor_set(t, original[i].data(), 0, original[i].size());
    }

    // Save at nonzero offsets, destroy GPU contents, then restore to new slots.
    for (uint32_t id = 0; id < 3; ++id) {
        mtp.on_stage_out(id);
        store.set_block_gpu_slot(id, 3 + 2 * id);
    }
    mtp.harvest_flush();
    std::vector<uint8_t> expected[2];
    for (int i = 0; i < 2; ++i) {
        expected[i].assign(original[i].size(), 0xa5);
        ggml_backend_tensor_set(tensors[i], expected[i].data(), 0, expected[i].size());
        const size_t row = ggml_row_size(type, tensors[i]->ne[0]);
        for (uint32_t id = 0; id < 3; ++id) {
            const size_t size = store.blocks()[id].n_tokens * row;
            std::copy_n(original[i].data() + id * block * row, size,
                        expected[i].data() + (3 + 2 * id) * block * row);
        }
    }
    mtp.follow_retrieval();
    for (int i = 0; i < 2; ++i) {
        compare(tensors[i], expected[i]);
    }
    for (const auto & b : store.blocks()) {
        require(mtp.slot_holds(b.gpu_slot, b.orig_pos_start), "restored slot position differs");
    }

    // A cycle must preserve unread sources and the tail of the partial block.
    const llama_memory_kvmem_mtp::LayoutMove moves[] = {
        {3, 5, block}, {5, 7, block}, {7, 3, 13},
    };
    for (int i = 0; i < 2; ++i) {
        const auto before = expected[i];
        const size_t row = ggml_row_size(type, tensors[i]->ne[0]);
        for (const auto & move : moves) {
            std::copy_n(before.data() + move.src_slot * block * row, move.n_tokens * row,
                        expected[i].data() + move.dst_slot * block * row);
        }
    }
    require(mtp.layout_d2d(moves, 3), "MTP CUDA layout failed");
    for (int i = 0; i < 2; ++i) {
        compare(tensors[i], expected[i]);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s model-mtp.gguf (requires CUDA)\n", argv[0]);
        return argc == 1 ? 77 : 1;
    }
    try {
        llama_backend_init();
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        mp.load_mtp = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
                llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");

        llama_kvmem_params kp{};
        kp.enabled = true;
        kp.block_tokens = 32;
        kp.budget = 256;
        kp.gen_reserve = 256;
        kp.query_begin = kp.query_end = kp.force_pos = -1;
        llama_kvmem_set_params(&kp);
        struct cache_case { ggml_type target; ggml_type draft; };
        const cache_case cases[] = {
            {GGML_TYPE_Q8_0, GGML_TYPE_COUNT},
            {GGML_TYPE_Q5_0, GGML_TYPE_COUNT},
            {GGML_TYPE_Q4_0, GGML_TYPE_COUNT},
            {GGML_TYPE_F16,  GGML_TYPE_COUNT},
            {GGML_TYPE_Q5_0, GGML_TYPE_F16},
            {GGML_TYPE_Q5_0, GGML_TYPE_Q8_0},
            {GGML_TYPE_Q5_0, GGML_TYPE_Q5_0},
        };
        for (const auto & test : cases) {
            const auto type = test.target;
            const auto draft_type = test.draft == GGML_TYPE_COUNT ? type : test.draft;
            auto cp = llama_context_default_params();
            cp.n_ctx = 1024;
            cp.n_batch = cp.n_ubatch = 128;
            cp.n_seq_max = 1;
            cp.n_rs_seq = 2;
            cp.n_outputs_max = cp.n_outputs_max_per_seq = 3;
            cp.type_k = cp.type_v = type;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
                    llama_init_from_model(model.get(), cp), llama_free);
            require(bool(ctx), "target context init failed");
            kvmem_spec_opts opts;
            opts.kvmem_enabled = true;
            opts.n_ctx = cp.n_ctx;
            opts.n_batch = opts.n_ubatch = cp.n_batch;
            opts.type_k = opts.type_v = type;
            opts.draft_type = test.draft;
            // Stop the follower before freeing its target, including on failure.
            test_spec_session sess;
            require(kvmem_spec_start(sess, model.get(), ctx.get(), opts), "MTP init failed");
            auto * mtp = dynamic_cast<llama_memory_kvmem_mtp *>(llama_get_memory(sess.ctx_dft));
            require(mtp != nullptr, "missing KVMem MTP follower");
            require(mtp->target()->get_kv()->type_k() == type && mtp->target()->get_kv()->type_v() == type,
                    "draft override changed target KV types");
            check_transfers(*mtp, draft_type);
            std::printf("PASS target=%s draft=%s override=%d: GPU save/restore, cycle, partial block\n",
                        ggml_type_name(type), ggml_type_name(draft_type), test.draft != GGML_TYPE_COUNT);
        }
        llama_kvmem_set_params(nullptr);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    llama_backend_free();
    return 0;
}
