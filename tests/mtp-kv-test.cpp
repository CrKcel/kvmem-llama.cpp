// Usage: CUDA_VISIBLE_DEVICES=... build/bin/kvmem-mtp-kv-test model-mtp.gguf
#include "kvmem-spec.h"
#include "llama-kvmem-hooks.h"
#include "llama-memory-kvmem-mtp.h"
#include "llama-kvmem-stagein.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

struct test_spec_session : kvmem_spec_session {
    ~test_spec_session() { kvmem_spec_stop(*this); }
};

struct kvmem_transfer_test_access {
    static void save(llama_memory_kvmem & mem, uint32_t id) { mem.harvest_gpu_v(id); }
    static void flush(llama_memory_kvmem & mem) { mem.harvest_gpu_v_commit(); }
    static void restore(llama_memory_kvmem & mem, uint32_t id) { mem.write_block_to_gpu(id); }
    static bool layout(llama_memory_kvmem & mem) { return mem.layout_gpu_slots_by_orig_pos(); }
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
    const uint32_t nt = 2 * block + 13;
    std::vector<llama_pos> positions(4*nt, 0), logical(nt);
    for (uint32_t i = 0; i < nt; ++i) {
        logical[i] = i;
        positions[i] = 17;
        positions[nt+i] = 17 + i/8;
        positions[2*nt+i] = 17 + i%8;
    }
    llama_ubatch ub{};
    ub.n_tokens = nt;
    ub.n_pos = 4;
    ub.pos = positions.data();
    ub.logical_pos = logical.data();
    llama_kv_cache::slot_info_vec_t slots;
    require(mtp.target()->prepare_ubatches({ub}, nt, slots), "visual row slot preparation failed");
    for (uint32_t id = 0; id < 3; ++id) mtp.occupy_block(id);

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
        if (id + 1 < 3) require(mtp.slot_holds(id + 1, (id + 1)*block), "stage-out removed patches sharing t in another block");
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
        const auto & cells = kv->get_cells(0);
        for (uint32_t i = 0; i < b.n_tokens; ++i) {
            const uint32_t cell = b.gpu_slot * block + i;
            const uint32_t row = b.orig_pos_start + i;
            require(cells.pos_get(cell) == positions[row], "restored temporal position differs");
            const auto & ext = cells.ext_get(cell);
            require(ext.y == positions[nt+row] && ext.x == positions[2*nt+row], "restored spatial position differs");
            require(ext.logical_pos == (llama_pos) row, "restored logical row differs");
        }
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

static void check_target_transfers(llama_memory_kvmem & mem, ggml_type type) {
    auto * kv = mem.get_kv();
    auto & store = mem.runtime().store();
    const uint32_t block = mem.block_tokens();
    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<uint8_t>> original, expected;
    for (int il : kv->get_layer_ids()) {
        tensors.push_back(kv->get_k_storage(il));
        tensors.push_back(kv->get_v_storage(il));
    }
    for (uint32_t id = 0; id < 3; ++id) mem.occupy_in(kv, id);
    for (auto * tensor : tensors) {
        original.push_back(pattern(ggml_nbytes(tensor), 123 + tensors.size()));
        ggml_backend_tensor_set(tensor, original.back().data(), 0, original.back().size());
    }
    for (uint32_t id = 0; id < 3; ++id) kvmem_transfer_test_access::save(mem, id);
    kvmem_transfer_test_access::flush(mem);
    for (size_t i = 0; i < tensors.size(); ++i) {
        expected.emplace_back(original[i].size(), 0x5a);
        ggml_backend_tensor_set(tensors[i], expected[i].data(), 0, expected[i].size());
        const size_t row = ggml_row_size(type, tensors[i]->ne[0]);
        for (uint32_t id = 0; id < 3; ++id) {
            std::copy_n(original[i].data() + (3 + 2*id)*block*row, store.blocks()[id].n_tokens*row,
                        expected[i].data() + (2 + 2*id)*block*row);
        }
    }
    for (uint32_t id = 0; id < 3; ++id) {
        require(kv->seq_rm_logical(0, id*block, (id+1)*block), "target logical removal failed");
        if (id < 2) require(!kv->get_cells(0).is_empty((5 + 2*id)*block), "target removed another block sharing t");
        store.set_block_gpu_slot(id, 2 + 2*id);
        kvmem_transfer_test_access::restore(mem, id);
    }
    require(kvmem_stagein_flush(nullptr, nullptr, nullptr, nullptr), "target stage-in flush failed");
    kvmem_stagein_sync();
    for (size_t i = 0; i < tensors.size(); ++i) compare(tensors[i], expected[i]);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto before = expected[i];
        const size_t row = ggml_row_size(type, tensors[i]->ne[0]);
        for (uint32_t id = 0; id < 3; ++id) {
            std::copy_n(before.data() + (2 + 2*id)*block*row, store.blocks()[id].n_tokens*row,
                        expected[i].data() + id*block*row);
        }
    }
    require(kvmem_transfer_test_access::layout(mem), "target D2D layout failed");
    for (size_t i = 0; i < tensors.size(); ++i) compare(tensors[i], expected[i]);
    for (const auto & b : store.blocks()) for (uint32_t i = 0; i < b.n_tokens; ++i) {
        const auto cell = b.gpu_slot*block + i;
        const auto logical = b.orig_pos_start + i;
        const auto & cells = kv->get_cells(0);
        const auto & ext = cells.ext_get(cell);
        require(cells.pos_get(cell) == 17 && ext.logical_pos == (llama_pos) logical
                && ext.y == 17 + (llama_pos) logical/8 && ext.x == 17 + (llama_pos) logical%8,
                "target restored position metadata differs");
    }
}

static void check_batch_inputs(llama_model * model) {
    constexpr int n = 9, width = 3;
    std::vector<float> input(n*width), hidden(n*width);
    std::vector<llama_pos> pos(4*n, 12), logical(n);
    for (int i = 0; i < n; ++i) {
        logical[i] = 100 + i;
        pos[n+i] = i/3;
        pos[2*n+i] = i%3;
        for (int j = 0; j < width; ++j) {
            input[i*width+j] = i*10+j;
            hidden[i*width+j] = -i*10-j-1;
        }
    }
    llama_batch batch{};
    batch.n_tokens = n;
    batch.embd = input.data();
    batch.embd_nextn = hidden.data();
    batch.pos = pos.data();
    batch.logical_pos = logical.data();
    llama_batch_allocr allocator(4);
    require(allocator.init(batch, *llama_model_get_vocab(model), nullptr, width, 1, true), "visual batch init failed");
    allocator.split_reset();
    int seen = 0;
    while (seen < n) {
        auto ub = allocator.split_simple(4);
        require(ub.n_tokens > 0, "visual batch split failed");
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            int row = ub.logical_pos[i] - 100;
            require(row == seen++, "split lost logical row order");
            for (int j = 0; j < width; ++j) {
                require(ub.embd[i*width+j] == input[row*width+j], "split lost input embedding");
                require(ub.embd_nextn[i*width+j] == hidden[row*width+j], "split lost MTP hidden input");
            }
            for (int j = 0; j < 4; ++j) require(ub.pos[j*ub.n_tokens+i] == pos[j*n+row], "split lost M-RoPE position plane");
        }
    }
    std::vector<int32_t> n_seq(n, 1);
    std::vector<llama_seq_id> seq(n);
    std::vector<llama_seq_id *> ids(n);
    for (int i = 0; i < n; ++i) { seq[i] = i%2; ids[i] = &seq[i]; }
    batch.n_seq_id = n_seq.data();
    batch.seq_id = ids.data();
    llama_batch_allocr reordered(4);
    require(reordered.init(batch, *llama_model_get_vocab(model), nullptr, width, 2, true), "reordered batch init failed");
    reordered.split_reset();
    std::vector<bool> visited(n, false);
    for (int count = 0; count < n;) {
        const auto ub = reordered.split_seq(4);
        require(ub.n_tokens > 0, "reordered batch split failed");
        for (uint32_t i = 0; i < ub.n_tokens; ++i, ++count) {
            const int row = ub.logical_pos[i] - 100;
            require(row >= 0 && row < n && !visited[row], "reorder duplicated or lost logical row");
            visited[row] = true;
            require(ub.seq_id[i][0] == seq[row], "reordered sequence differs");
            for (int j = 0; j < width; ++j) {
                require(ub.embd[i*width+j] == input[row*width+j], "reorder lost visual input");
                require(ub.embd_nextn[i*width+j] == hidden[row*width+j], "reorder lost hidden input");
            }
            for (int j = 0; j < 4; ++j) require(ub.pos[j*ub.n_tokens+i] == pos[j*n+row], "reorder lost M-RoPE plane");
        }
    }
    std::puts("PASS visual batch split: logical rows, M-RoPE, separate hidden input");
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
        check_batch_inputs(model.get());

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
            check_target_transfers(*mtp->target(), type);
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
