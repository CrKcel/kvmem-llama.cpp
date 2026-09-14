#include "kvmem/kvmem_runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

using namespace kvmem;

static int g_fail = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

struct RecordingBackend : KvMemBackend {
    int32_t next = 0;
    std::vector<int32_t> allocs;
    std::vector<int32_t> frees;
    std::vector<std::string> ops;

    int32_t alloc_gpu_slot() override {
        const int32_t s = next++;
        allocs.push_back(s);
        ops.push_back("alloc");
        return s;
    }
    void free_gpu_slot(int32_t slot) override {
        frees.push_back(slot);
        ops.push_back("free");
    }
};

static KvMemRuntimeConfig make_cfg() {
    KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = 32;
    cfg.store.select_budget = 32 * 4;
    cfg.store.sink_blocks = 1;
    cfg.store.recent_blocks = 1;
    cfg.store.estimated_block_bytes = 1024;
    cfg.cpu_bytes = 1024 * 16;
    return cfg;
}

static void test_stage_out_before_stage_in() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    be.ops.clear();
    be.allocs.clear();
    be.frees.clear();

    auto plan = rt.prepare_reselect();
    CHECK(plan.remaps.size() == 4);
    rt.finish_reselect();

    // All GPU frees (evictions) must precede any new alloc.
    bool saw_alloc = false;
    bool order_ok = true;
    for (const auto &op : be.ops) {
        if (op == "alloc") saw_alloc = true;
        if (op == "free" && saw_alloc) order_ok = false;
    }
    CHECK(order_ok);
    CHECK(!be.frees.empty());
}

static void test_high_overlap_skips_stage_in() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    rt.reselect();
    const auto first = rt.last_plan();
    CHECK(first.remaps.size() == 4);

    auto second = rt.prepare_reselect();
    CHECK(second.stage_in.empty());
    CHECK(second.gpu_reused_blocks == 4);
    for (const auto &rm : second.remaps) {
        CHECK(rm.skip || rm.working_k_resident);
    }
    rt.finish_reselect();
}

static void test_pressure_keeps_sink_and_tail() {
    KvMemRuntime rt(make_cfg());
    rt.register_append(32 * 10);
    auto plan = rt.prepare_prefill_pressure();
    CHECK(plan.remaps.size() == 4);
    CHECK(plan.remaps.front().block_id == 0);
    CHECK(plan.remaps.back().block_id == 9);
    rt.finish_reselect();
}

static void test_maybe_offload_evicts_before_stage_in() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    be.ops.clear();
    be.allocs.clear();
    be.frees.clear();

    const uint32_t pool = 32 * 4;
    CHECK(rt.maybe_offload_during_prefill(/*incoming=*/32, /*resident=*/32 * 10, pool));
    CHECK(rt.last_plan().remaps.size() == 4);
    rt.finish_reselect();

    bool saw_alloc = false;
    bool order_ok = true;
    for (const auto &op : be.ops) {
        if (op == "alloc") saw_alloc = true;
        if (op == "free" && saw_alloc) order_ok = false;
    }
    CHECK(order_ok);
    CHECK(!be.frees.empty());

    KvMemRuntime idle(make_cfg());
    idle.register_append(32 * 2);
    CHECK(!idle.maybe_offload_during_prefill(32, 32, 32 * 8));
}

static void test_cpu_full_spills_to_nvme_and_roundtrips() {
    struct MemoryBackend : KvMemBackend {
        uint64_t slot_bytes = 64;
        int32_t next = 0;
        std::map<int32_t, std::vector<uint8_t>> gpu;
        int32_t alloc_gpu_slot() override {
            const int32_t s = next++;
            gpu[s].assign(slot_bytes, static_cast<uint8_t>(s + 1));
            return s;
        }
        void free_gpu_slot(int32_t slot) override { gpu.erase(slot); }
        void copy_block_to_host(uint32_t, int32_t gpu_slot, void *host,
                                uint64_t bytes) override {
            auto it = gpu.find(gpu_slot);
            if (it == gpu.end() || !host) return;
            std::memcpy(host, it->second.data(),
                        static_cast<size_t>(std::min(bytes, slot_bytes)));
        }
        void copy_block_from_host(uint32_t, int32_t gpu_slot, const void *host,
                                  uint64_t bytes) override {
            if (!host) return;
            gpu[gpu_slot].assign(static_cast<const uint8_t *>(host),
                                 static_cast<const uint8_t *>(host) +
                                     static_cast<size_t>(bytes));
        }
    };

    const char *base = std::getenv("TMPDIR");
    if (!base) base = "/tmp";
    const std::string dir = std::string(base) + "/kvmem_p32_nvme";

    MemoryBackend be;
    KvMemRuntimeConfig cfg = make_cfg();
    cfg.store.estimated_block_bytes = 64;
    cfg.cpu_bytes = 64 * 2;          // two CPU slots
    cfg.nvme_bytes = 64 * 8;
    cfg.nvme_dir = dir;
    KvMemRuntime rt(cfg, &be);
    CHECK(rt.cpu_tier() && rt.cpu_tier()->enabled());
    CHECK(rt.nvme_tier() && rt.nvme_tier()->enabled());

    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }

    auto plan = rt.prepare_prefill_pressure();
    CHECK(plan.remaps.size() == 4);
    rt.finish_reselect();

    uint32_t on_nvme = 0;
    for (const auto &b : rt.store().blocks()) {
        if (b.nvme_slot >= 0 || b.tier == KvTier::SSD) {
            ++on_nvme;
        }
    }
    CHECK(on_nvme > 0);

    // Bring a spilled middle block back via retrieval scores.
    std::vector<double> scores(10, 0.0);
    scores[4] = 100.0;
    rt.store().set_retrieval_scores(scores);
    auto back = rt.prepare_reselect();
    bool staged = false;
    for (uint32_t id : back.stage_in) {
        if (id == 4) staged = true;
    }
    CHECK(staged);
    rt.finish_reselect();
    CHECK(rt.store().blocks()[4].gpu_slot >= 0);
    const auto &payload = be.gpu[rt.store().blocks()[4].gpu_slot];
    CHECK(!payload.empty());
    // Original GPU slot for block 4 was 4, filled with byte 5.
    CHECK(payload[0] == 5);
}

static void test_selection_preview_and_resident_commit() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 3 + 7);
    for (uint32_t id = 0; id < rt.store().block_count(); ++id) {
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    be.ops.clear();
    const auto before = rt.store().blocks();
    const auto selected = rt.preview_reselect();
    CHECK(selected.size() == 4);
    CHECK(be.ops.empty());
    for (uint32_t id = 0; id < before.size(); ++id) {
        const auto & after = rt.store().blocks()[id];
        CHECK(after.gpu_slot == before[id].gpu_slot);
        CHECK(after.in_working_set == before[id].in_working_set);
        CHECK(after.baked_pos == before[id].baked_pos);
    }
    CHECK(!rt.commit_resident_selection({0, 1, 2}));
    CHECK(!rt.store().blocks()[0].in_working_set);
    CHECK(rt.commit_resident_selection(selected));
    CHECK(be.ops.empty());
    CHECK(rt.last_plan().total_window_tokens == 103);
    CHECK(rt.store().blocks().back().n_tokens == 7);
    auto pending = rt.prepare_selection(selected);
    CHECK(!rt.commit_resident_selection(selected));
    rt.finish_reselect();
    CHECK(rt.commit_resident_selection(selected));
}

int main() {
    test_selection_preview_and_resident_commit();
    test_stage_out_before_stage_in();
    test_high_overlap_skips_stage_in();
    test_pressure_keeps_sink_and_tail();
    test_maybe_offload_evicts_before_stage_in();
    test_cpu_full_spills_to_nvme_and_roundtrips();
    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
