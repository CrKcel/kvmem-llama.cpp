#pragma once

#include <chrono>
#include <cstdint>

struct kvmem_prefill_perf {
    double select_checkpoint_ms = 0;
    double save_ms = 0;
    double restore_ms = 0;
    double mean_ms = 0;
    double carry_ms = 0;
    double first_ms = 0;
    double replay_ms = 0;
    double retrieval_ms = 0;
    double decision_ms = 0;
    double target_ms = 0; // diagnostic mode only: completed main-model calls
    double draft_ms = 0;  // diagnostic mode only: completed MTP process calls
    uint32_t saves = 0;
    uint32_t restores = 0;
    uint32_t shared = 0;
    uint32_t restore_skips = 0;
};

class kvmem_scoped_ms {
public:
    explicit kvmem_scoped_ms(double & dst) : dst_(dst), begin_(std::chrono::steady_clock::now()) {}
    ~kvmem_scoped_ms() {
        dst_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin_).count();
    }
    kvmem_scoped_ms(const kvmem_scoped_ms &) = delete;
    kvmem_scoped_ms & operator=(const kvmem_scoped_ms &) = delete;
private:
    double & dst_;
    std::chrono::steady_clock::time_point begin_;
};
