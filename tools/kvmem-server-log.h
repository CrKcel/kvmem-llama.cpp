#pragma once

#include "log.h"
#include <chrono>

// Owned by the inference slot; HTTP status readers use kvmem_server_progress.
// Rates count newly evaluated prompt rows, excluding cached and replayed rows.
class kvmem_server_log {
    using clock = std::chrono::steady_clock;
    clock::time_point start_, last_;
    int last_generated_ = 0;
    int processed_ = 0, total_ = 0;
public:
    void start() { start_ = last_ = clock::now(); last_generated_ = processed_ = total_ = 0; }
    void start_generation() { start(); }
    void prefilled(int rows, int remaining) {
        if (processed_ == 0) total_ = remaining;
        processed_ += rows;
        const auto now = clock::now();
        if (now - last_ < std::chrono::seconds(3)) return;
        last_ = now;
        const double elapsed = std::chrono::duration<double>(now - start_).count();
        LOG_INF("slot   prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                processed_, total_ > 0 ? double(processed_) / total_ : 1.0,
                elapsed, elapsed > 0 ? processed_ / elapsed : 0.0);
    }
    void generated(int count) {
        const auto now = clock::now();
        if (count < 100 || now - last_ < std::chrono::seconds(3)) return;
        const double elapsed = std::chrono::duration<double>(now - start_).count();
        const double window = std::chrono::duration<double>(now - last_).count();
        LOG_INF("slot   n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n",
                count, count / elapsed, (count - last_generated_) / window);
        last_ = now;
        last_generated_ = count;
    }
};
