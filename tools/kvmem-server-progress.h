#pragma once

#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdint>
#include <mutex>

// HTTP status readers never acquire the model's long-running inference lock.
class kvmem_server_progress {
public:
    struct state {
        bool busy = false;
        uint64_t task = 0;
        int prompt = 0, processed = 0, cached = 0, generated = 0, limit = 0;
        nlohmann::json params = nlohmann::json::object();
    };
    void start() {
        std::lock_guard<std::mutex> lock(mu_);
        const auto task = data_.task + 1;
        data_ = state{};
        data_.task = task;
        data_.busy = true;
    }
    void prompt(int count, int limit, nlohmann::json params) {
        std::lock_guard<std::mutex> lock(mu_);
        data_.prompt = count;
        data_.limit = limit;
        data_.params = std::move(params);
    }
    void prefilled(int cached) {
        std::lock_guard<std::mutex> lock(mu_);
        data_.cached = std::clamp(cached, 0, data_.prompt);
        data_.processed = data_.prompt - data_.cached;
    }
    void generated(int count) {
        std::lock_guard<std::mutex> lock(mu_);
        data_.generated = count;
    }
    void finish() {
        std::lock_guard<std::mutex> lock(mu_);
        data_.busy = false;
    }
    state snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        return data_;
    }
private:
    mutable std::mutex mu_;
    state data_;
};

// Finish the public status before releasing the slot, including error/disconnect paths.
class kvmem_server_slot_guard {
public:
    kvmem_server_slot_guard(std::mutex & mu, kvmem_server_progress & progress)
        : lock_(mu), progress_(progress) { progress_.start(); }
    ~kvmem_server_slot_guard() { if (lock_.owns_lock()) unlock(); }
    void unlock() {
        progress_.finish();
        lock_.unlock();
    }
private:
    std::unique_lock<std::mutex> lock_;
    kvmem_server_progress & progress_;
};
