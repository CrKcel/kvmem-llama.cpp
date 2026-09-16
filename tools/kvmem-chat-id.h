#pragma once

#include <atomic>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

// Random process namespace prevents reuse when a client continues after restart;
// the atomic sequence guarantees distinct requests within this process.
inline std::string kvmem_chat_request_id() {
    static const std::string process = [] {
        std::random_device random;
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (int i = 0; i < 4; ++i) out << std::setw(8) << uint32_t(random());
        return out.str();
    }();
    static std::atomic<uint64_t> sequence {0};
    return process + "_" + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

inline std::string kvmem_chat_tool_id(const std::string & request_id, uint64_t index) {
    return "call_" + request_id + "_" + std::to_string(index);
}
