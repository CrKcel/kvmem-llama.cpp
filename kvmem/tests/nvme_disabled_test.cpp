#include "kvmem/kvmem_runtime.hpp"
#include "kvmem/raw_kv_store.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>

template<class F> bool rejects(F f) {
    try { f(); }
    catch (const std::runtime_error & e) {
        return std::string(e.what()).find("NVMe offload is disabled") != std::string::npos;
    }
    return false;
}

int main() {
    // Reject requested storage even when the directory/slot config is empty;
    // otherwise a caller could accidentally believe its KV was on disk.
    kvmem::NvmeKvTierConfig tier;
    tier.total_bytes = 4096;
    kvmem::KvMemRuntimeConfig runtime;
    runtime.nvme_bytes = 4096;
    kvmem::RawKvStoreConfig raw;
    raw.nvme_bytes = 4096;
    if (!rejects([&] { kvmem::NvmeKvTier x(tier); }) ||
        !rejects([&] { kvmem::KvMemRuntime x(runtime); }) ||
        !rejects([&] { kvmem::RawKvStore x(raw); })) return 1;
    kvmem::NvmeKvTier inactive({});
    if (inactive.enabled() || !rejects([&] { inactive.write_block(0, nullptr, 0); })) return 1;
    std::puts("NVMe requests rejected; memory-only tests run separately");
}
