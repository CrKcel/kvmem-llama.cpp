#pragma once

// Opt-in KVMEM_* diagnostic output.
//
// Adapted from dockylf's PR #9. Preserve machine-readable stderr records for
// benchmark scripts while keeping normal server output on the common logger.
//
// The gate lives here because it serves both sides: the KVMem tools (tools/) and
// the llama.cpp adapter (src/adapter/). Those are separate modules with separate
// copies of the static below, so the *environment variable* is the authoritative
// switch -- the tool also mirrors --kvmem-trace into it for exactly that reason.
//
//   KVMEM_TRACE=1        environment variable, as the benchmark scripts set it
//   --kvmem-trace        same thing, per invocation (the tool sets the env var)
//   --no-kvmem-trace     force off even when the environment sets it
//
// Errors and the independently enabled KVMEM_PERF counters bypass this gate.
// Configure only during startup, before inference threads or adapter creation.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// One shared flag per module: the static is function-local inside an inline
// function, so every translation unit that includes this header in the same
// module sees the same instance.
inline bool & kvmem_diag_state() {
    static bool on = [] {
        const char * e = std::getenv("KVMEM_TRACE");
        // Unset, empty and "0" all mean off.
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return on;
}

inline void kvmem_diag_set(bool on) {
    kvmem_diag_state() = on;
}

inline bool kvmem_diag_enabled() {
    return kvmem_diag_state();
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline void kvmem_diag(const char * fmt, ...) {
    if (!kvmem_diag_enabled()) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}
