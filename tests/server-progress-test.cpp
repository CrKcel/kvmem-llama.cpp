#include "kvmem-server-progress.h"
#include <atomic>
#include <cstdlib>
#include <thread>

static void check(bool condition) { if (!condition) std::abort(); }

int main() {
    kvmem_server_progress progress;
    std::mutex slot;
    std::atomic<bool> done{false};
    std::thread reader([&] {
        while (!done) {
            const auto s = progress.snapshot();
            check(s.cached >= 0 && s.cached <= s.prompt);
            check(s.generated >= 0 && s.generated <= s.limit);
            if (s.processed) check(s.processed + s.cached == s.prompt);
        }
    });
    for (int turn = 0; turn < 200; ++turn) {
        kvmem_server_slot_guard guard(slot, progress);
        check(progress.snapshot().busy);
        check(progress.snapshot().generated == 0);
        progress.prompt(100, 64, {{"max_tokens", 64}});
        progress.prefilled(80);
        for (int n = 1; n <= 64; ++n) progress.generated(n);
    }
    done = true;
    reader.join();
    check(!progress.snapshot().busy);
    check(progress.snapshot().task == 200);
    try {
        kvmem_server_slot_guard guard(slot, progress);
        throw 1;
    } catch (int) {}
    check(!progress.snapshot().busy);
    check(slot.try_lock());
    slot.unlock();
}
