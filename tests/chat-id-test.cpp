#include "kvmem-chat-id.h"
#include "chat.h"

#include <cstdio>
#include <cstdlib>
#include <set>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

int main() {
    std::set<std::string> requests, calls;
    for (int request = 0; request < 1000; ++request) {
        const auto id = kvmem_chat_request_id();
        CHECK(requests.insert(id).second);
        std::vector<std::string> cached;
        unsigned next = 0;
        auto generate = [&] { return kvmem_chat_tool_id(id, ++next); };
        common_chat_msg previous;
        // Repeated partial parses plus final parse must not change existing IDs.
        for (int phase = 0; phase < 3; ++phase) {
            common_chat_msg msg;
            msg.role = "assistant";
            msg.tool_calls.resize(phase == 0 ? 1 : 2);
            msg.tool_calls[0].name = "read";
            msg.tool_calls[0].arguments = phase == 0 ? "{\"path\":" : "{\"path\":\"a.txt\"}";
            if (phase > 0) {
                msg.tool_calls[1].name = "write";
                msg.tool_calls[1].arguments = "{}";
            }
            msg.set_tool_call_ids(cached, generate);
            if (phase == 0) CHECK(calls.insert(msg.tool_calls[0].id).second);
            if (phase > 0) CHECK(msg.tool_calls[0].id == previous.tool_calls[0].id);
            if (phase == 1) CHECK(calls.insert(msg.tool_calls[1].id).second);
            if (phase == 2) {
                CHECK(msg.tool_calls[1].id == previous.tool_calls[1].id);
                CHECK(common_chat_msg_diff::compute_diffs(previous, msg).empty());
            }
            previous = msg;
        }
        CHECK(next == 2);
        // Non-streaming uses the same request-scoped assignment.
        common_chat_msg complete = previous;
        for (auto & call : complete.tool_calls) call.id.clear();
        cached.clear(); next = 0;
        complete.set_tool_call_ids(cached, generate);
        CHECK(complete == previous);
    }
    std::puts("request/tool ID uniqueness and partial/final stability PASS");
}
