#pragma once

#include "kvmem-chat-sampling.h"
#include "httplib.h"
#include <filesystem>

inline bool kvmem_output_limit(const nlohmann::json & body, int limit, int & value, std::string & error) {
    const char * key = body.contains("max_tokens") ? "max_tokens" : "max_completion_tokens";
    if (body.contains(key)) {
        const auto & n = body[key];
        if (!n.is_number_integer() || n.get<double>() < -1 || n.get<double>() > limit || n == 0) {
            error = std::string(key) + " must be -1 (server default) or an integer in 1.." + std::to_string(limit);
            return false;
        }
        if (n != -1) value = n.get<int>();
    }
    if (value < 1) value = limit;
    value = std::min(value, limit);
    return true;
}

inline nlohmann::json kvmem_ui_sampling(bool thinking, const nlohmann::json & overrides) {
    auto sp = kvmem_chat_sampling_defaults(thinking);
    std::string error;
    if (!kvmem_chat_sampling_override(overrides, sp, error)) throw std::runtime_error(error);
    return {{"temperature", sp.temp}, {"top_p", sp.top_p}, {"top_k", sp.top_k}, {"min_p", sp.min_p},
            {"presence_penalty", sp.penalty_present}, {"frequency_penalty", sp.penalty_freq},
            {"repeat_penalty", sp.penalty_repeat}};
}

inline bool kvmem_mount_ui(httplib::Server & server, const std::string & requested, bool disabled, const char * argv0) {
    namespace fs = std::filesystem;
    if (disabled) return true;
    std::error_code ec;
    auto binary = fs::read_symlink("/proc/self/exe", ec);
    if (ec) binary = fs::absolute(argv0);
    const auto directory = requested.empty() ? binary.parent_path().parent_path() / "share/kvmem/ui" : fs::path(requested);
    if (!fs::is_regular_file(directory / "index.html")) {
        if (requested.empty()) return true;
        fprintf(stderr, "UI directory has no index.html: %s\n", directory.string().c_str());
        return false;
    }
    if (!server.set_mount_point("/", directory.string())) return false;
    server.set_file_request_handler([](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Cache-Control", req.path.find("/_app/immutable/") == 0 ?
                       "public, max-age=31536000, immutable" : "no-cache");
    });
    fprintf(stderr, "KVMEM_UI directory=%s\n", directory.string().c_str());
    return true;
}
