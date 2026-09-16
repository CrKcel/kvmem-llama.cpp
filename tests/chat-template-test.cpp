#include "kvmem-chat-template.h"
#include "chat.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

static void test_options() {
    using json = nlohmann::json;
    bool thinking = true;
    std::map<std::string, std::string> kwargs;
    std::string err;
    CHECK(kvmem_chat_template_override({{"chat_template_kwargs", {{"reasoning_effort", "xhigh"}, {"flag", true}, {"count", 2}, {"nested", {{"x", 7}}}}}}, thinking, kwargs, err));
    CHECK(kwargs["flag"] == "true" && kwargs["count"] == "2" && json::parse(kwargs["nested"])["x"] == 7);
    CHECK(kvmem_chat_template_override({{"reasoning_effort", "low"}}, thinking, kwargs, err));
    CHECK(kwargs["reasoning_effort"] == "\"low\"");
    CHECK(kvmem_chat_template_override({{"reasoning_effort", "default"}}, thinking, kwargs, err));
    CHECK(!kwargs.count("reasoning_effort") && thinking);
    CHECK(kvmem_chat_template_override({{"enable_thinking", true}, {"chat_template_kwargs", {{"enable_thinking", true}, {"reasoning_effort", "low"}}}, {"reasoning_effort", "none"}}, thinking, kwargs, err));
    CHECK(!thinking && kwargs["enable_thinking"] == "false" && !kwargs.count("reasoning_effort"));
    CHECK(kvmem_chat_template_override({{"enable_thinking", true}}, thinking, kwargs, err));
    CHECK(thinking && kwargs["enable_thinking"] == "true");
    for (const auto & body : {json{{"chat_template_kwargs", "{}"}}, json{{"chat_template_kwargs", json::array()}},
                             json{{"enable_thinking", "false"}}, json{{"reasoning_effort", 2}},
                             json{{"chat_template_kwargs", {{"enable_thinking", "false"}}}},
                             json{{"chat_template_kwargs", {{"reasoning_effort", false}}}}}) {
        const auto saved = kwargs;
        CHECK(!kvmem_chat_template_override(body, thinking, kwargs, err));
        CHECK(thinking && kwargs == saved && !err.empty());
    }
}

static common_chat_params render(const common_chat_templates * tmpls, const nlohmann::json & options) {
    common_chat_templates_inputs inputs;
    common_chat_msg user;
    user.role = "user"; user.content = "Hello";
    inputs.messages = {user};
    inputs.enable_thinking = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    std::string err;
    CHECK(kvmem_chat_template_override(options, inputs.enable_thinking, inputs.chat_template_kwargs, err));
    return common_chat_templates_apply(tmpls, inputs);
}

static void test_render() {
    const std::string source = R"JINJA({% if reasoning_effort is defined %}effort={{ reasoning_effort }};{% endif %}{% if flag is defined and flag is true %}typed-bool;{% endif %}{% if count is defined and count == 2 %}typed-number;{% endif %}{% for message in messages %}{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %})JINJA";
    auto tmpls = common_chat_templates_init(nullptr, source);
    const auto chat = render(tmpls.get(), {{"reasoning_effort", "low"}, {"chat_template_kwargs", {{"flag", true}, {"count", 2}}}});
    CHECK(chat.prompt.find("effort=low;typed-bool;typed-number;") != std::string::npos);
    CHECK(chat.prompt.find("Hello") != std::string::npos);
    CHECK(render(tmpls.get(), {{"reasoning_effort", "default"}}).prompt.find("effort=") == std::string::npos);
}

int main(int argc, char ** argv) {
    test_options();
    test_render();
    // Optional real GSQ model template extracted from its GGUF; no weights/GPU needed.
    if (argc > 1) {
        std::ifstream f(argv[1]); CHECK(bool(f));
        const std::string source((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto tmpls = common_chat_templates_init(nullptr, source);
        const auto low = render(tmpls.get(), {{"reasoning_effort", "low"}});
        const auto medium = render(tmpls.get(), {{"reasoning_effort", "medium"}});
        const auto high = render(tmpls.get(), {{"reasoning_effort", "xhigh"}});
        CHECK(low.prompt.find("Keep your thinking brief and focused") != std::string::npos);
        CHECK(high.prompt.find("validate key assumptions") != std::string::npos);
        CHECK(low.prompt != medium.prompt && medium.prompt != high.prompt);
        CHECK(render(tmpls.get(), nlohmann::json::object()).prompt == high.prompt);
        const auto off = render(tmpls.get(), {{"reasoning_effort", "none"}});
        CHECK(off.prompt.find("Reasoning effort is set to") == std::string::npos);
        bool rejected = false;
        try { render(tmpls.get(), {{"reasoning_effort", "high"}}); }
        catch (const std::exception &) { rejected = true; }
        CHECK(rejected);
        std::puts("actual GSQ low/medium/xhigh/default/none and invalid effort PASS");
    }
    std::puts("chat template options and native Jinja rendering PASS");
}
