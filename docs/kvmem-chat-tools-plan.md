# KVMem chat：工具调用与 llama.cpp 协议能力

**状态：** T1–T5 已进 **v0.10.0**（0.8B 门 + 27B 60k Q4/IQ3 多轮工具）。下一步 T6（可选 json_schema）。不改 FA，不窗口化 pos，不并进上游 `llama-server`。  
**前提：** 阶段 A–C 已在独立 `llama-kvmem-server` 上（packed KV、decode mean-K、进程内 prefix reuse）。工具调用是 **chat 协议层**，不是 KV 后端。

---

## 目标

让 `llama-kvmem-server` 的 `/v1/chat/completions` 按 llama.cpp 同款路径支持：

1. OpenAI `tools` / `tool_choice` / 历史里的 `tool_calls` 与 `role=tool`
2. 模板生成的 **tool-call grammar**（`common_sampler`）
3. 输出解析成 `choices[0].message.tool_calls`，`finish_reason=tool_calls`
4. 流式 `delta.tool_calls`
5. 与现有 retrieval + prefix reuse 共存（客户端仍带完整 messages）

服务端 **不执行** 工具。执行是客户端的事：拿到 `tool_calls` → 跑函数 → 下一轮把 `role=tool` 塞回 messages。

## 非目标

- 接入上游 `llama-server` 核心 / 多 slot / continuous batching
- embeddings、rerank、mmproj（图/音/视频）
- 阶段 D 落盘
- 自己发明 tool JSON 格式或手写 Qwen parser
- 把「0.8B 会不会主动选对工具」当 GO/NO-GO（格式与协议才是）

## 为何不并 llama-server

KVMem 的 retrieval pin、query replay、GDN ckpt、decode mean-K、MTP follow 都挂在薄服务的单 slot 循环上。并进 llama-server 等于重做 C，且原计划明确非目标。

llama.cpp 的工具调用已经抽在 `common/chat.h` + `common_sampler`，llama-server 只是调用方。薄服务走同一套即可。

```text
请求 JSON
  → common_chat_msgs_parse_oaicompat
  → common_chat_tools_parse_oaicompat
  → common_chat_templates_apply   // 出 prompt + grammar + parser
  → KVMem prefill / retrieval / prefix reuse   // 不动
  → common_sampler（含 grammar）decode
  → common_chat_parse
  → OpenAI message / tool_calls
```

---

## 现状缺口

`llama-kvmem-server` 今天：

- 只读 `messages[].content` 字符串；丢掉 `tool_calls`、`role=tool`、`tools`
- `common_chat_templates_apply` 不传 `tools`
- 采样是 `llama_sampler` greedy/temp，无 grammar
- 响应只有 `message.content` 字符串，不 parse

因此客户端发 `tools` 等于没发；模型若吐出 tool JSON，服务端也当普通文本。

---

## 数据流（一轮 tool call）

```text
turn 1  user: "secret code 是什么？"  + tools: [lookup_code]
        → 模型受 grammar 约束，finish_reason=tool_calls
        → { name: lookup_code, arguments: "{\"q\":\"secret\"}" }

turn 2  messages = turn1 + assistant.tool_calls + {role:tool, content:"BLUEBIRD-42"}
        → prefix reuse：LCP ≈ turn1 prompt + 助手 tool 文本
        → query span = 最后一条 role=user（不是 tool JSON）
        → 模型用工具结果作答
```

Retrieval 的 query 必须是 **最后一条 user 文本**，不能是 tool 回包。tool JSON 对 mean-K · Q 没有语义。

Prefix reuse 不改：完整 messages 重新 tokenize，和 C 一样 LCP。tool 轮会在 cached 里多一段 assistant tool 文本 + tool 结果；对不齐就 GDN catch-up / fallback clear。

---

## 实现要点

### 1. 解析请求

用 `common_chat_msgs_parse_oaicompat` 替代手写 role/content 循环。补：

| 字段 | 行为 |
|---|---|
| `tools` | `common_chat_tools_parse_oaicompat` |
| `tool_choice` | `auto` / `required` / `none` / `{"type":"function","function":{"name"}}` |
| `parallel_tool_calls` | 传给 templates；模板不支持则忽略 |
| `response_format` / `json_schema` / `grammar` | 与 tools 互斥（llama-server 同款：有 tools 时禁止自定义 grammar） |
| `stop` | 并入模板 `additional_stops` |

`kvmem` 对象（query_begin、force_substr、enable_thinking）保留。

### 2. 模板

```cpp
inputs.messages = msgs;           // 含 tool_calls / role=tool
inputs.tools = tools;
inputs.tool_choice = ...;
inputs.use_jinja = true;
inputs.enable_thinking = cr.enable_thinking;
chat = common_chat_templates_apply(tmpls, inputs);
```

Qwen3.5 的 GGUF 自带 jinja。不要为 KVMem 特写 tool 模板。

`last_user`：从后往前找 `role=="user"` 的 `content`，供 `derive_query_span`。没有 user（只有 tool 轮）则 fallback `--kvmem-query-last`。

### 3. 采样

greedy / MTP 都改走 `common_sampler`：

- `sparams.temp` 来自请求
- `chat.grammar` 非空 → `sparams.grammar = {COMMON_GRAMMAR_TYPE_*, chat.grammar}`
- `grammar_lazy` + `grammar_triggers` 原样传（llama-server 的 lazy tool grammar）

MTP：`kvmem-spec.cpp` 已用 `common_sampler_sample_and_accept_n`。把同一套 `sparams` 传进去即可；grammar 失败则该轮 MTP 不可用、回退 greedy（不要为此改 `speculative.cpp`）。

### 4. 解析输出

非流式：整段生成文本 → `common_chat_parse(text, /*partial*/false, parser_params)`。

- `msg.tool_calls` 非空 → `finish_reason = "tool_calls"`，`message.content` 可空
- 否则 `finish_reason = "stop"`，`content` 为解析后的文本（thinking 按 `enable_thinking` / reasoning_format）

流式：每段增量 `common_chat_parse(acc, true, ...)` + `common_chat_msg_diff::compute_diffs`，按 llama-server `server-task.cpp` 发 `delta.tool_calls` / `delta.content`。结束再 parse 一次 `partial=false`。

`parser_params` 从 `common_chat_params` 填：`format`、`generation_prompt`、`parse_tool_calls=true`。

### 5. KVMem 交互（不要改错）

| 点 | 规则 |
|---|---|
| prefill / retrieval / pin / query replay | 仍在 `run_prefill_retrieval` |
| query span | 最后一条 **user**，不是最后一条 message |
| prefix reuse | 完整 tokenize 后 LCP；tool 轮允许 LCP < cached（C 的 GDN catch-up） |
| decode mean-K | 接受的 token 照旧；tool 文本也要进 mean，否则下一轮 ranker 看不到助手/工具段 |
| 不 pin 的短请求 | 与现在相同，`query_begin==0` 不 retrieval |

### 6. CLI

`llama-kvmem-cli` 仍是单 prompt，不做 tools。协议只在 server。

---

## 顺序

```text
T1  解析 messages/tools（模板能看见 tool 历史）
T2  grammar + common_sampler（约束生成）
T3  非流式 parse + OpenAI tool_calls 响应
T4  流式 delta.tool_calls
T5  query=last user；prefix reuse 跑通 tool 第二轮
T6  可选：response_format / json_schema（同一 grammar 管线）
```

T1 不做 T2 则模型可能乱说话但历史能进模板。T2 不做 T3 则客户端拿不到结构化 `tool_calls`。T5 依赖 C。

**T1 0.8B / 5050（2026-09-07）：** `chat_parse n_msg=3 n_tools=1 tool_choice=auto tool_hist=2 prompt_has_tool=1 grammar_bytes=1067`；tools+grammar → 400；retrieval / prefix reuse 未回退。

**T2 0.8B / 5050（2026-09-07）：** `chat_sample grammar_type=tool_calls lazy=0`；`tool_choice=required` 生成 `<tool_call> <function=lookup_code> ...`；无 tools 回归未回退。响应仍是 `message.content` 字符串（T3 再 parse）。

**T3 0.8B / 5050（2026-09-07）：** `finish_reason=tool_calls`；`tool_calls[0].function.name=lookup_code`，`arguments="{\"q\":\"secret code\"}"`，`id=call_1`；retrieval / prefix reuse 未回退。流式仍走 content delta（T4）。

**T4 0.8B / 5050（2026-09-07）：** stream `delta.tool_calls` 出现，`name=lookup_code`，末包 `finish_reason=tool_calls`，`chat_stream n_tc_delta=6`；无 tools 的 stream 回归未回退。

**T5 0.8B / 5050（2026-09-07）：** turn-2 `role=tool` + `tool_choice=none`；`prefix_reuse reused=1 n_past=312 n_prompt=336 query=[260,277)`（`q1 < n_prompt`）；`query_q_capture recapture=1`；`retrieval_protect suffix=[277,312)`；`finish_reason=stop`；content **BLUEBIRD-42**。无 tools 的 retrieval / prefix reuse 未回退（turn-2 `query=[869,882)` 同样止于 last user）。

T6 可与 T2 一起做（`inputs.json_schema` / `response_format`），不挡 T3 门。

---

## 测试（0.8B / 5050，`--no-think`）

模型：`Qwen3.5-0.8B-Q8_0`。速度不当 GO/NO-GO。召回 miss、MTP 0%、`tool_calls` 解析失败算失败。

| 项 | 过线 |
|---|---|
| 无 tools 回归 | `server_smoke`：greedy / stream / retrieval BLUEBIRD / turn-2 prefix reuse |
| `tool_choice=required` | 响应 `finish_reason=tool_calls`，`tool_calls[0].name` 为所给函数名，`arguments` 为 JSON 对象字符串 |
| tool 第二轮 | 客户端回 `role=tool`；`prefix_reuse reused=1`；最终 content 含工具返回值（如 BLUEBIRD-42） |
| `tool_choice=none` | 不出现 `tool_calls` |
| stream | 至少一段 `delta.tool_calls`，结尾 `[DONE]` |
| 无 tools 的 retrieval | 行为与 C 相同 |

`tool_choice=required` 应触发模板 grammar，0.8B 只要格式对即可，不要求参数语义完美。

建议在 `scripts/server_smoke.py` 加 tools 段，或单独 `scripts/tools_canary.py`，不要把 llama-server 测试搬过来。

---

## 关键决定

1. **留在 `llama-kvmem-server`。** 并 llama-server 会重做 C，且与「先独立薄服务」冲突。
2. **只复用 `common/chat` + `common_sampler`。** 不手写 Qwen tool parser。
3. **服务端不跑工具。** 只做协议。
4. **retrieval query = 最后 user。** tool 回包不当 query。
5. **MTP 不为此改 FA / speculative.cpp。** grammar 能进现有 spec sampler 就用；不行就该请求 greedy。
6. **阶段 D 仍可选、仍后置。** 工具调用不依赖落盘。

---

## 与 A–D 的关系

```text
A packed K/V
B decode mean-K          ← tool 轮生成的 token 也要进 mean
C server 前缀复用        ← tool 第二轮靠完整 messages
T 本计划（chat tools）
D state_write 落盘       ← 仍可选，T 不依赖 D
```
