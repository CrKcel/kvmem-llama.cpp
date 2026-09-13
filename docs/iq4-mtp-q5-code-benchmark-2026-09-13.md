# IQ4 MTP / Q5 KV 代码生成测速

2026-09-13，在 RTX 5060 Ti 上完成一次无缓存命中的约 61K token 代码请求。
默认启动脚本：[scripts/start-iq4.sh](../scripts/start-iq4.sh)。

模型为 `Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf`，主权重 IQ4_XS，
MTP 权重 Q4_0；主模型及 MTP KV 均为 Q5_0，草稿长度 2。
GPU UUID：`GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`。

```text
-c 262144 -n 16000 -b 512 -ngl 99
--kvmem --kvmem-method retrieval
--kvmem-budget 32000 --kvmem-gen-reserve 12000
--kvmem-block-tokens 128 --kv-dtype q5_0
--spec-type draft-mtp --spec-draft-n-max 2
--enable-thinking --reasoning-budget 4096
```

实际预算 32000、生成预留 11904（向下按 128-token 块对齐），主/草稿池均为 43904 tokens。
保留原来的最大输出 16000 和思考预算 4096。

输入沿用 `logs/kvmem_prog_60k.txt` 中的 C++ 项目片段及线程安全 LRU 实现任务，
去掉原始 ChatML 包装后通过 `/v1/chat/completions` 发送单条 user 消息。
请求使用流式输出、temperature=0，其余由启动默认值决定。

| 指标 | 结果 |
| --- | ---: |
| 输入 tokens | 61159 |
| 缓存命中 tokens | 0 |
| Prefill 完整耗时 | 116.097 s |
| Prefill 有效吞吐 | 526.79 tok/s |
| 首 token 延迟（客户端） | 116.965 s |
| 生成 tokens（含思考） | 2096 |
| Decode 完整耗时 | 66.664 s |
| Decode 吞吐 | 31.44 tok/s |
| MTP 接受率 | 75.9%（1265 / 1666） |
| 客户端请求总耗时 | 182.953 s |
| 启动就绪显存 | 15167.10 MiB |
| Prefill 采样峰值显存 | 15687.10 MiB |
| Decode 采样峰值显存 | 15687.10 MiB |
| 最低可用显存 | 364.90 MiB |
| 请求结束后显存 | 15687.10 MiB |

Prefill 为服务端 `KVMEM_CHAT_PREFILL` 的端到端耗时，包含 MTP 跟随、检索及查询重放。
Decode 使用 `KVMEM_GEN_WALL`，包括思考与最终代码 token，不使用会混入 MTP 验证的普通 eval 计数。
检索后主/草稿工作集为 250 块，MTP 保留 184 块、从主机恢复 66 块，`n_no_raw=0`。
请求以 `finish_reason=stop` 正常结束，未发生 OOM。

显存从加载前开始通过 NVML v2 采样，共 3991 次，无查询错误；目标间隔 50 ms，最大实际间隔 114.16 ms。
统计为整张 5060 Ti 的 used memory，加载前为 0 MiB，不含 259 MiB 驱动预留；
`nvidia-smi` 的整数显示为 15688 MiB。此处为采样峰值，短于采样间隔的瞬时峰值可能漏测。
这是一次 61K 输入、2096 输出的测试，不能代表完整 256K 输入或 12K 生成的峰值。

生成代码的 C++17 编译检查未通过：`get()`、`put()` 的 `list::splice` 第三个参数误用了 key，
应传入 list iterator。原始输出保留，没有人工修正；以上速度为本次原始生成的测量值。

完整证据：[结果 JSON](../logs/iq4_mtp_q5_32k12k_code_20260913_191415/result.json)、
[服务日志](../logs/iq4_mtp_q5_32k12k_code_20260913_191415/server.stderr.log)、
[显存采样 CSV](../logs/iq4_mtp_q5_32k12k_code_20260913_191415/vram.csv)、
[生成代码](../logs/iq4_mtp_q5_32k12k_code_20260913_191415/generated.cpp)、
[编译日志](../logs/iq4_mtp_q5_32k12k_code_20260913_191415/compile.log)。
同目录保存了请求、SSE 响应、思考内容及测速脚本快照。

服务保持运行在 `http://127.0.0.1:18200`。使用默认脚本可检查/启动，使用 `--restart` 重启。
